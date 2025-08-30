import asyncio
import aiofiles
import ollama
import json
import time
import re
import os


from typing import Callable, Awaitable, List, Dict, Any
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from config import settings


setup_logger("lightrag", level="INFO")


def remove_think_tags(text: str) -> str:
    cleaned_text = re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def _extract_contexts(rag_response_str: str) -> List[str]:
    """ Parses the raw string output from the RAG model to extract just the document chunk content for Ragas evaluation."""
    try:
        dc_marker = "-----Document Chunks(DC)-----"
        if dc_marker not in rag_response_str:
            print("\nNo document chunks found in response.")
            return []
        json_str_part = rag_response_str.split(dc_marker)[1]
        json_str = json_str_part.split("```json")[1].split("```")[0].strip()
        doc_chunks = json.loads(json_str)
        return [chunk.get("content", "") for chunk in doc_chunks]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"\nError parsing context from response: {e}")
        return [rag_response_str]


def _clean_answer(answer_str: str) -> str:
    """ Cleans the raw answer string."""
    main_content = answer_str.split("Reference")[0]
    if main_content.strip().startswith("**Answer**"):
        main_content = main_content.replace("**Answer**", "", 1).strip()
    return main_content.strip()


async def build_rag_once(model_name: str) -> LightRAG:
    """ Builds and initializes a LightRAG instance for a specific model."""
    print(f"\nBuilding RAG index for model: {model_name}...")
    rag = LightRAG(
        working_dir=settings.WORKDIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_kwargs={"host": settings.OLLAMA_HOST},
        embedding_func=EmbeddingFunc(
            embedding_dim=settings.EMBEDDING_DIM,
            max_token_size=settings.MAX_TOKEN_SIZE,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=settings.EMBEDDING_MODEL,
                host=settings.OLLAMA_HOST,
            ),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Ingest data only if the index doesn't already exist
    if not (Path(settings.WORKDIR) / "vdb_chunks.json").exists():
        print("\nNo existing index found. Ingesting SRD text file...")
        text_file_path = settings.TEXT_FILE
        if not Path(text_file_path).exists():
            raise FileNotFoundError(f"No text file found at {text_file_path}")
        text = Path(text_file_path).read_text(encoding="utf-8")
        await rag.ainsert(input=text, split_by_character=None)
        print("\nIngestion complete.")
    else:
        print("\nExisting RAG index found. Skipping ingestion.")
        
    return rag


async def ask_rag(rag: LightRAG, question: str, top_k: int = 20) -> Dict[str, Any]:
    full_prompt = f"{settings.SYSTEM_PROMPT_RAG}\n{question}"
    t_retrieval_start = time.perf_counter()
    param_context = QueryParam(mode="mix", top_k=top_k, only_need_context=True)
    resp_context = await rag.aquery(full_prompt, param=param_context)
    t_retrieval_end = time.perf_counter()
    retrieval_time = t_retrieval_end - t_retrieval_start
    contexts = _extract_contexts(resp_context)
    t_query_start = time.perf_counter()
    param = QueryParam(mode="mix", top_k=top_k)
    resp = await rag.aquery(full_prompt, param=param)
    t_query_end = time.perf_counter()
    query_time = t_query_end - t_query_start
    answer = _clean_answer(remove_think_tags(resp))
    total_time = retrieval_time + query_time
    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "retrieval_time": retrieval_time,
        "query_time": query_time,
        "total_time": total_time,
    }


async def ask_base(client: ollama.AsyncClient, question: str, model_name: str) -> Dict[str, Any]:
    t_start = time.perf_counter()
    resp = await client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": settings.SYSTEM_PROMPT_BASE},
            {"role": "user", "content": question}
        ]
    )
    t_end = time.perf_counter()
    query_time = t_end - t_start
    raw_answer = resp['message']['content']
    cleaned_answer = remove_think_tags(raw_answer)
    return {"answer": cleaned_answer, "query_time": query_time}


async def process_questions(
    question_file: str,
    output_file: str,
    answer_func: Callable[[str], Awaitable[Dict[str, Any]]],
    config_name: str
):
    """ Generic function to process a list of questions and save the outputs."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(question_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in await f.readlines() if line.strip()]

    async with aiofiles.open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, start=1):
            print(f"\nProcessing for {config_name.upper()} ({i}/{len(questions)}): {q}")
            result_data = await answer_func(q)
            json_output = {"question": q, **result_data}
            await out.write(json.dumps(json_output, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(questions)} answers to {output_file}")


async def main():
    """ Orchestrates the answer generation for all configured models."""
    client = ollama.AsyncClient(host=settings.OLLAMA_HOST)
    rag_instance = None # We only need one RAG instance since the source doc is the same

    for config in settings.MODEL_CONFIGS:
        model_name = config["name"]
        llm_model = config["model_name"]
        output_file = config["output_file"]
        is_rag = config["is_rag"]

        print(f"\n--- Running configuration: {model_name.upper()} ---")

        if is_rag:
            
            # Clear the cache before running any RAG configuration, otherwise an old answer will be fetched
            cache_file = Path(settings.WORKDIR) / "kv_store_llm_response_cache.json"
            if cache_file.exists():
                print(f"\nClearing RAG cache file: {cache_file}")
                os.remove(cache_file)

            if rag_instance is None:
                # Build RAG instance for the first RAG model encountered.
                # Assumes the same base retriever is used for all RAG configs.
                rag_instance = await build_rag_once(llm_model)
            else:
                # For subsequent RAG models, just update the LLM model name
                rag_instance.llm_model_name = llm_model
                print(f"\nUpdated RAG instance to use model: {llm_model}")
                
            await process_questions(
                settings.QUESTIONS_FILE,
                output_file,
                lambda q: ask_rag(rag_instance, q),
                model_name
            )
        else: # Not a RAG model
            await process_questions(
                settings.QUESTIONS_FILE,
                output_file,
                lambda q: ask_base(client, q, llm_model),
                model_name
            )
            
    if rag_instance:
        await rag_instance.finalize_storages()
        print("\nFinalized RAG storages.")


if __name__ == "__main__":
    asyncio.run(main())
