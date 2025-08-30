from pydantic_settings import BaseSettings
from typing import List, Dict, Any


class AppConfig(BaseSettings):
    """ Main configuration class for the application. Holds paths, Ollama settings, prompts, and a list of all model configurations to run."""


    # General Paths and Settings
    WORKDIR: str = "./data/processed/question_answer_pairs/pdf_rag_store"
    TEXT_FILE: str = "./data/raw/SRD.txt"
    QUESTIONS_FILE: str = "./data/processed/question_answer_pairs/questions.txt"
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "bge-m3:latest"
    EMBEDDING_DIM: int = 1024
    MAX_TOKEN_SIZE: int = 8192


    # Prompts
    SYSTEM_PROMPT_RAG: str = """
    You are a professional assistant with deep knowledge of Dungeons & Dragons.
    When a question is posed, first query the D&D documentation.
    Quote any relevant excerpts verbatim, then add a concise, self-contained answer.
    If the docs contain no answer, say "not covered in the documents" and give your best informed reply.
    Keep the response clear, correct, and as brief as possible.
    Question: """
    SYSTEM_PROMPT_BASE: str = """
    You are a professional assistant and Dungeons & Dragons expert.
    Answer the user’s question using your comprehensive knowledge of the game’s rules, lore, and mechanics.
    If you are unsure about a detail, state the uncertainty and still provide the best answer you can.
    Make your answer correct, complete, and concise.
    Question: """


    # Model Configurations to Process
    MODEL_CONFIGS: List[Dict[str, Any]] = [
        {
            "name": "base",
            "model_name": "gpt-oss:20b",
            "output_file": "./data/processed/question_answer_pairs/qa_base.jsonl",
            "is_rag": False,
        },
        {
            "name": "rag",
            "model_name": "gpt-oss:20b",
            "output_file": "./data/processed/question_answer_pairs/qa_rag.jsonl",
            "is_rag": True,
        },
        {
            "name": "ft",
            "model_name": "hf.co/astrevallion/Qwen3-14B-FT:Q4_K_M",
            "output_file": "./data/processed/question_answer_pairs/qa_ft.jsonl",
            "is_rag": False,
        },
        {
            "name": "ft_rag",
            "model_name": "hf.co/astrevallion/Qwen3-14B-FT:Q4_K_M",
            "output_file": "./data/processed/question_answer_pairs/qa_ft_rag.jsonl",
            "is_rag": True,
        },
    ]


settings = AppConfig()
