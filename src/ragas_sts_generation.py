import typing as t


from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor, apply_transforms
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.base import BaseSynthesizer


QueryDistribution = t.List[t.Tuple[BaseSynthesizer, float]]


# Setup Local Models
ollama_llm = ChatOllama(model="gpt-oss:20b")
ragas_llm = LangchainLLMWrapper(ollama_llm)
ollama_embeddings = OllamaEmbeddings(model="bge-m3:latest")
ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
print("\nModels loaded successfully")


# Define Personas and Reusable Components
persona_new_dm = Persona(
    name="Novice",
    role_description=(
        "Familiar with D&D basics but still building confidence. "
        "Often needs step‑by‑step explanations of mechanics, quick lore references, "
        "and simple sanity‑checks for rules that are rarely used."
    ),
)
persona_expert_dm = Persona(
    name="Veteran",
    role_description=(
        "Has run many campaigns and wants fast, authoritative answers. "
        "Prefers concise rule citations, subtle lore nuances, and quick fact‑checks "
        "without long introductions."
    ),
)
persona_story_dm = Persona(
    name="Narrativeist",
    role_description=(
        "Runs campaigns driven by lore and setting. "
        "Requests contextual background, flavor details, and creative usage of rules "
        "to enhance story, rather than just the mechanics."
    ),
)
personas = [persona_new_dm, persona_expert_dm, persona_story_dm]


# Extractors, Splitter and Synthesizer setup
headline_extractor = HeadlinesExtractor(llm=ragas_llm, max_num=20)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=ragas_llm)
transforms = [headline_extractor, headline_splitter, keyphrase_extractor]
query_distribution: QueryDistribution = [
    (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, property_name="headlines"), 0.5),
    (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, property_name="keyphrases"), 0.5),
]


# We will update the generator knowledge_graph attribute inside the loop
generator = TestsetGenerator(
    llm=ragas_llm,
    embedding_model=ragas_embeddings,
    persona_list=personas,
)


# Define the Directory and Loop Through Files
srd_directory = Path("./data/synthetic_test_set/source_excerpts/")
for file_path in srd_directory.glob("*.txt"):
    file_name = file_path.stem
    print(f"\n--- Processing: {file_path.name} ---")

    try:
        # Read file and create a fresh KnowledgeGraph for it
        text = file_path.read_text(encoding="utf-8")
        docs = [Document(page_content=text)]
        
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
                )
            )
        print("\nKG created")

        # Apply the pre-defined transforms to the current KG
        apply_transforms(kg, transforms=transforms)

        # Update the generator with the new knowledge graph for this file
        generator.knowledge_graph = kg
        print(f"\nDocument processed for '{file_name}'")

        # Generate the test set using the updated generator
        testset = generator.generate(testset_size=4, query_distribution=query_distribution)
        print(f"\nSynthetic test set generated for '{file_name}'")

        # Save the results
        df = testset.to_pandas()
        output_filename = f"data/synthetic_test_set/generated_sts_files/synthetic_test_data_{file_name}.json"
        df.to_json(output_filename, force_ascii=False, indent=4, orient="records", lines=True)
        print(f"\nResults saved to '{output_filename}'")

    except Exception as e:
        print(f"\nFailed to process {file_path.name}: {e}")
print("\n\nAll files processed successfully!")