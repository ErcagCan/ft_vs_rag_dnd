import json
import os


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall, #rag
    ContextRelevance, #rag
    ResponseGroundedness, #rag

    answer_relevancy,
    answer_similarity,
    BleuScore,
    RougeScore,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings


def run_evaluation_for_config(config, llm, embeddings):
    """ Runs the Ragas evaluation for a single model configuration."""
    model_name = config["name"]
    model_output_file = config["input_file"]
    synthetic_data_file = config["synthetic_data_file"]
    results_output_file = config["output_file"]
    is_rag = config["is_rag"]

    print(f"\n--- Starting Ragas evaluation for: {model_name.upper()} ---")
    
    # Define Ragas Metrics based on configuration
    base_metrics = [
        answer_relevancy,
        answer_similarity,
        BleuScore(),
        RougeScore(name="rouge_L_f1", rouge_type="rougeL", mode="fmeasure"),
        RougeScore(name="rouge_L_precision", rouge_type="rougeL", mode="precision"),
        RougeScore(name="rouge_L_recall", rouge_type="rougeL", mode="recall"),
    ]
    
    rag_metrics = [
        context_recall,
        ContextRelevance(),
        ResponseGroundedness(),
    ]
    
    current_metrics = base_metrics + rag_metrics if is_rag else base_metrics

    # Ensure output directory exists
    os.makedirs(os.path.dirname(results_output_file), exist_ok=True)

    try:
        with open(model_output_file, "r", encoding="utf-8") as f_model, \
             open(synthetic_data_file, "r", encoding="utf-8") as f_synth, \
             open(results_output_file, "w", encoding="utf-8") as f_out:

            for i, (model_line, synth_line) in enumerate(zip(f_model, f_synth)):
                print(f"\nProcessing line {i + 1} for {model_name.upper()}...")

                model_data = json.loads(model_line)
                synth_data = json.loads(synth_line)

                # Prepare the data entry for Ragas
                eval_entry = {
                    "question": [model_data.get("question", "")],
                    "answer": [model_data.get("answer", "")],
                    "ground_truth": [synth_data.get("answer", "")],
                    "ground_truths": [[synth_data.get("answer", "")]],
                }

                if is_rag:
                    eval_entry["contexts"] = [model_data.get("contexts", [])]
                    eval_entry["ground_truth_context"] = [synth_data.get("contexts", [])]
                
                eval_dataset = Dataset.from_dict(eval_entry)

                # Run Evaluation on the Single Data Point
                result = evaluate(
                    dataset=eval_dataset, 
                    metrics=current_metrics, 
                    llm=llm, 
                    embeddings=embeddings
                )

                # Write the Result Immediately
                result_df = result.to_pandas()
                single_result = result_df.iloc[0].to_dict()
                
                # Add original user input for easier tracking
                single_result['user_input'] = model_data.get("question", "")
                f_out.write(json.dumps(single_result, ensure_ascii=False) + '\n')

    except FileNotFoundError as e:
        print(f"\nError: Could not find a file for {model_name.upper()}. Details: {e}")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nError: Failed to parse JSON or find a key on line {i + 1} for {model_name.upper()}. Details: {e}")
        return

    print(f"\nEvaluation complete for {model_name.upper()}!")
    print(f"\nSaved results to '{results_output_file}'")


def main():
    """ Orchestrates the Ragas evaluation pipeline for all model configurations."""
    # Define Models
    llm = ChatOllama(model="gpt-oss:20b", num_predict=8192)
    embeddings = OllamaEmbeddings(model="bge-m3:latest")

    # Define all configurations to run
    configs = [
        {
            "name": "base",
            "is_rag": False,
            "input_file": "./data/processed/question_answer_pairs/qa_base.jsonl",
            "synthetic_data_file": "./data/synthetic_test_set/combined_sts.jsonl",
            "output_file": "./results/ragas_evaluation/ragas_evaluation_results_base.jsonl",
        },
        {
            "name": "ft",
            "is_rag": False,
            "input_file": "./data/processed/question_answer_pairs/qa_ft.jsonl",
            "synthetic_data_file": "./data/synthetic_test_set/combined_sts.jsonl",
            "output_file": "./results/ragas_evaluation/ragas_evaluation_results_ft.jsonl",
        },
        {
            "name": "rag",
            "is_rag": True,
            "input_file": "./data/processed/question_answer_pairs/qa_rag.jsonl",
            "synthetic_data_file": "./data/synthetic_test_set/combined_sts.jsonl",
            "output_file": "./results/ragas_evaluation/ragas_evaluation_results_rag.jsonl",
        },
        {
            "name": "ft_rag",
            "is_rag": True,
            "input_file": "./data/processed/question_answer_pairs/qa_ft_rag.jsonl",
            "synthetic_data_file": "./data/synthetic_test_set/combined_sts.jsonl",
            "output_file": "./results/ragas_evaluation/ragas_evaluation_results_ft_rag.jsonl",
        },
    ]

    # Loop through and process each configuration
    for config in configs:
        run_evaluation_for_config(config, llm, embeddings)

    print("\n\nAll Ragas evaluations are complete.")


if __name__ == "__main__":
    main()
