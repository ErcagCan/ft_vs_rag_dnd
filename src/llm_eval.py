import json
import re
import requests
import time
import os


from pathlib import Path


# Configuration
CONTEXT_FOLDER = "./data/synthetic_test_set/source_excerpts"
QUESTIONS_FOLDER = "./data/synthetic_test_set/generated_sts_files"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
PROMPT_PREAMBLE = """
You are a meticulous Dungeons and Dragons rules editor. Your task is to evaluate the provided Answer based on the Question and the official Context from the System Reference Document.

First, respond with ONLY the completed evaluation template below.
Then, after the template, provide a brief, one-sentence reasoning summary for your scores, starting your summary with the label "Reasoning:".

**Evaluation Metrics:**
- **Relevance**: How well does the answer address the specific question? (1 = Off-topic, 5 = Perfectly on-topic)
- **Correctness**: Is the information factually accurate according to the Context? (1 = Completely inaccurate, 5 = Flawlessly accurate)
- **Creativity**: Does the answer synthesize or format the information in a uniquely clear and helpful way (e.g., using tables, lists, or summaries)? (1 = Just copied text, 5 = Exceptionally clear and well-presented)
- **Completeness**: Does the answer include all the necessary details from the Context to fully address the question? (1 = Missing critical information, 5 = Fully comprehensive)

Evaluation Template:
\"\"\"
Relevance: 4
Correctness: 4
Creativity: 3
Completeness: 5
\"\"\"
"""


def load_data_sources(evaluation_file):
    """Loads a specific .jsonl file into a question-response map."""
    eval_path = Path(evaluation_file)
    if not eval_path.is_file():
        print(f"\nError: Evaluation file not found at {evaluation_file}")
        return None
    results_map = {}
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Ensure both keys exist before adding to the map
                if 'question' in data and 'answer' in data:
                    results_map[data['question']] = data['answer']
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"\nSuccessfully loaded {len(results_map)} answer pairs from {evaluation_file}.")
    return results_map


def query_ollama(prompt):
    """Sends a prompt to the Ollama API and returns the response text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Error: Could not connect to Ollama. Details: {e}")
        return None


def parse_llm_response(response_text):
    """Parses the LLM's text response to extract scores and reasoning."""
    try:
        scores = {}
        metrics = ["Relevance", "Correctness", "Creativity", "Completeness"]
        for metric in metrics:
            match = re.search(rf"{metric}:\s*([1-5])", response_text, re.IGNORECASE)
            scores[metric.lower()] = int(match.group(1)) if match else None

        reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.DOTALL)
        scores["reasoning"] = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
        
        if all(v is not None for k, v in scores.items() if k != "reasoning"):
            return scores
        else:
            return None
    except Exception:
        return None


def run_evaluation_for_model(config):
    """Main script to run the full evaluation pipeline for one model configuration."""
    model_name = config['name']
    evaluation_file = config['input']
    output_file = config['output']
    
    print(f"\n--- Starting evaluation for {model_name.upper()} model ---")

    results_map = load_data_sources(evaluation_file)
    if not results_map:
        print(f"\nCould not load data. Skipping {model_name.upper()}.")
        return

    questions_dir = Path(QUESTIONS_FOLDER)
    context_dir = Path(CONTEXT_FOLDER)
    if not all([questions_dir.is_dir(), context_dir.is_dir()]):
        print("\nError: Make sure source data folders exist.")
        return

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        pass # Clear the file
    print(f"\nOutput will be saved to '{output_file}'.")

    total_evaluated = 0
    for json_file in sorted(list(questions_dir.glob("*.json"))):
        base_name = json_file.stem
        context_key = base_name.split('_')[-1]
        context_file = context_dir / f"{context_key}.txt"

        if not context_file.is_file(): continue
        context_text = context_file.read_text(encoding='utf-8')

        with open(json_file, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
        
        # Handle potentially malformed JSON files with multiple objects
        json_objects = re.findall(r'{.*?}', content, re.DOTALL)

        for obj_str in json_objects:
            try:
                data = json.loads(obj_str)
                question = data.get('user_input')

                if question and question in results_map:
                    answer = results_map[question]
                    print(f"\nEvaluating question: '{question[:60]}...'")

                    for attempt in range(MAX_ATTEMPTS):
                        dynamic_part = f"Question:\n\"\"\"\n{question}\n\"\"\"\n\nAnswer:\n\"\"\"\n{answer}\n\"\"\"\n\nContext:\n\"\"\"\n{context_text}\n\"\"\""
                        full_prompt = PROMPT_PREAMBLE.strip() + "\n\n\n" + dynamic_part.strip()
                        llm_response = query_ollama(full_prompt)

                        if llm_response:
                            parsed_data = parse_llm_response(llm_response)
                            if parsed_data:
                                parsed_data["question"] = question
                                with open(output_file, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(parsed_data, ensure_ascii=False) + '\n')
                                total_evaluated += 1
                                break
                        
                        print(f"\nWarning: Could not parse response on attempt {attempt + 1}/{MAX_ATTEMPTS}.")
                        if attempt < MAX_ATTEMPTS - 1:
                            print(f"\nRetrying in {RETRY_DELAY_SECONDS} seconds...")
                            time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"\nERROR: Failed to get a valid response for this question after {MAX_ATTEMPTS} attempts. Skipping.")
            except json.JSONDecodeError:
                continue

    print(f"\nPipeline finished for {model_name.upper()}. Total evaluations saved: {total_evaluated}.")


def main():
    """Orchestrates the evaluation for all model configurations."""
    
    # Configuration for all models to be processed
    configs = [
        {
            "name": "base",
            "input": "./data/processed/question_answer_pairs/qa_base.jsonl",
            "output": "./results/llm_evaluation/llm_evaluations_base.jsonl"
        },
        {
            "name": "rag",
            "input": "./data/processed/question_answer_pairs/qa_rag.jsonl",
            "output": "./results/llm_evaluation/llm_evaluations_rag.jsonl"
        },
        {
            "name": "ft",
            "input": "./data/processed/question_answer_pairs/qa_ft.jsonl",
            "output": "./results/llm_evaluation/llm_evaluations_ft.jsonl"
        },
        {
            "name": "ft_rag",
            "input": "./data/processed/question_answer_pairs/qa_ft_rag.jsonl",
            "output": "./results/llm_evaluation/llm_evaluations_ft_rag.jsonl"
        },
    ]

    for config in configs:
        run_evaluation_for_model(config)

    print("\n\nAll LLM evaluations are complete.")


if __name__ == "__main__":
    main()
