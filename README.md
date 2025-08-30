# Fine-Tuning versus Retrieval-Augmented Generation in Large Language Models: A Comparative Study on Dungeons & Dragons

This repository contains the full source code, data, and analysis for the scientific paper titled "Fine-Tuning versus Retrieval-Augmented Generation in Large Language Models: A Comparative Study on Dungeons & Dragons".

This project conducts a comparative study between fine-tuning (FT), retrieval-augmented generation (RAG), and a hybrid approach (FT+RAG) to adapt Large Language Models for the complex, rule-dense domain of Dungeons & Dragons 5th Edition.

**Fine-Tuned Model**: The final fine-tuned model is available on Hugging Face at [astrevallion/Qwen3-14B-FT](https://huggingface.co/astrevallion/Qwen3-14B-FT).

## Project Overview

The core research question is: **Which adaptation strategy delivers the most reliable and inspiring D&D assistant when both GPU memory and response time are strictly limited?**

To answer this, we evaluate four model configurations:
1.  **BASE**: The vanilla `gpt-oss:20b` and `qwen3:14b-reasoning` models.
2.  **FT**: The `qwen3:14b-reasoning` model fine-tuned on the D&D Systems Reference Document (SRD).
3.  **RAG**: The `gpt-oss:20b` model augmented with a LightRAG-powered retrieval system.
4.  **FT+RAG**: The fine-tuned `qwen3:14b-reasoning` model augmented with the same retrieval system.

Evaluation is performed using a comprehensive suite of automated metrics from the Ragas framework, as well as a double-blind qualitative analysis by both a domain expert and an LLM-as-expert.

## Repository Structure

The repository is organized to clearly separate data, source code, analysis scripts, and results:

-   `analysis/`: Python scripts to summarize metrics of the results and generating plots.
-   `data/`: Contains all raw, processed, and synthetic data used in the experiments.
-   `results/`: All generated outputs from the experiments, including answer sets used for evaluation scores.
-   `src/`: Core Python source code for data preprocessing, answer generation, and evaluation.
-   `requirements.txt`: A list of most Python dependencies.

## Setup & Installation

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ErcagCan/ft_vs_rag_dnd.git
    cd ft_vs_rag_dnd
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Ollama**: This project requires a running Ollama instance with the necessary models pulled. Please refer to the [Ollama documentation](https://ollama.ai/) for installation instructions.

## Replicating the Experiment on Windows
The entire experimental workflow can be replicated by running the the following scripts for their respective parts in order:

# Fine-Tuning

1.  **Prepare the Data**: Run the srd preprocessing and augment script to clean the SRD and generate the augmented dataset for fine-tuning.
    ```bash
    python src/srd_preprocessing.py
    python src/augment.py
    ```

2.  **Fine-Tune the Model**: Open and run the `src/fine_tuning.ipynb` notebook in Google Collab to perform the QLoRA fine-tuning. In the stage of this project, I got it to work locally on Windows as well, but had to endure a lot of unnecessary headache, since this notebook is mainly meant for Linux distros. It is perfectly fine and possible to fine tune it multiple times in the same session for free on Google Collab, without any prerequisites and out of the box. You can save it to Hugging Face with your own credentials and hf token at the end of the notebook.

3.  **Generate the Synthetic Test Set**: Run the Ragas STS generation and the single sts jsonl file creation script, to create the `combined_sts.jsonl` file used for Ragas evaluation. Run questions from json script, to create one questions.txt file which will be used to create qa pairs for the 4 configurations.
    ```bash
    python src/ragas_sts_generation.py
    python src/create_single_sts_jsonl_file.py
    python src/questions_from_json.py
    ```

4.  **Generate Answers**: Run the answer generation script to query all four model configurations and produce the `qa_*.jsonl` files.
    ```bash
    python src/rag_sdr_and_generate_answers.py
    ```

5.  **Generate Evaluation Outputs**: Execute the evaluation scripts to perform the Ragas evaluation and LLM-as-expert evaluation. Make sure to add an expert evaluated equivalent of the output in the results/llm_evaluation folder to the results/expert_evaluation folder.
    ```bash
    python src/ragas_eval.py
    python src/llm_eval.py
    ```

6.  **Generate Analysis Outputs**: Execute the scripts to generate all plots and statistical summaries.
    ```bash
    python analysis/latency_analysis.py
    python analysis/likert_grading_analysis.py
    python analysis/likert_comparative_analysis.py
    python analysis/ragas_analysis.py
    ```

## Key Findings

Our findings show conclusively that **RAG is the single most impactful strategy for factual accuracy**.

-   The **RAG** model achieved an **81% improvement in expert-judged correctness** over the baseline.
-   Standalone **Fine-Tuning (FT)** provided no significant gains in correctness.
-   The **Hybrid (FT+RAG)** model yielded the highest semantic similarity to ground-truth answers, demonstrating a synergistic effect where the stylistic alignment from FT complements the factual grounding from RAG.
-   A key takeaway is that a powerful base LLM can successfully reason over a broad, high-recall context, providing accurate answers even when specific automated metrics like Context Relevance score low due to evaluation artifacts.

Even a test run with only the Wizard.txt excerpt, reveals very similar results, with only 4 Q/A pairs generated by the Ragas STS generation. The main difference with this test run is that we actually achieve Response Groundedness and Context Relevance scores which are around 0.9 and 0.8. These scores were around 0.1 and 0.2 approximately in our large scale full SRD run for the main experiment.

For a detailed breakdown of all metrics, please refer to our paper.
