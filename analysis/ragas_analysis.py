import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_jsonl(filepath):
    """Loads a .jsonl file into a list of dictionaries, handling potential errors."""
    if not os.path.exists(filepath):
        print(f"\nError: File not found at {filepath}")
        return None
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"\nWarning: Skipping invalid JSON on line {line_num} in {filepath}")
    return data


def create_plots(df, metric_map, rag_only_metrics, model_order, plot_type='bar'):
    """Generates and saves a separate plot for each metric. Can create 'bar', 'box', or 'violin' plots."""
    plot_functions = {
        'bar': sns.barplot,
        'box': sns.boxplot,
        'violin': sns.violinplot
    }
    
    if plot_type not in plot_functions:
        print(f"\nError: Invalid plot_type '{plot_type}'. Choose from 'bar', 'box', 'violin'.")
        return

    print(f"\nGenerating separate {plot_type} charts...")
    for raw_metric, clean_metric in metric_map.items():
        plt.figure(figsize=(10, 8))
        
        metric_data = df[df['metric_type'] == raw_metric]
        if metric_data.empty:
            print(f"\nSkipping '{clean_metric}' - No data found.")
            continue

        current_model_order = model_order
        if raw_metric in rag_only_metrics:
            current_model_order = ['RAG', 'FT+RAG']

        plot_params = {
            'data': metric_data, 'x': 'model', 'y': 'score',
            'order': current_model_order, 'palette': 'viridis'
        }
        title_prefix = 'Mean Score'
        ylabel = 'Mean Score (0-1 Scale)'
        ylim = (0, 1.05)
        
        if plot_type in ['box', 'violin']:
            title_prefix = 'Score Distribution'
            ylabel = 'Score (0-1 Scale)'
            ylim = (-0.05, 1.05)
            if plot_type == 'violin':
                plot_params['inner'] = 'quartile'
        elif plot_type == 'bar':
             plot_params.update({'errorbar': 'sd', 'capsize': 0.1})

        ax = plot_functions[plot_type](**plot_params)
        ax.set_title(f'{title_prefix} for "{clean_metric}"', fontsize=18, pad=20)
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=12)

        if plot_type == 'bar':
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)
            
        filename = f"./analysis/plots_and_summaries/ragas/plot_{plot_type}_{clean_metric.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"\nSaved {filename}")
        plt.close()


# Configuration
base_path = './results/ragas_evaluation'
output_summary_file = './analysis/plots_and_summaries/ragas/ragas_metrics_summary.txt'
files_to_process = {
    'BASE': os.path.join(base_path, 'ragas_evaluation_results_base.jsonl'),
    'RAG': os.path.join(base_path, 'ragas_evaluation_results_rag.jsonl'),
    'FT': os.path.join(base_path, 'ragas_evaluation_results_ft.jsonl'),
    'FT+RAG': os.path.join(base_path, 'ragas_evaluation_results_ft_rag.jsonl')
}
metric_map = {
    "answer_relevancy": "Answer Relevancy",
    "semantic_similarity": "Semantic Similarity",
    "context_recall": "Context Recall",
    "nv_context_relevance": "Context Relevance",
    "nv_response_groundedness": "Response Groundedness",
    "bleu_score": "BLEU Score",
    "rouge_L_f1(mode=fmeasure)": "ROUGE-L F1",
    "rouge_L_precision(mode=precision)": "ROUGE-L Precision",
    "rouge_L_recall(mode=recall)": "ROUGE-L Recall"
}
rag_only_metrics = ["context_recall", "nv_context_relevance", "nv_response_groundedness"]
all_records = []


# Data Loading and Processing 
for model_name, filepath in files_to_process.items():
    data = load_jsonl(filepath)
    if data is None: continue
    print(f"\nProcessing {len(data)} records from {filepath}...")
    for entry in data:
        for raw_metric in metric_map.keys():
            if raw_metric in entry and entry[raw_metric] is not None:
                all_records.append({
                    'model': model_name,
                    'metric_type': raw_metric,
                    'score': float(entry[raw_metric])
                })
if not all_records:
    print("\nNo data processed. Exiting.")
else:
    # Create DataFrame
    df = pd.DataFrame(all_records)
    model_order = ['BASE', 'FT', 'RAG', 'FT+RAG']
    
    # Map raw metric names to clean names for a readable summary
    df['clean_metric_type'] = df['metric_type'].map(metric_map)
    
    summary = df.groupby(['clean_metric_type', 'model'])['score'].describe()
    
    # Define the order for the summary file
    clean_metric_order = list(metric_map.values())
    summary = summary.reindex(clean_metric_order, level='clean_metric_type').reindex(model_order, level='model')
    
    with open(output_summary_file, 'w') as f:
        f.write(f"--- Distribution Summary for Ragas Metrics ---\n\n")
        f.write(summary.to_string())
        
    print(f"\n\nStatistical summary saved to '{output_summary_file}'")
    print(summary)

    # Generate All 3 Types of Plots 
    create_plots(df, metric_map, rag_only_metrics, model_order, plot_type='bar')
    create_plots(df, metric_map, rag_only_metrics, model_order, plot_type='box')
    create_plots(df, metric_map, rag_only_metrics, model_order, plot_type='violin')