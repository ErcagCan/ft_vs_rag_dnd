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


def create_plots(df, output_dir, model_order):
    """Generates and saves a separate plot for each metric type (bar, box, violin)."""
    metrics = df['metric_type'].unique()
    plot_types = ['bar', 'box', 'violin']
    plot_functions = {'bar': sns.barplot, 'box': sns.boxplot, 'violin': sns.violinplot}

    print(f"\nGenerating plots and saving to '{output_dir}'...")
    for metric in metrics:
        for plot_type in plot_types:
            plt.figure(figsize=(10, 8))
            
            metric_data = df[df['metric_type'] == metric]
            if metric_data.empty:
                continue

            plot_params = {
                'data': metric_data, 'x': 'model', 'y': 'score',
                'order': model_order, 'palette': 'viridis'
            }
            title_prefix = 'Mean Score' if plot_type == 'bar' else 'Score Distribution'
            
            if plot_type == 'bar':
                plot_params.update({'errorbar': 'sd', 'capsize': 0.1})
            elif plot_type == 'violin':
                plot_params['inner'] = 'quartile'

            ax = plot_functions[plot_type](**plot_params)
            ax.set_title(f'{title_prefix} for "{metric}"', fontsize=18, pad=20)
            ax.set_xlabel('Model Configuration', fontsize=14)
            ax.set_ylabel('Score (1-5 Scale)', fontsize=14)
            ax.set_ylim(-0.05, 5.5)
            ax.tick_params(axis='both', labelsize=12)

            if plot_type == 'bar':
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)
                
            filename = f"plot_{plot_type}_{metric.replace(' ', '_').lower()}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
    print("\nPlot generation complete.")


def process_evaluations(grader_type):
    """Processes the evaluation files for a given grader type ('expert' or 'llm'), saves a summary, and generates plots."""
    print(f"\n--- Processing {grader_type.upper()} evaluations ---")

    # Dynamically configure paths based on grader_type
    base_path = f'./results/{grader_type}_grading'
    output_plot_dir = f'./analysis/plots_and_summaries/{grader_type}_grading/'
    output_summary_file = os.path.join(output_plot_dir, f'evaluation_metrics_summary_{grader_type}.txt')
    file_prefix = f'{grader_type}_evaluations'

    os.makedirs(output_plot_dir, exist_ok=True)

    files_to_process = {
        'BASE': os.path.join(base_path, f'{file_prefix}_base.jsonl'),
        'RAG': os.path.join(base_path, f'{file_prefix}_rag.jsonl'),
        'FT': os.path.join(base_path, f'{file_prefix}_ft.jsonl'),
        'FT+RAG': os.path.join(base_path, f'{file_prefix}_ft_rag.jsonl')
    }

    # Data Loading and Processing
    all_records = []
    metrics = ['relevance', 'correctness', 'creativity', 'completeness']
    model_order = ['BASE', 'FT', 'RAG', 'FT+RAG']

    for model_name, filepath in files_to_process.items():
        data = load_jsonl(filepath)
        if data is None: continue
        print(f"\nProcessing {len(data)} records from {filepath}...")
        for entry in data:
            if not all(metric in entry for metric in metrics): continue
            
            scores = [entry[metric] for metric in metrics]
            average_score = sum(scores) / len(scores) if scores else 0
            
            for metric in metrics:
                all_records.append({
                    'model': model_name,
                    'metric_type': metric.capitalize(),
                    'score': entry[metric]
                })
            all_records.append({
                'model': model_name,
                'metric_type': 'Average Score',
                'score': average_score
            })

    if not all_records:
        print(f"\nNo data processed for {grader_type}. Exiting this run.")
        return

    # Create DataFrame, Summary, and Plots
    df = pd.DataFrame(all_records)
    metric_order = ['Relevance', 'Correctness', 'Creativity', 'Completeness', 'Average Score']
    
    summary = df.groupby(['metric_type', 'model'])['score'].describe()
    summary = summary.reindex(metric_order, level='metric_type').reindex(model_order, level='model')
    
    with open(output_summary_file, 'w') as f:
        f.write(f"--- Distribution Summary for {grader_type.upper()} Grading Metrics ---\n\n")
        f.write(summary.to_string())
        
    print(f"\nStatistical summary saved to '{output_summary_file}'")
    
    create_plots(df, output_plot_dir, model_order)
    print(f"\n--- Finished processing for {grader_type.upper()} ---")


if __name__ == "__main__":
    # A list of grader types to process
    grader_types_to_process = ['expert', 'llm']
    
    for grader in grader_types_to_process:
        process_evaluations(grader)

    print("\nAll evaluation analyses are complete.")
