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


def generate_comparison_bar_charts(df, metric_order, model_order, evaluator_order):
    """Generates grouped bar charts comparing evaluator types for each metric."""
    print("\nGenerating comparison bar charts...")
    for metric in metric_order:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            data=df[df['metric_type'] == metric],
            x='model', y='score', hue='evaluator',
            order=model_order, hue_order=evaluator_order,
            palette='muted', errorbar='sd', capsize=0.1
        )
        ax.set_title(f'Mean Score Comparison for "{metric}"', fontsize=18, pad=20)
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel('Mean Score (1-5 Scale)', fontsize=14)
        ax.set_ylim(0, 5.5)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(title='Evaluator', fontsize=12, title_fontsize=14)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)
        filename = f"plot_comparison_barchart_{metric.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(f"./analysis/plots_and_summaries/expert_vs_llm/{filename}", dpi=300)
        print(f"\nSaved {filename}")
        plt.close()


def generate_comparison_boxplots(df, metric_order, model_order, evaluator_order):
    """Generates grouped box plots comparing evaluator types for each metric."""
    print("\nGenerating comparison box plots...")
    for metric in metric_order:
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(
            data=df[df['metric_type'] == metric],
            x='model', y='score', hue='evaluator',
            order=model_order, hue_order=evaluator_order,
            palette='muted'
        )
        ax.set_title(f'Score Distribution Comparison for "{metric}"', fontsize=18, pad=20)
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel('Score (1-5 Scale)', fontsize=14)
        ax.set_ylim(0.5, 5.5)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(title='Evaluator', fontsize=12, title_fontsize=14)
        filename = f"plot_comparison_boxplot_{metric.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(f"./analysis/plots_and_summaries/expert_vs_llm/{filename}", dpi=300)
        print(f"\nSaved {filename}")
        plt.close()


def generate_comparison_violin_plots(df, metric_order, model_order, evaluator_order):
    """Generates grouped violin plots comparing evaluator types for each metric."""
    print("\nGenerating comparison violin plots...")
    for metric in metric_order:
        plt.figure(figsize=(12, 8))
        ax = sns.violinplot(
            data=df[df['metric_type'] == metric],
            x='model', y='score', hue='evaluator',
            order=model_order, hue_order=evaluator_order,
            palette='muted',
            inner='quartile',
            split=True
        )
        ax.set_title(f'Score Distribution Comparison for "{metric}"', fontsize=18, pad=20)
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel('Score (1-5 Scale)', fontsize=14)
        ax.set_ylim(0.5, 5.5)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(title='Evaluator', fontsize=12, title_fontsize=14)
        filename = f"plot_comparison_violin_{metric.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(f"./analysis/plots_and_summaries/expert_vs_llm/{filename}", dpi=300)
        print(f"\nSaved {filename}")
        plt.close()


# Main Script Logic
output_summary_file = './analysis/plots_and_summaries/expert_vs_llm/comparison_metrics_summary.txt'
evaluators_config = {'Expert': './results/expert_evaluation', 'LLM': './results/llm_evaluation'}
models_config = {
    'BASE': 'evaluations_base.jsonl', 'RAG': 'evaluations_rag.jsonl',
    'FT': 'evaluations_ft.jsonl', 'FT+RAG': 'evaluations_ft_rag.jsonl'
}
metrics = ['relevance', 'correctness', 'creativity', 'completeness']
all_records = []


for evaluator_name, base_path in evaluators_config.items():
    for model_name, filename_suffix in models_config.items():
        prefix = (base_path.split('_')[0]).split('/')[2]
        filepath = os.path.join(base_path, f"{prefix}_{filename_suffix}")
        data = load_jsonl(filepath)
        if data is None: continue
        print(f"\nProcessing {len(data)} records from {filepath}...")
        for entry in data:
            if not all(metric in entry for metric in metrics): continue
            scores = [entry[metric] for metric in metrics]
            average_score = sum(scores) / len(scores)
            for metric in metrics:
                all_records.append({
                    'model': model_name, 'evaluator': evaluator_name,
                    'metric_type': metric.capitalize(), 'score': entry[metric]
                })
            all_records.append({
                'model': model_name, 'evaluator': evaluator_name,
                'metric_type': 'Average Score', 'score': average_score
            })


if not all_records:
    print("\nNo data processed. Exiting.")
else:
    df = pd.DataFrame(all_records)
    metric_order = ['Relevance', 'Correctness', 'Creativity', 'Completeness', 'Average Score']
    model_order = ['BASE', 'FT', 'RAG', 'FT+RAG']
    evaluator_order = ['Expert', 'LLM']
    summary = df.groupby(['metric_type', 'evaluator', 'model'])['score'].describe()
    summary = summary.reindex(metric_order, level='metric_type')
    summary = summary.reindex(evaluator_order, level='evaluator')
    summary = summary.reindex(model_order, level='model')
    with open(output_summary_file, 'w') as f:
        f.write("--- Comparative Distribution Summary for Evaluation Metrics ---\n\n")
        f.write(summary.to_string())
    print(f"\nComparative statistical summary saved to '{output_summary_file}'")
    print(summary)

    # Generate All Comparison Plots 
    generate_comparison_bar_charts(df, metric_order, model_order, evaluator_order)
    generate_comparison_boxplots(df, metric_order, model_order, evaluator_order)
    generate_comparison_violin_plots(df, metric_order, model_order, evaluator_order)