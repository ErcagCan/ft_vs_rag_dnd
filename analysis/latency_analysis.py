import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# File Paths
base_file = './data/processed/question_answer_pairs/qa_base.jsonl'
ft_file = './data/processed/question_answer_pairs/qa_ft.jsonl'
rag_file = './data/processed/question_answer_pairs/qa_rag.jsonl'
ft_rag_file = './data/processed/question_answer_pairs/qa_ft_rag.jsonl'
output_summary_file = './analysis/plots_and_summaries/latency/latency_analysis.txt'
output_plot_dir = './analysis/plots_and_summaries/latency/'


def load_jsonl(filepath):
    """Loads a .jsonl file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Create output directory if it doesn't exist
os.makedirs(output_plot_dir, exist_ok=True)
# Load Data
base_data = load_jsonl(base_file)
ft_data = load_jsonl(ft_file)
rag_data = load_jsonl(rag_file)
ft_rag_data = load_jsonl(ft_rag_file)
# Prepare Data for DataFrame
records = []
# Non-RAG models
for item in base_data:
    records.append({'model': 'BASE', 'retrieval_time': 0, 'query_time': item['query_time'], 'end_to_end_time': item['query_time']})
for item in ft_data:
    records.append({'model': 'FT', 'retrieval_time': 0, 'query_time': item['query_time'], 'end_to_end_time': item['query_time']})
# RAG models
for item in rag_data:
    records.append({'model': 'RAG', 'retrieval_time': item['retrieval_time'], 'query_time': item['query_time'], 'end_to_end_time': item['total_time']})
for item in ft_rag_data:
    records.append({'model': 'FT+RAG', 'retrieval_time': item['retrieval_time'], 'query_time': item['query_time'], 'end_to_end_time': item['total_time']})
# Create DataFrame
df = pd.DataFrame(records)
# Calculate and Save Averages & Distribution Summary
avg_stats = df.groupby('model')[['retrieval_time', 'query_time', 'end_to_end_time']].mean().round(2)
avg_stats = avg_stats.reindex(['BASE', 'FT', 'RAG', 'FT+RAG'])
distribution_summary = df.groupby('model')['end_to_end_time'].describe()
distribution_summary = distribution_summary.reindex(['BASE', 'FT', 'RAG', 'FT+RAG'])


# Save summary to a text file
with open(output_summary_file, 'w', encoding='utf-8') as f:
    f.write("--- Comprehensive Latency Analysis (in seconds) ---\n\n")
    f.write("--- Average Times ---\n")
    f.write(avg_stats.to_string())
    f.write("\n\nNote: 'query_time' for RAG models is the generation time *after* retrieval.\n")
    f.write("\n--- Distribution Summary (End-to-End Time) ---\n")
    f.write(distribution_summary.to_string())
print(f"Latency analysis summary saved to '{output_summary_file}'")


# Stacked Bar Chart for Averages
avg_summary = avg_stats
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(avg_summary.index, avg_summary['retrieval_time'], label='Retrieval Time', color='darkorange')
ax.bar(avg_summary.index, avg_summary['query_time'], bottom=avg_summary['retrieval_time'], label='Generation Time', color='royalblue')
ax.axhline(y=2, color='y', linestyle='--', linewidth=2, label='2s Conversational Limit')
ax.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10s Problem-Solving Limit')
ax.set_title('Average End-to-End Latency by Configuration', fontsize=16, pad=20)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_xlabel('Model Configuration', fontsize=12)
ax.legend()
ax.set_ylim(0, avg_summary['end_to_end_time'].max() * 1.15)


for i, model in enumerate(avg_summary.index):
    total_time = avg_summary.loc[model, 'end_to_end_time']
    ax.text(i, total_time + 0.5, f'{total_time:.1f}s', ha='center', fontweight='bold')


plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, 'latency_bar_chart.png'), dpi=300)


# Box Plot for Distribution
plt.figure(figsize=(10, 7))
sns.boxplot(
    data=df,
    x='model',
    y='end_to_end_time',
    order=['BASE', 'FT', 'RAG', 'FT+RAG'],
    palette='pastel'
)


plt.axhline(y=2, color='y', linestyle='--', linewidth=2, label='2s Conversational Limit')
plt.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10s Problem-Solving Limit')
plt.title('Latency Distribution by Configuration', fontsize=16, pad=20)
plt.xlabel('Model Configuration', fontsize=12)
plt.ylabel('End-to-End Time (seconds)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, 'latency_box_plot.png'), dpi=300)
