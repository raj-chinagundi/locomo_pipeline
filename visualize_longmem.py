import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
print("Loading longmem_results.json...")
with open('longmem_results.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions")

# Extract retrievers
retrievers = ['BM25', 'TF-IDF', 'SVM', 'FAISS', 'Time-Weighted']

# Calculate metrics for each question and retriever
def calculate_metrics_for_question(question_data):
    """Calculate recall@5 and precision@5 for a single question."""
    answer_session_ids = set(question_data.get('answer_session_ids', []))
    if not answer_session_ids:
        return {}
    
    metrics = {}
    for retriever_name in retrievers:
        retrieved = question_data['Retriever'][retriever_name]['Top5_retrieved_ids']
        retrieved_set = set(retrieved[:5])
        
        relevant_retrieved = len(retrieved_set & answer_session_ids)
        recall = relevant_retrieved / len(answer_session_ids) if answer_session_ids else 0.0
        precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
        success = 1 if (retrieved_set & answer_session_ids) else 0
        
        metrics[retriever_name] = {
            'recall@5': recall,
            'precision@5': precision,
            'success': success
        }
    
    return metrics

# Calculate all metrics
print("Calculating metrics...")
all_metrics = {r: {'recall@5': [], 'precision@5': [], 'success': []} for r in retrievers}

for question_data in data:
    metrics = calculate_metrics_for_question(question_data)
    for retriever in retrievers:
        if retriever in metrics:
            all_metrics[retriever]['recall@5'].append(metrics[retriever]['recall@5'])
            all_metrics[retriever]['precision@5'].append(metrics[retriever]['precision@5'])
            all_metrics[retriever]['success'].append(metrics[retriever]['success'])

# Calculate averages
avg_metrics = {}
for retriever in retrievers:
    avg_metrics[retriever] = {
        'recall@5': np.mean(all_metrics[retriever]['recall@5']),
        'precision@5': np.mean(all_metrics[retriever]['precision@5']),
        'success_rate': np.mean(all_metrics[retriever]['success'])
    }

# ========== FIGURE 1: Overall Performance Comparison ==========
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Average Recall@5
ax1 = axes1[0]
recall_values = [avg_metrics[r]['recall@5'] for r in retrievers]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
bars = ax1.bar(retrievers, recall_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Recall@5', fontsize=12, fontweight='bold')
ax1.set_title('Average Recall@5 by Retriever', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, max(recall_values) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, recall_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Success Rate
ax2 = axes1[1]
success_rates = [avg_metrics[r]['success_rate'] * 100 for r in retrievers]
bars = ax2.bar(retrievers, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Success Rate by Retriever', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylim(0, max(success_rates) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, success_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('longmem_overall_performance.png', dpi=300, bbox_inches='tight')
print("Saved: longmem_overall_performance.png")
plt.close()

# ========== FIGURE 2: Question Type-Wise Performance ==========
# Calculate question type metrics
question_type_metrics = defaultdict(lambda: defaultdict(list))
for question_data in data:
    question_type = question_data.get('question_type', 'Unknown')
    metrics = calculate_metrics_for_question(question_data)
    for retriever in retrievers:
        if retriever in metrics:
            question_type_metrics[question_type][retriever].append(metrics[retriever]['recall@5'])

question_types = sorted(question_type_metrics.keys())
question_type_recall = {}
for qtype in question_types:
    question_type_recall[qtype] = {}
    for retriever in retrievers:
        if retriever in question_type_metrics[qtype]:
            question_type_recall[qtype][retriever] = np.mean(question_type_metrics[qtype][retriever])

# Create heatmap
fig2, ax2 = plt.subplots(figsize=(12, 6))
heatmap_data = []
for retriever in retrievers:
    row = [question_type_recall[qtype].get(retriever, 0) for qtype in question_types]
    heatmap_data.append(row)

im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)
ax2.set_xticks(np.arange(len(question_types)))
ax2.set_yticks(np.arange(len(retrievers)))
ax2.set_xticklabels([qtype.replace('-', '\n') for qtype in question_types], fontsize=10)
ax2.set_yticklabels(retrievers, fontsize=11)
ax2.set_xlabel('Question Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Retriever', fontsize=12, fontweight='bold')
ax2.set_title('Recall@5 by Retriever and Question Type', fontsize=14, fontweight='bold', pad=15)

# Add text annotations
for i in range(len(retrievers)):
    for j in range(len(question_types)):
        val = heatmap_data[i][j]
        color = 'white' if val < 0.5 else 'black'
        ax2.text(j, i, f'{val:.3f}',
                ha="center", va="center", color=color, fontsize=9, fontweight='bold')

cbar = plt.colorbar(im, ax=ax2, label='Recall@5', shrink=0.8)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('longmem_question_type_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: longmem_question_type_heatmap.png")
plt.close()

# ========== FIGURE 3: Question Type Comparison (Bar Chart) ==========
fig3, ax3 = plt.subplots(figsize=(16, 6))

# Show ALL retrievers per question type
x = np.arange(len(question_types))
width = 0.15

for i, retriever in enumerate(retrievers):
    values = [question_type_recall[qtype].get(retriever, 0) for qtype in question_types]
    offset = (i - len(retrievers)/2 + 0.5) * width
    bars = ax3.bar(x + offset, values, width, label=retriever, alpha=0.8, 
                   color=colors[i], edgecolor='black', linewidth=1)

ax3.set_xlabel('Question Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Recall@5', fontsize=12, fontweight='bold')
ax3.set_title('Recall@5 by Question Type (All Retrievers)', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels([qtype.replace('-', '\n') for qtype in question_types], fontsize=10)
ax3.legend(fontsize=10, framealpha=0.9, ncol=5, loc='upper left')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('longmem_question_type_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: longmem_question_type_comparison.png")
plt.close()

# ========== FIGURE 4: Distribution Comparison ==========
fig4, ax4 = plt.subplots(figsize=(12, 6))

# Box plot for recall distribution
recall_data = [all_metrics[r]['recall@5'] for r in retrievers]
bp = ax4.boxplot(recall_data, labels=retrievers, patch_artist=True, 
                 showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax4.set_ylabel('Recall@5', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Recall@5 Scores', fontsize=14, fontweight='bold', pad=15)
ax4.set_xticklabels(retrievers, fontsize=11)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('longmem_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: longmem_distribution.png")
plt.close()

# ========== Print Summary ==========
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"{'Retriever':<15} {'Recall@5':<12} {'Precision@5':<12} {'Success Rate':<12}")
print("-"*70)
for retriever in retrievers:
    recall = avg_metrics[retriever]['recall@5']
    precision = avg_metrics[retriever]['precision@5']
    success = avg_metrics[retriever]['success_rate'] * 100
    print(f"{retriever:<15} {recall:<12.4f} {precision:<12.4f} {success:<12.2f}%")
print("="*70)

# Question type-wise summary
print("\n" + "="*80)
print("QUESTION TYPE-WISE PERFORMANCE")
print("="*80)
for qtype in question_types:
    num_questions = len(question_type_metrics[qtype][retrievers[0]]) if retrievers[0] in question_type_metrics[qtype] else 0
    print(f"\n{qtype} ({num_questions} questions):")
    print("-" * 80)
    sorted_retrievers = sorted(retrievers, 
                              key=lambda x: question_type_recall[qtype].get(x, 0), 
                              reverse=True)
    for retriever in sorted_retrievers:
        recall = question_type_recall[qtype].get(retriever, 0)
        print(f"  {retriever:<15} Recall@5: {recall:.4f}")
print("="*80)

print("\n✅ All visualizations saved successfully!")
print("   - longmem_overall_performance.png")
print("   - longmem_question_type_heatmap.png")
print("   - longmem_question_type_comparison.png")
print("   - longmem_distribution.png")

