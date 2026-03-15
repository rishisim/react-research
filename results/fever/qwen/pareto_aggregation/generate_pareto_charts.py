import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'pareto_summary.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'fever_charts_500')
BIN_WIDTH = 2500
MAX_TOKEN_LIMIT = 50000  # Cap for visualization focus

# Validated mappings for FEVER (Qwen)
FRAMEWORK_MAPPING = {
    'react': 'ReAct',
    'reflexion_react': 'Reflexion', 
    'cot_sc': 'CoT-SC',
    'majority_voting': 'Majority Voting',
    'Trajectory-Conditioned Answer Revision (TCAR)': 'Trajectory-Conditioned Answer Revision (TCAR)',
}

def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 300})

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Drop rows where critical metrics are missing
    df = df.dropna(subset=['em', 'total_tokens', 'framework'])
    return df

def generate_bins(df):
    bins = list(range(0, MAX_TOKEN_LIMIT + BIN_WIDTH, BIN_WIDTH))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    return bins, labels

def calculate_metrics(df_framework, bins, labels):
    # Assign bins
    df_framework = df_framework.copy()
    df_framework['bin'] = pd.cut(df_framework['total_tokens'], bins=bins, labels=labels, include_lowest=True, right=True)
    
    # 1. Bar Chart Data: Accuracy (Mean EM) per bin
    bin_stats = df_framework.groupby('bin', observed=False)['em'].agg(['mean', 'sum']).reset_index()
    bin_stats.rename(columns={'mean': 'accuracy', 'sum': 'successes'}, inplace=True)
    bin_stats['successes'] = bin_stats['successes'].fillna(0)
    
    # 2. Cumulative Data
    total_tasks = len(df_framework)
    cumulative_accuracy = []
    
    for edge in bins[1:]:  # Skip 0
        successes = df_framework[
            (df_framework['total_tokens'] <= edge) & (df_framework['em'] == 1)
        ].shape[0]
        acc_at_budget = successes / total_tasks
        cumulative_accuracy.append(acc_at_budget)
        
    return bin_stats, cumulative_accuracy

def plot_individual_chart(framework_name, bin_stats, cumulative_accuracy, labels, output_path):
    display_name = FRAMEWORK_MAPPING.get(framework_name, framework_name)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar Chart (Left Y) - Accuracy per Bin
    sns.barplot(data=bin_stats, x='bin', y='accuracy', color='skyblue', alpha=0.6, ax=ax1, errorbar=None)
    ax1.set_ylabel('Accuracy (Mean EM) per Bin', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Token Bins', fontsize=12)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title(f'Pareto Chart (FEVER - Qwen): {display_name}', fontsize=16)
    
    # Line Chart (Right Y) - Cumulative Accuracy
    ax2 = ax1.twinx()
    ax2.plot(labels, cumulative_accuracy, color='darkgreen', marker='o', linewidth=2, label='Cumulative Accuracy')
    ax2.set_ylabel('Cumulative Accuracy (Budget <= Bin)', color='darkgreen', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylim(0, 1.05)
    
    # Add table with success counts below x-axis
    success_counts = [int(x) for x in bin_stats['successes'].tolist()]
    
    the_table = ax1.table(cellText=[success_counts],
                          rowLabels=['S'],
                          loc='bottom',
                          cellLoc='center',
                          bbox=[0, -0.35, 1, 0.08])
                          
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    plt.subplots_adjust(bottom=0.3)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_combined_chart(frameworks_data, labels, output_path):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = sns.color_palette("tab10", n_colors=len(frameworks_data))
    
    for i, (name, stats) in enumerate(frameworks_data.items()):
        cumulative_acc = stats['cumulative']
        ax.plot(labels, cumulative_acc, marker='', linewidth=2.5, label=name, color=colors[i])
        
    ax.set_title('Combined Pareto Comparison (FEVER - Qwen): Cumulative Accuracy vs Token Budget', fontsize=16)
    ax.set_xlabel('Token Budget', fontsize=12)
    ax.set_ylabel('Cumulative Accuracy (Tasks Solved / Total)', fontsize=12)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xticks(range(len(labels)))
    
    # Move legend to bottom
    ax.legend(title='Framework', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, title_fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    setup_plotting_style()
    df = load_and_clean_data(INPUT_FILE)
    bins, labels = generate_bins(df)
    
    frameworks = df['framework'].unique()
    combined_data = {}
    
    print(f"Generating charts for {len(frameworks)} frameworks...")
    
    for fw in frameworks:
        print(f"Processing {fw}...")
        df_fw = df[df['framework'] == fw]
        
        bin_stats, cumul_acc = calculate_metrics(df_fw, bins, labels)
        
        # Use mapped name for combined chart key
        display_name = FRAMEWORK_MAPPING.get(fw, fw)
        combined_data[display_name] = {'cumulative': cumul_acc}
        
        # Plot Individual
        out_file = os.path.join(OUTPUT_DIR, f'pareto_{fw}.png')
        plot_individual_chart(fw, bin_stats, cumul_acc, labels, out_file)
        
    # Plot Combined
    print("Generating combined chart...")
    plot_combined_chart(combined_data, labels, os.path.join(OUTPUT_DIR, 'combined_pareto_accuracy.png'))
    print(f"Done! Charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
