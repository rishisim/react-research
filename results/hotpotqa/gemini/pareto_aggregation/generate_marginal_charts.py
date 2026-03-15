import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
INPUT_FILE = '/Users/rishisim/Documents/research/react-research/results/hotpotqa/pareto_aggregation/pareto_summary.csv'
OUTPUT_DIR = '/Users/rishisim/Documents/research/react-research/results/hotpotqa/pareto_aggregation/hotpotqa_charts_500'
TASK_BIN_SIZE = 10  # Number of tasks per bin for smoothing

def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (14, 8), 'figure.dpi': 300})

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Drop rows where critical metrics are missing
    df = df.dropna(subset=['em', 'total_tokens', 'framework'])
    return df

def calculate_marginal_data(df, bin_size=TASK_BIN_SIZE):
    """
    Calculates marginal tokens per task bin.
    """
    # Define the frameworks we want to include and their display names
    # Framework ID -> Display Name
    FRAMEWORK_MAPPING = {
        'reflexion': 'Reflexion',
        'cot_sc': 'CoT-SC',
        'Trajectory-Conditioned Answer Revision (TCAR)': 'Trajectory-Conditioned Answer Revision (TCAR)',
        'react': 'ReAct',
        'majority_voting': 'Majority Voting'
    }

    results = {}

    for fw_id, fw_label in FRAMEWORK_MAPPING.items():
        # Filter for solved tasks only (em=1) and specific framework
        df_fw = df[(df['framework'] == fw_id) & (df['em'] == 1)].copy()
        
        # Sort by cost (tokens) to find the "cheapest" tasks first
        df_fw = df_fw.sort_values('total_tokens')
        
        # Create bins
        # We want to group every N tasks
        num_solved = len(df_fw)
        if num_solved == 0:
            print(f"Warning: No solved tasks found for framework '{fw_id}'")
            continue
            
        bins_x = []
        marginal_costs_y = []
        
        # Iterate through chunks of sorted tasks
        for i in range(0, num_solved, bin_size):
            chunk = df_fw.iloc[i : i + bin_size]
            
            # X value: essentially the "Nth task" solved. 
            # We can use the end of the bin (e.g., "Tasks 1-10", plotting at 10) 
            # or the midpoint. Let's strictly say "Number of Tasks Solved"
            # so we plot at the end of the bin (i + len(chunk))
            x_val = i + len(chunk)
            
            # Y value: Average tokens for tasks in this chunk
            # This represents "How much did it cost on average to solve these specific tasks?"
            # Since they are sorted by cheapness, this curve should go UP.
            y_val = chunk['total_tokens'].mean()
            
            bins_x.append(x_val)
            marginal_costs_y.append(y_val)
            
        results[fw_label] = {
            'x': bins_x,
            'y': marginal_costs_y
        }
        
    return results

def plot_marginal_chart(data, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("tab10", n_colors=len(data))
    
    for i, (name, stats) in enumerate(data.items()):
        ax.plot(stats['x'], stats['y'], marker='o', markersize=4, linewidth=2.5, label=name, color=colors[i])
        
    ax.set_title(f'Marginal Cost Analysis (HotPotQA): Tokens per Additional Task Solved (Smoothed by {TASK_BIN_SIZE} tasks)', fontsize=16)
    ax.set_xlabel('Number of Tasks Solved (Cheapest to Most Expensive)', fontsize=14)
    ax.set_ylabel('Average Tokens per Task (in bin)', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend
    ax.legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Chart saved to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    setup_plotting_style()
    df = load_and_clean_data(INPUT_FILE)
    
    print("Calculating marginal data...")
    marginal_data = calculate_marginal_data(df)
    
    output_file = os.path.join(OUTPUT_DIR, 'marginal_cost_per_task.png')
    print("Generating chart...")
    plot_marginal_chart(marginal_data, output_file)

if __name__ == "__main__":
    main()
