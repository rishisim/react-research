import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
import numpy as np

def generate_pareto_plot(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check required columns
    required_columns = ['em', 'total_tokens']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'")
            return

    print(f"Loaded {len(df)} records.")
    
    # 1. Create Bins for total_tokens
    # We'll use 10 bins.
    num_bins = 10
    df['token_bin'] = pd.cut(df['total_tokens'], bins=num_bins)
    
    # 2. Group by bin
    # We need sum of em (correct count) and count (total items) to calculate cumulative metrics
    bin_stats = df.groupby('token_bin', observed=True).agg({
        'em': ['sum', 'count', 'mean']
    })
    
    # Flatten columns
    bin_stats.columns = ['correct_count', 'total_count', 'bin_accuracy']
    
    # Sort bins (they should be categorical sorted, but let's ensure)
    bin_stats = bin_stats.sort_index()
    
    # 3. Calculate Cumulative Accuracy
    # Cumulative Correct / Cumulative Total Count
    bin_stats['cumsum_correct'] = bin_stats['correct_count'].cumsum()
    bin_stats['cumsum_total'] = bin_stats['total_count'].cumsum()
    bin_stats['cumulative_accuracy'] = bin_stats['cumsum_correct'] / bin_stats['cumsum_total']
    
    # Convert to percentages
    bin_stats['bin_accuracy_pct'] = bin_stats['bin_accuracy'] * 100
    bin_stats['cumulative_accuracy_pct'] = bin_stats['cumulative_accuracy'] * 100
    
    # Create labels for X-axis (Intervals)
    bin_labels = [str(interval) for interval in bin_stats.index]
    
    # 4. Plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar Chart: Bin Accuracy
    # We use range(len) to position bars uniformly
    x_positions = range(len(bin_stats))
    bars = ax1.bar(x_positions, bin_stats['bin_accuracy_pct'], color='#3498db', alpha=0.7, label='Bin Accuracy')
    
    ax1.set_xlabel('Total Tokens (Bins)')
    ax1.set_ylabel('Bin Accuracy (%)', color='#3498db', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    
    # Add values on top of bars
    for i, v in enumerate(bin_stats['bin_accuracy_pct']):
        ax1.text(i, v + 1, f"{v:.1f}%", ha='center', color='#2980b9', fontsize=9)
        # Also add count of items in bin for context
        count = bin_stats.iloc[i]['total_count']
        ax1.text(i, v/2, f"n={count}", ha='center', color='white', fontsize=9)

    # Line Chart: Cumulative Accuracy
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    line = ax2.plot(x_positions, bin_stats['cumulative_accuracy_pct'], color='#e74c3c', marker='o', linewidth=3, label='Cumulative Accuracy')
    
    ax2.set_ylabel('Cumulative Accuracy (%)', color='#e74c3c', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 100)
    
    # Add values for line points
    for i, v in enumerate(bin_stats['cumulative_accuracy_pct']):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', color='#c0392b', fontsize=9, fontweight='bold')

    plt.title('Accuracy vs Token Usage (Pareto Analysis)', fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add legends
    # Combine handles for a single legend if desired, or keep separate axis labels which are clear enough.
    # Let's add a custom legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Save output
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.tight_layout()
    # Adjust layout to make room for rotated x labels and legend
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    
    # Print stats for verification
    print("\nBin Statistics:")
    print(bin_stats[['correct_count', 'total_count', 'bin_accuracy_pct', 'cumulative_accuracy_pct']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Custom Pareto chart from CSV.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output image file", default="pareto_chart.png")
    
    args = parser.parse_args()
    
    generate_pareto_plot(args.input_file, args.output)
