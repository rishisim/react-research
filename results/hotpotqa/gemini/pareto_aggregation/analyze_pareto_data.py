import pandas as pd
import numpy as np
import os

# Load the CSV
INPUT_FILE = '/Users/rishisim/Documents/research/react-research/results/hotpotqa/pareto_aggregation/pareto_summary.csv'

if not os.path.exists(INPUT_FILE):
    print(f"File not found: {INPUT_FILE}")
    exit()

df = pd.read_csv(INPUT_FILE)

# Unique frameworks
frameworks = df['framework'].unique()
print(f"Unique Frameworks: {frameworks}")

# Global stats
print("\nGlobal Token Stats:")
print(df['total_tokens'].describe())

# Per framework stats
print("\nPer Framework Token Stats:")
print(df.groupby('framework')['total_tokens'].describe())

# Percentiles for binning insights
print("\nGlobal Percentiles (0-100, step 10):")
print(np.percentile(df['total_tokens'], np.arange(0, 101, 10)))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
