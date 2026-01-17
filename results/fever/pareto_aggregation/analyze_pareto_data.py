import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('/Users/rishisim/Documents/research/react-research/results/fever/pareto_aggregation/pareto_summary.csv')

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
