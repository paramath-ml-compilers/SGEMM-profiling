import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data, assuming the file is named 'data.txt'
df = pd.read_csv('benchmark-data.txt', sep=',')

# Remove rows where any column contains zero
df_filtered = df[(df != 0).all(axis=1)]

# Create a directory to save plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Set plot style
num_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns

sns.set(style='whitegrid', context='talk')

# ------------------------------
# 1) Single Histogram of GFLOPS
# ------------------------------
plt.figure(figsize=(8, 6))
sns.histplot(df_filtered['GFLOPS'], bins=30, kde=True)
plt.title('Distribution of GFLOPS')
plt.xlabel('GFLOPS')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'hist_GFLOPS.png'))
plt.close()

# -------------------------------------------------------
# 2) Scatter Plot: GFLOPS vs. Execution_Time_us
#    - color-coded by DRAM_Throughput_Pct
# -------------------------------------------------------
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    data=df_filtered,
    x='Execution_Time_us',
    y='GFLOPS',
    hue='DRAM_Throughput_Pct',
    palette='viridis'
)
plt.title('GFLOPS vs. Execution Time (µs)\nColor: DRAM Throughput (%)')
plt.xlabel('Execution Time (µs)')
plt.ylabel('GFLOPS')
plt.legend(title='DRAM Thruput (%)', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_GFLOPS_vs_ExecutionTime.png'))
plt.close()

# -------------------------------------------------------
# 3) Scatter Plot: GFLOPS vs. Warps_Active_Pct
#    - color-coded by Registers_Per_Thread
# -------------------------------------------------------
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    data=df_filtered,
    x='Warps_Active_Pct',
    y='GFLOPS',
    hue='Registers_Per_Thread',
    palette='plasma'
)
plt.title('GFLOPS vs. Warps Active (%)\nColor: Registers per Thread')
plt.xlabel('Warps Active (%)')
plt.ylabel('GFLOPS')
plt.legend(title='Registers/Thread', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_GFLOPS_vs_WarpsActive.png'))
plt.close()

# -------------------------------------------------------------------
# 4) Correlation Heatmap: Focus on Key Performance & Config Columns
# -------------------------------------------------------------------
# Choose columns of interest for correlation
cols_of_interest = [
    'MatrixSize',
    'GFLOPS',
    'DRAM_Throughput_Pct',
    'Execution_Time_us',
    'Registers_Per_Thread',
    'Warps_Active_Pct',
    'BK', 'BM', 'BN', 'WM', 'WN', 'WN_ITER', 'TM', 'TN', 'NUM_THREADS'
]

df_corr = df_filtered[cols_of_interest].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap (Key Metrics & Config Params)')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

print(f"Analysis completed. Selected plots have been saved in the '{output_dir}' directory.")

# Generate correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df_filtered.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# Generate pair plots for selected features
pairplot = sns.pairplot(df_filtered[num_cols[:5]])  # Limit to first 5 numerical columns for readability
pairplot.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.close()

print(f"Analysis completed. All plots have been saved in the '{output_dir}' directory.")