import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import plotly.express as px
from plotly.subplots import make_subplots
# Import pearsonr from scipy to calculate p-values
from scipy.stats import pearsonr

# Load the dataset, parsing dates and setting the DATE column as the index
# Note: Ensure you have a 'health_data.csv' file in the same directory.
try:
    df = pd.read_csv("health_data.csv", parse_dates=['DATE'], index_col='DATE')
except FileNotFoundError:
    print("Error: 'health_data.csv' not found. Please make sure the CSV file is in the correct directory.")
    exit()


# Select only the columns needed for the analysis
cols_to_use = ['WEIGHT', 'BPAVGSYS', 'BPAVGDIA', 'BPMAXSYS', 'BPMAXDIA', 'BPMINSYS',
               'BPMINDIA', 'SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP', 'HRAVG', 'HRMAX', 'HRMIN',
               'STRESSAVG', 'STRESSMAX', 'STRESSMIN']
df = df[cols_to_use]

# --- Data Cleaning: No data is dropped or filled. ---
# Missing values will be handled by each calculation on a pairwise basis.

# --- Analysis (Distributions and Statistics) ---
# Basic statistics
summary = df.describe(percentiles=[.25, .5, .75]).T
summary['IQR'] = summary['75%'] - summary['25%']

# Add skewness and kurtosis
summary['skewness'] = df.skew()
summary['kurtosis'] = df.kurtosis()

# Display key statistics
print("--- Summary Statistics ---")
print(summary[['mean', 'std', 'min', '50%', 'max', 'IQR', 'skewness', 'kurtosis']])
print("\n")

# Generate and display summary table
table_data = []
for col in df.columns:
    table_data.append([
        col,
        f"{summary.loc[col, 'mean']:.2f} ± {summary.loc[col, 'std']:.2f}",
        f"{summary.loc[col, 'min']:.1f} - {summary.loc[col, 'max']:.1f}",
        summary.loc[col, '50%'],
        summary.loc[col, 'IQR'],
        f"{summary.loc[col, 'skewness']:.2f}"
    ])

print("--- Detailed Marker Summary ---")
print(tabulate(table_data,
               headers=['Marker', 'Mean ± SD', 'Range', 'Median', 'IQR', 'Skewness'],
               tablefmt='github',
               floatfmt=".2f"))
print("\n")

# --- Static Distribution Plots ---
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 20))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, color='skyblue', ax=axes[i])
    axes[i].set_title(f'Distribution: {col}')
    axes[i].set_xlabel('')

# If there are fewer than 16 columns, hide the unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Health Marker Distributions', fontsize=24)
plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, hspace=0.8, wspace=0.3)
plt.show()

# --- Normalized Statistics Heatmap ---
stats_to_show = summary[['mean', 'std', '50%', 'IQR']]
# Normalization formula: (value - min) / (max - min)
normalized_stats = (stats_to_show - stats_to_show.min()) / (stats_to_show.max() - stats_to_show.min())

plt.figure(figsize=(12, 8))
sns.heatmap(normalized_stats.T,
            annot=stats_to_show.T, # Show the original, unnormalized values on the plot
            fmt=".2f",
            cmap='viridis',
            cbar_kws={'label': 'Normalized Value'})
plt.title('Normalized Summary Statistics Comparison', fontsize=16)
plt.xlabel('Health Markers')
plt.ylabel('')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.1, right=1.0, top=0.95, bottom=0.25)
plt.show()


# --- Lagged Sleep Analysis ---
df_lagged = df.copy()
sleep_cols = ['SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP']
other_cols = [col for col in df.columns if col not in sleep_cols]

for col in sleep_cols:
    df_lagged[f'PREV_NIGHT_{col}'] = df_lagged[col].shift(1)


# --- REVISED: Function to calculate a matrix of p-values ---
def calculate_pvalues(df):
    """
    Calculates a matrix of p-values for the Pearson correlation between columns.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: A dataframe containing the p-values for each pair of columns.
    """
    p_values = pd.DataFrame(index=df.columns, columns=df.columns)
    
    for r in df.columns:
        for c in df.columns:
            # --- FIX: Handle the case where a column is correlated with itself ---
            # This was the source of the error.
            if r == c:
                p_values.loc[r, c] = 0.0
                continue # Skip to the next pair

            # Create a temporary DataFrame and drop missing values for the pair
            df_pair = df[[r, c]].dropna()
            
            # Ensure there are enough data points to calculate correlation
            if len(df_pair) > 2:
                # Use .values to pass 1D numpy arrays to pearsonr
                corr, p_val = pearsonr(df_pair[r].values, df_pair[c].values)
                p_values.loc[r, c] = p_val
            else:
                p_values.loc[r, c] = np.nan # Not enough data
                
    # Convert the DataFrame to a numeric type
    return p_values.astype(float)


# --- Correlation Analysis (with Significance) ---

# --- 1. Correlation of metrics with the *following* night's sleep ---
plt.figure(figsize=(16, 14))

# Calculate correlation and p-values
corr_matrix_future_sleep = df.corr()
p_values_future = calculate_pvalues(df)

# Create a mask for the upper triangle
mask_future = np.triu(np.ones_like(corr_matrix_future_sleep, dtype=bool))

# Create annotation labels, showing correlation only if p < 0.05
annot_future = corr_matrix_future_sleep.applymap(lambda x: f'{x:.2f}')
annot_future[p_values_future >= 0.05] = '' # Hide non-significant correlations

sns.heatmap(corr_matrix_future_sleep,
            mask=mask_future,
            annot=annot_future,  # Use the new filtered annotations
            fmt="s",  # Set format to string since we are passing pre-formatted strings
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)

plt.title("Correlation Matrix: Metrics vs. Following Night's Sleep (p < 0.05)", fontsize=16)
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
plt.show()


# --- 2. Correlation of metrics with the *previous* night's sleep ---
cols_for_prev_night_corr = other_cols + [f'PREV_NIGHT_{col}' for col in sleep_cols]
df_for_prev_night_corr = df_lagged[cols_for_prev_night_corr]

# Calculate correlation and p-values for the lagged data
corr_matrix_prev_sleep = df_for_prev_night_corr.corr()
p_values_prev = calculate_pvalues(df_for_prev_night_corr)

# Isolate the correlations and p-values of interest
corr_of_interest = corr_matrix_prev_sleep[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]
p_values_of_interest = p_values_prev[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]

# Create annotation labels, showing correlation only if p < 0.05
annot_prev = corr_of_interest.applymap(lambda x: f'{x:.2f}')
annot_prev[p_values_of_interest >= 0.05] = '' # Hide non-significant correlations

plt.figure(figsize=(14, 6))
sns.heatmap(corr_of_interest,
            annot=annot_prev,  # Use the new filtered annotations
            fmt="s",  # Set format to string
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)

plt.title("Correlation Matrix: Daily Metrics vs. Previous Night's Sleep (p < 0.05)", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.35)
plt.show()


# --- Interactive Visualizations (Optional) ---
# Note: Significance filtering is not applied to these interactive plots.

# Interactive correlation matrix for the original data
fig = px.imshow(corr_matrix_future_sleep, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig.update_layout(title="Interactive Correlation: Metrics vs. Following Night's Sleep", height=800, width=800)
fig.show()

# Interactive correlation matrix for the lagged data
fig_lagged = px.imshow(corr_of_interest, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig_lagged.update_layout(title="Interactive Correlation: Daily Metrics vs. Previous Night's Sleep", height=500, width=900)
fig_lagged.show()
