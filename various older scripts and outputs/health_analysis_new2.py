import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import plotly.express as px
from plotly.subplots import make_subplots

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
# For example, the correlation function will use all available data for each pair of variables.


# --- Analysis (Distributions and Statistics) ---
# Note: Descriptive statistics functions in pandas (like describe, skew)
# automatically ignore missing values in their calculations.
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
# FIX: Adjusted layout to prevent all overlapping text.
# 'top' is lowered to give the main title space.
# 'hspace' is increased to add vertical padding between subplot rows.
plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, hspace=0.8, wspace=0.3)
plt.show()


# --- Lagged Sleep Analysis ---

# Create a new dataframe for the lagged analysis
df_lagged = df.copy()

# Define sleep columns
sleep_cols = ['SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP']
other_cols = [col for col in df.columns if col not in sleep_cols]

# Shift sleep columns by one day to represent the previous night's sleep
for col in sleep_cols:
    df_lagged[f'PREV_NIGHT_{col}'] = df_lagged[col].shift(1)

# No need to drop rows; the correlation function will handle the NaN created by the shift.


# --- Correlation Analysis ---

# 1. Correlation of metrics with the *following* night's sleep (Original approach)
plt.figure(figsize=(16, 14))
# .corr() computes pairwise correlation, ignoring NaNs for each pair.
corr_matrix_future_sleep = df.corr()
mask_future = np.triu(np.ones_like(corr_matrix_future_sleep, dtype=bool))

sns.heatmap(corr_matrix_future_sleep,
            mask=mask_future,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)
plt.title("Correlation Matrix: Metrics vs. Following Night's Sleep", fontsize=16)
plt.xticks(rotation=45)
# Adjust bottom margin to prevent labels from being cut off.
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
plt.show()


# 2. Correlation of metrics with the *previous* night's sleep (New approach)
cols_for_prev_night_corr = other_cols + [f'PREV_NIGHT_{col}' for col in sleep_cols]
corr_matrix_prev_sleep = df_lagged[cols_for_prev_night_corr].corr()

# Isolate the correlations of interest
corr_of_interest = corr_matrix_prev_sleep[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]


plt.figure(figsize=(14, 6))
sns.heatmap(corr_of_interest,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)
plt.title("Correlation Matrix: Daily Metrics vs. Previous Night's Sleep", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
# Adjust margins to prevent labels from being cut off.
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.35)
plt.show()


# --- Interactive Visualizations (Optional) ---

# Interactive correlation matrix for the original data
fig = px.imshow(corr_matrix_future_sleep, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig.update_layout(title="Interactive Correlation: Metrics vs. Following Night's Sleep", height=800, width=800)
fig.show()

# Interactive correlation matrix for the lagged data
fig_lagged = px.imshow(corr_of_interest, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig_lagged.update_layout(title="Interactive Correlation: Daily Metrics vs. Previous Night's Sleep", height=500, width=900)
fig_lagged.show()
