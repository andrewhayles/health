import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

# --- Configuration (Constants) ---
# Constants are defined at the top level for easy configuration.
FILEPATH = "health_data.csv"
COLS_TO_USE = [
    'WEIGHT', 'BPAVGSYS', 'BPAVGDIA', 'BPMAXSYS', 'BPMAXDIA', 'BPMINSYS',
    'BPMINDIA', 'SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP', 'HRAVG', 'HRMAX', 'HRMIN',
    'STRESSAVG', 'STRESSMAX', 'STRESSMIN'
]
UNITS = {
    'WEIGHT': 'kg',
    'BPAVGSYS': 'mmHg', 'BPAVGDIA': 'mmHg', 'BPMAXSYS': 'mmHg',
    'BPMAXDIA': 'mmHg', 'BPMINSYS': 'mmHg', 'BPMINDIA': 'mmHg',
    'SLEEPTOTAL': 'hours', 'DEEPSLEEP': 'hours', 'LIGHTSLEEP': 'hours',
    'HRAVG': 'bpm', 'HRMAX': 'bpm', 'HRMIN': 'bpm',
    'STRESSAVG': 'N/A', 'STRESSMAX': 'N/A', 'STRESSMIN': 'N/A'
}

# --- Data Loading and Preparation ---

def load_and_prepare_data(filepath, columns_to_use):
    """
    Loads health data from a CSV, parses dates, and selects relevant columns.

    Args:
        filepath (str): The path to the CSV file.
        columns_to_use (list): A list of column names to keep for analysis.

    Returns:
        pd.DataFrame: A prepared DataFrame ready for analysis.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=['DATE'], index_col='DATE')
        return df[columns_to_use]
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please ensure the file is in the correct directory.")
        exit()

# --- Analysis Functions ---

def calculate_summary_statistics(df):
    """
    Calculates and returns detailed summary statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics.
    """
    summary = df.describe(percentiles=[.25, .5, .75]).T
    summary['IQR'] = summary['75%'] - summary['25%']
    summary['skewness'] = df.skew()
    summary['kurtosis'] = df.kurtosis()
    print("--- Summary Statistics ---")
    print(summary[['mean', 'std', 'min', '50%', 'max', 'IQR', 'skewness', 'kurtosis']])
    print("\n")
    return summary

def print_detailed_summary(summary_df, units_dict):
    """
    Prints a formatted table of summary statistics with units.

    Args:
        summary_df (pd.DataFrame): The summary statistics DataFrame.
        units_dict (dict): A dictionary mapping markers to their units.
    """
    table_data = []
    for col in summary_df.index:
        unit = units_dict.get(col, 'N/A')
        table_data.append([
            col, unit,
            f"{summary_df.loc[col, 'mean']:.2f} ± {summary_df.loc[col, 'std']:.2f}",
            f"{summary_df.loc[col, 'min']:.1f} - {summary_df.loc[col, 'max']:.1f}",
            summary_df.loc[col, '50%'],
            summary_df.loc[col, 'IQR'],
            f"{summary_df.loc[col, 'skewness']:.2f}"
        ])
    print("--- Detailed Marker Summary ---")
    print(tabulate(table_data,
                   headers=['Marker', 'Unit', 'Mean ± SD', 'Range', 'Median', 'IQR', 'Skewness'],
                   tablefmt='github', floatfmt=".2f"))
    print("\n")

def create_lagged_sleep_data(df):
    """
    Creates a new DataFrame with lagged columns for sleep metrics.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with new 'PREV_NIGHT_*' columns.
    """
    df_lagged = df.copy()
    sleep_cols = ['SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP']
    for col in sleep_cols:
        df_lagged[f'PREV_NIGHT_{col}'] = df_lagged[col].shift(1)
    return df_lagged

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
            if r == c:
                p_values.loc[r, c] = 0.0
                continue
            df_pair = df[[r, c]].dropna()
            if len(df_pair) > 2:
                _, p_val = pearsonr(df_pair[r].values, df_pair[c].values)
                p_values.loc[r, c] = p_val
            else:
                p_values.loc[r, c] = np.nan
    return p_values.astype(float)

# --- Visualization Functions ---

def plot_distributions(df, units_dict):
    """
    Generates and displays a grid of histograms for each health marker.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        units_dict (dict): A dictionary mapping markers to their units.
    """
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 20))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, color='skyblue', ax=axes[i])
        unit = units_dict.get(col, '')
        title = f'Distribution: {col}'
        if unit and unit != 'N/A':
            title += f' ({unit})'
        axes[i].set_title(title)
        axes[i].set_xlabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Health Marker Distributions', fontsize=24)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, hspace=0.8, wspace=0.3)
    plt.show()

def plot_normalized_stats_heatmap(summary_df):
    """
    Generates and displays a heatmap of normalized summary statistics.

    Args:
        summary_df (pd.DataFrame): The summary statistics DataFrame.
    """
    stats_to_show = summary_df[['mean', 'std', '50%', 'IQR']]
    normalized_stats = (stats_to_show - stats_to_show.min()) / (stats_to_show.max() - stats_to_show.min())
    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_stats.T, annot=stats_to_show.T, fmt=".2f",
                cmap='viridis', cbar_kws={'label': 'Normalized Value'})
    plt.title('Normalized Summary Statistics Comparison', fontsize=16)
    plt.xlabel('Health Markers')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.1, right=1.0, top=0.95, bottom=0.25)
    plt.show()

def plot_future_sleep_correlation(df):
    """
    Plots a correlation matrix of metrics vs. the following night's sleep,
    showing only statistically significant (p < 0.05) correlations.

    Args:
        df (pd.DataFrame): The main DataFrame.
    """
    corr_matrix = df.corr()
    p_values = calculate_pvalues(df)
    annot = corr_matrix.applymap(lambda x: f'{x:.2f}')
    annot[p_values >= 0.05] = ''
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt="s", cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation Matrix: Metrics vs. Following Night's Sleep (p < 0.05)", fontsize=16)
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.show()

def plot_previous_sleep_correlation(df_lagged):
    """
    Plots a correlation matrix of daily metrics vs. the previous night's sleep,
    showing only statistically significant (p < 0.05) correlations.

    Args:
        df_lagged (pd.DataFrame): The DataFrame with lagged sleep data.
    """
    sleep_cols = ['SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP']
    other_cols = [col for col in df_lagged.columns if col not in sleep_cols and 'PREV_NIGHT' not in col]
    cols_for_corr = other_cols + [f'PREV_NIGHT_{col}' for col in sleep_cols]
    df_for_corr = df_lagged[cols_for_corr]

    corr_matrix = df_for_corr.corr()
    p_values = calculate_pvalues(df_for_corr)
    
    corr_of_interest = corr_matrix[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]
    p_values_of_interest = p_values[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]

    annot = corr_of_interest.applymap(lambda x: f'{x:.2f}')
    annot[p_values_of_interest >= 0.05] = ''
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(corr_of_interest, annot=annot, fmt="s", cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation Matrix: Daily Metrics vs. Previous Night's Sleep (p < 0.05)", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.35)
    plt.show()

def show_interactive_plots(df, df_lagged):
    """
    Displays optional interactive correlation plots using Plotly.

    Args:
        df (pd.DataFrame): The main DataFrame.
        df_lagged (pd.DataFrame): The DataFrame with lagged sleep data.
    """
    # Interactive plot for future sleep correlation
    fig = px.imshow(df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title="Interactive Correlation: Metrics vs. Following Night's Sleep", height=800, width=800)
    fig.show()

    # Interactive plot for previous sleep correlation
    sleep_cols = ['SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP']
    other_cols = [col for col in df.columns if col not in sleep_cols]
    cols_for_corr = other_cols + [f'PREV_NIGHT_{col}' for col in sleep_cols]
    corr_of_interest = df_lagged[cols_for_corr].corr()[other_cols].loc[[f'PREV_NIGHT_{col}' for col in sleep_cols]]
    
    fig_lagged = px.imshow(corr_of_interest, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_lagged.update_layout(title="Interactive Correlation: Daily Metrics vs. Previous Night's Sleep", height=500, width=900)
    fig_lagged.show()

# --- Main Execution Block ---
def main():
    """
    Main function to run the entire health data analysis pipeline.
    """
    # 1. Load and prepare data
    health_df = load_and_prepare_data(FILEPATH, COLS_TO_USE)
    
    # 2. Perform and display summary analysis
    summary_stats = calculate_summary_statistics(health_df)
    print_detailed_summary(summary_stats, UNITS)
    
    # 3. Generate static visualizations
    plot_distributions(health_df, UNITS)
    plot_normalized_stats_heatmap(summary_stats)
    
    # 4. Perform correlation analysis
    plot_future_sleep_correlation(health_df)
    
    # 5. Create lagged data for previous night's sleep analysis
    health_df_lagged = create_lagged_sleep_data(health_df)
    plot_previous_sleep_correlation(health_df_lagged)
    
    # 6. Show optional interactive plots
    show_interactive_plots(health_df, health_df_lagged)
    
if __name__ == "__main__":
    main()