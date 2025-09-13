import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("health.csv", usecols=['WEIGHT', 'BPAVGSYS',	'BPAVGDIA',	'BPMAXSYS',	'BPMAXDIA',	'BPMINSYS',	
                                        'BPMINDIA',	'SLEEPTOTAL','DEEPSLEEP','LIGHTSLEEP','HRAVG','HRMAX','HRMIN',
                                        'STRESSAVG','STRESSMAX','STRESSMIN'])
                                        
# Basic statistics
summary = df.describe(percentiles=[.25, .5, .75]).T
summary['IQR'] = summary['75%'] - summary['25%']

# Add skewness and kurtosis
summary['skewness'] = df.skew()
summary['kurtosis'] = df.kurtosis()

# Display key statistics
print(summary[['mean', 'std', 'min', '50%', 'max', 'IQR', 'skewness', 'kurtosis']])
               
plt.figure(figsize=(16, 20))
plt.suptitle('Health Marker Distributions', fontsize=24, y=0.98)


for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution: {col}')
    plt.xlabel('')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect=[left, bottom, right, top]
plt.show()

plt.figure(figsize=(16, 10))
sns.boxplot(data=df.melt(var_name='Marker'), 
            x='Marker', 
            y='value')
plt.xticks(rotation=45)
plt.title('Comparison of Health Markers')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Normalize key statistics for visualization
stats_to_show = summary[['mean', 'std', '50%', 'IQR']]
normalized_stats = (stats_to_show - stats_to_show.min()) / (stats_to_show.max() - stats_to_show.min())

plt.figure(figsize=(12, 8))
sns.heatmap(normalized_stats.T, 
            annot=stats_to_show.T,
            fmt=".2f",
            cmap='viridis',
            cbar_kws={'label': 'Normalized Value'})
plt.title('Normalized Summary Statistics Comparison')
plt.xlabel('Health Markers')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 14))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)
plt.title('Health Marker Correlations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from tabulate import tabulate

# Generate table data
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

# Display as formatted table
print(tabulate(table_data,
               headers=['Marker', 'Mean ± SD', 'Range', 'Median', 'IQR', 'Skewness'],
               tablefmt='github',
               floatfmt=".2f"))
               
import plotly.express as px
from plotly.subplots import make_subplots

# Create interactive distributions
fig = make_subplots(rows=4, cols=4, subplot_titles=df.columns)
for i, col in enumerate(df.columns, 1):
    row = (i-1)//4 + 1
    col_pos = (i-1)%4 + 1
    fig.add_trace(px.histogram(df, x=col, nbins=50).data[0], row=row, col=col_pos)
fig.update_layout(height=900, width=1200, title_text="Health Marker Distributions")
fig.show()

# Interactive correlation matrix
fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig.update_layout(title="Health Marker Correlations", height=800, width=800)
fig.show()