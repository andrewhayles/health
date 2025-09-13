# Import the libraries that we will be using
import statsmodels.api as sm
import statsmodels
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as py
import scipy.stats as stats

da = pd.read_csv("health.csv")

vars = ["DATE",	"WEIGHT", "PCTBODYFAT",	"PCTWATER",	"VISCFATMASS",	"MUSCLEMASS",	"BONEMASS",	"BODYSCORE",	"BPAVGSYS",	"BPAVGDIA",	"BPMAXSYS",	"BPMAXDIA",	"BPMINSYS",
        "BPMINDIA",	"SLEEPTOTAL",	"DEEPSLEEP",	"LIGHTSLEEP",	"HRAVG",	"HRMAX",	"HRMIN",	"O2SATAVG",	"STRESSAVG",	"STRESSMAX",	"STRESSMIN",	"MISCNOTES"]

bpsys = da["BPAVGSYS"]
bpdia = da["BPAVGDIA"]
weight = da["WEIGHT"]

half = len(bpsys) // 2
end = len(bpsys)
halfweight = len(weight) // 2
endweight = len(weight)

bpsys_firsthalf = bpsys.iloc[:half].dropna()
bpsys_secondhalf = bpsys.iloc[half:end].dropna()
weight_firsthalf = weight.iloc[:halfweight].dropna()
weight_secondhalf = weight.iloc[halfweight:endweight].dropna()


print("First half values:", bpsys_firsthalf.unique())
print("Second half values:", bpsys_secondhalf.unique())

print("First half variance:", bpsys_firsthalf.var())
print("Second half variance:", bpsys_secondhalf.var())

print("First half/Second half t-test:", sm.stats.ttest_ind(bpsys_firsthalf, bpsys_secondhalf, alternative="larger", usevar="unequal"))

from statsmodels.stats.power import TTestIndPower
pooled_sd = np.sqrt(((len(bpsys_firsthalf)-1)*np.var(bpsys_firsthalf) + (len(bpsys_secondhalf)-1)*np.var(bpsys_secondhalf)) / (len(bpsys_firsthalf)+len(bpsys_secondhalf)-2))
cohens_d = (np.mean(bpsys_firsthalf) - np.mean(bpsys_secondhalf)) / pooled_sd
analysis = TTestIndPower()
power = analysis.solve_power(effect_size = cohens_d, nobs1=half, alpha=0.05)
print(f"Power: {power:.2f}")  # Should be >0.80

# After getting high p-value
from scipy.stats import ttest_ind

tstat, pval = ttest_ind(bpsys_firsthalf, bpsys_secondhalf)

if pval < 0.05:
    print(f"Significant difference (p={pval:.4f})")
else:
    # Calculate effect size
    pooled_sd = np.sqrt(((len(bpsys_firsthalf)-1)*np.var(bpsys_firsthalf) + (len(bpsys_secondhalf)-1)*np.var(bpsys_secondhalf)) / (len(bpsys_firsthalf)+len(bpsys_secondhalf)-2))
    cohens_d = (np.mean(bpsys_firsthalf) - np.mean(bpsys_secondhalf)) / pooled_sd
    
    print(f"No significant difference (p={pval:.4f}, d={cohens_d:.2f})")
    print(f"Practical interpretation: {'Negligible' if abs(cohens_d)<0.2 else 'Small'} effect size")
    
#Visualizations




sns.histplot(x=da["BPAVGSYS"], binwidth=2, kde=False)
sns.histplot(x=da["BPAVGDIA"], binwidth=2, kde=False).set_title("Histograms of Blood Pressure (Orange=Diastolic, Blue=Systolic)");
plt.show()
sm.qqplot(da["BPAVGSYS"])
py.show()
sm.qqplot(da["BPAVGDIA"])
py.show()

stats.probplot(da["BPAVGSYS"], dist="norm", plot=plt)
plt.show()

stats.probplot(da["BPAVGDIA"], dist="norm", plot=plt)
plt.show()


stressdf = da[["STRESSMIN", "STRESSAVG", "STRESSMAX"]].copy()
sleepdf = da[["SLEEPTOTAL", "DEEPSLEEP", "LIGHTSLEEP"]].copy()
bpsysdf = da[["BPAVGSYS", "BPMINSYS", "BPMAXSYS"]].copy()
bpdiadf = da[["BPAVGDIA", "BPMINDIA", "BPMAXDIA"]].copy()

sns.boxplot(data=stressdf, color=".8", linecolor="#137", linewidth=.75).set_xlabel("Stress")
plt.show()
sns.boxplot(data=sleepdf, color=".8", linecolor="#137", linewidth=.75).set_xlabel("Sleep")
plt.show()
sns.boxplot(data=bpsysdf, color=".8", linecolor="#137", linewidth=.75).set_xlabel("Systolic BP")
plt.show()
sns.boxplot(data=bpdiadf, color=".8", linecolor="#137", linewidth=.75).set_xlabel("Diastolic BP")
plt.show()
sns.boxplot(data=da, x=da["WEIGHT"], color=".8", linecolor="#137", linewidth=.75)
plt.show()


plt.figure(figsize=(12,4))
sns.violinplot(data=bpsysdf,  legend=False)
plt.show()

plt.figure(figsize=(12,4))
sns.violinplot(data=bpdiadf, legend=False)
plt.show()

plt.figure(figsize=(10,6))
sns.kdeplot(data=stressdf, 
            fill=True, common_norm=False, 
            palette='Spectral', alpha=0.8, linewidth=1.2,
            bw_adjust=0.8)
plt.title('Stress Levels', fontsize=16)
plt.xlabel('Stress', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.show()
    