# Project Title

> This project evaluates health data from a CSV file with a specific set of columns in it.

## About The Project

This evaluates a person's health CSV file and makes many graphical images of the data so that one can get a better idea of the quality of the data and the ranges that one falls in for personal health.

### Built With

* Smartwatch
* Smartphone
* Python

## Getting Started

All you need is a CSV file with the following columns: 'WEIGHT', 'BPAVGSYS', 'BPAVGDIA', 'BPMAXSYS', 'BPMAXDIA', 'BPMINSYS', 'BPMINDIA', 'SLEEPTOTAL', 'DEEPSLEEP', 'LIGHTSLEEP', 'HRAVG', 'HRMAX', 'HRMIN', 'STRESSAVG', 'STRESSMAX', 'STRESSMIN'

Put that CSV named "health_data.csv" into the same directory as the health_analysis_version3.py and run the script from the command prompt (some python libraries may need to be added to your system).

### Prerequisites

These are the imports in the script and so these libraries are needed to run the script:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
```
### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/andrewhayles/health.git
    ```
2.  Install packages
3.  Run the program


## Contact

Andrew Hayles (andyhayles@gmail.com)

Project Link: [https://github.com/andrewhayles/health](https://github.com/andrewhayles/health)