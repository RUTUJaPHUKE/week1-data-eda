# Week 1 Task – Data Acquisition, Cleaning & Exploratory Analysis

## Overview
This repository contains the complete Week 1 data science deliverable covering:
- **Data Acquisition** – Titanic dataset sourced via Seaborn / GitHub
- **Data Cleaning** – Missing value imputation, duplicate removal, type corrections
- **Feature Engineering** – 4 derived features (family_size, is_alone, fare_log, age_group)
- **Exploratory Data Analysis** – Univariate, bivariate, and correlation analysis
- **Visualizations** – 6 annotated figures (missing values, survival analysis, distributions, correlations, age plots, scatter)
- **Report** – Full documentation in HTML and DOCX formats

## Dataset
**Titanic Passenger Records (1912)**
- 891 rows × 15 columns (raw)
- Source: [Seaborn datasets](https://github.com/mwaskom/seaborn-data) / [Kaggle](https://www.kaggle.com/competitions/titanic)
- Target variable: `survived` (binary: 0 = did not survive, 1 = survived)
- Overall survival rate: 38.4%

## Project Structure
```
Week1_EDA/
├── run_all.py                  # One-click runner – installs deps, runs analysis, builds report
├── RUN_ME.bat                  # Double-click to run on Windows
├── eda_analysis.py             # Full annotated EDA script
├── requirements.txt            # Python dependencies
├── Week1_EDA_Report.html       # Complete report (open in browser or Word)
├── titanic_raw.csv             # Raw downloaded dataset (generated on run)
├── titanic_cleaned.csv         # Cleaned + engineered dataset (generated on run)
└── visualizations/
    ├── fig1_missing_heatmap.png
    ├── fig2_survival_overview.png
    ├── fig3_distributions.png
    ├── fig4_correlation_heatmap.png
    ├── fig5_age_violin_box.png
    └── fig6_fare_vs_age_scatter.png
```

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis (downloads data, cleans, generates charts + .docx report)
python run_all.py
```
Or on Windows, simply **double-click `RUN_ME.bat`**.

## Key Findings
1. **Gender** was the strongest predictor — female survival 74.2% vs male 18.9%
2. **Passenger class** created a clear hierarchy — 1st: 63%, 2nd: 47%, 3rd: 24%
3. **Travelling alone** reduced survival — solo: 30.4% vs with family: 50.6%
4. **Children (0–12)** had the highest survival rate at 55.9%
5. **Log-transforming fare** reduced skewness from 4.79 → 0.45
6. **Missing data** in `deck` (77.2%) was MNAR; replaced with binary indicator

## Tools & Libraries
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Core language |
| pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical operations |
| Matplotlib | 3.7+ | Visualization |
| Seaborn | 0.13+ | Statistical plots |
| python-docx | 1.1+ | Word document generation |

## Author
Data Science Intern — August 2026
