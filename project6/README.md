# ESG Score Disagreement Analysis

## Research Question
How much do different ESG rating providers disagree?

## Methodology
**Language:** Python  
**Methods:** Cross-provider comparison, rank correlation

## Data
Simulated multi-provider scores calibrated to Berg et al. (2022)

## Key Findings
Provider correlations ~0.4–0.56; disagreement highest in industrials; significant rank reversals across providers.

## How to Run
```bash
pip install -r requirements.txt
python code/project6_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
