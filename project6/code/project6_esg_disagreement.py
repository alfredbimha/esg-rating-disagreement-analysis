"""
===============================================================================
PROJECT 6: ESG Score Disagreement Analysis Dashboard
===============================================================================
RESEARCH QUESTION:
    How much do different ESG rating providers disagree, and what drives
    the divergence?
METHOD:
    Compare ESG ratings across providers, compute disagreement metrics,
    analyze sector and pillar-level divergence.
DATA:
    Simulated multi-provider ESG scores (real scores require Bloomberg/WRDS)
    Calibrated to published research on ESG disagreement (Berg et al. 2022)
NOTE:
    Includes Streamlit dashboard code (app.py) for interactive deployment.
===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 1: Generate multi-provider ESG scores (calibrated to real divergence)
# =============================================================================
print("STEP 1: Generating multi-provider ESG scores...")
print("  (Calibrated to Berg, Kolbel & Rigobon 2022 divergence estimates)")

np.random.seed(42)
n_firms = 200

sectors = np.random.choice(
    ['Technology','Energy','Financials','Healthcare','Industrials',
     'Consumer','Utilities','Materials','Real Estate','Telecom'],
    n_firms
)

# True latent ESG quality (unobserved)
sector_base = {'Technology':65,'Energy':35,'Financials':55,'Healthcare':60,
               'Industrials':50,'Consumer':60,'Utilities':55,'Materials':40,
               'Real Estate':50,'Telecom':58}
true_esg = np.array([sector_base[s] for s in sectors]) + np.random.normal(0, 12, n_firms)
true_esg = np.clip(true_esg, 5, 95)

# Provider scores: each has different methodology → systematic disagreement
# Correlation between providers ~0.5-0.6 (as in Berg et al.)
providers = {
    'Provider_A (MSCI-style)': true_esg + np.random.normal(5, 15, n_firms),
    'Provider_B (Sustainalytics-style)': true_esg + np.random.normal(-3, 18, n_firms),
    'Provider_C (S&P-style)': true_esg + np.random.normal(0, 14, n_firms),
    'Provider_D (Refinitiv-style)': true_esg + np.random.normal(2, 16, n_firms),
}

tickers = [f'FIRM_{i:03d}' for i in range(n_firms)]

df = pd.DataFrame({'ticker': tickers, 'sector': sectors})
for prov, scores in providers.items():
    df[prov] = np.clip(scores, 0, 100).round(1)

# E, S, G pillar scores
for prov in providers:
    base = df[prov]
    df[f'{prov}_E'] = np.clip(base + np.random.normal(0, 10, n_firms), 0, 100).round(1)
    df[f'{prov}_S'] = np.clip(base + np.random.normal(-5, 12, n_firms), 0, 100).round(1)
    df[f'{prov}_G'] = np.clip(base + np.random.normal(3, 8, n_firms), 0, 100).round(1)

df.to_csv('data/esg_multi_provider.csv', index=False)
print(f"  Generated scores for {n_firms} firms across {len(providers)} providers")

# =============================================================================
# STEP 2: Disagreement metrics
# =============================================================================
print("\nSTEP 2: Computing disagreement metrics...")
prov_cols = [c for c in df.columns if c.startswith('Provider') and '_E' not in c and '_S' not in c and '_G' not in c]

df['esg_mean'] = df[prov_cols].mean(axis=1)
df['esg_std'] = df[prov_cols].std(axis=1)  # Standard deviation = disagreement
df['esg_range'] = df[prov_cols].max(axis=1) - df[prov_cols].min(axis=1)

# Rank disagreement
for p in prov_cols:
    df[f'{p}_rank'] = df[p].rank(pct=True) * 100
rank_cols = [c for c in df.columns if '_rank' in c]
df['rank_std'] = df[rank_cols].std(axis=1)

print(f"  Mean disagreement (std): {df['esg_std'].mean():.1f}")
print(f"  Mean range: {df['esg_range'].mean():.1f}")

# Provider correlations
corr = df[prov_cols].corr()
print(f"\n  Provider correlation matrix:")
print(corr.round(3).to_string())
corr.to_csv('output/tables/provider_correlations.csv')

# Sector-level disagreement
sector_disagree = df.groupby('sector')[['esg_std','esg_range']].mean().sort_values('esg_std', ascending=False)
sector_disagree.to_csv('output/tables/sector_disagreement.csv')
print(f"\n  Highest disagreement sector: {sector_disagree.index[0]} (std={sector_disagree['esg_std'].iloc[0]:.1f})")

# =============================================================================
# STEP 3: Visualizations
# =============================================================================
print("\nSTEP 3: Creating visualizations...")

# Fig 1: Provider correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['MSCI-style','Sustainalytics','S&P-style','Refinitiv']
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, 
            xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, linewidths=1)
ax.set_title('ESG Provider Score Correlations', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig1_provider_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Disagreement by sector
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sector_disagree)))
sector_disagree['esg_std'].plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_title('ESG Disagreement by Sector (Std Dev Across Providers)', fontweight='bold')
ax.set_ylabel('Average Std Dev of ESG Scores')
ax.set_xlabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/figures/fig2_sector_disagreement.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Pairwise scatter plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [(0,1), (0,2), (1,3)]
for idx, (i, j) in enumerate(pairs):
    ax = axes[idx]
    ax.scatter(df[prov_cols[i]], df[prov_cols[j]], alpha=0.4, s=20, 
              c=df['sector'].map(sector_base), cmap='RdYlGn')
    ax.plot([0,100],[0,100], 'r--', alpha=0.5)
    r = df[prov_cols[i]].corr(df[prov_cols[j]])
    ax.set_title(f'{labels[i]} vs {labels[j]}\nr={r:.2f}', fontweight='bold')
    ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('output/figures/fig3_pairwise_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Distribution of disagreement
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['esg_std'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(df['esg_std'].mean(), color='red', linestyle='--', label=f'Mean={df["esg_std"].mean():.1f}')
axes[0].set_title('Distribution of ESG Disagreement', fontweight='bold')
axes[0].set_xlabel('Std Dev Across Providers')
axes[0].legend()

axes[1].hist(df['esg_range'], bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].axvline(df['esg_range'].mean(), color='red', linestyle='--', label=f'Mean={df["esg_range"].mean():.1f}')
axes[1].set_title('Distribution of ESG Score Range', fontweight='bold')
axes[1].set_xlabel('Max - Min Score')
axes[1].legend()
plt.tight_layout()
plt.savefig('output/figures/fig4_disagreement_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
