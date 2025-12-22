"""
═══════════════════════════════════════════════════════════════════
TABLE 3 REPLICATION
Durante & Zhuravskaya (2018), Journal of Political Economy
"Attack When the World Is Not Watching? 
US News and the Israeli-Palestinian Conflict"
═══════════════════════════════════════════════════════════════════

USAGE:
    python table3_replication.py

OUTPUT:
    - table3_results.csv
    - table3_results.xlsx
    - diagnostics_plots.png

REQUIREMENTS:
    pip install pandas numpy statsmodels scipy pyreadstat openpyxl matplotlib

═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════

def load_data(path):
    """Load data and create fixed effects"""
    # Load
    df = pd.read_stata(path) if path.endswith('.dta') else pd.read_csv(path)
    
    # Filter: exclude Gaza War period
    df = df[df['gaza_war'] == 0].copy()
    
    # Create fixed effects (dummy variables)
    for var in ['month', 'year', 'dow']:
        dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
        # Convert to integer (0/1)
        dummies = dummies.astype(int)
        df = pd.concat([df, dummies], axis=1)
    
    # Cluster variable for standard errors
    df['cluster'] = df['year'].astype(str) + '_' + df['month'].astype(str)
    
    return df

# ═══════════════════════════════════════════════════════════════════
# REGRESSION MODELS
# ═══════════════════════════════════════════════════════════════════

def run_ols_cluster(y, X, cluster, data):
    """OLS regression with clustered standard errors"""
    clean = data[[y] + X + [cluster]].dropna()
    X_mat = sm.add_constant(clean[X].astype(float))
    y_vec = clean[y].astype(float)
    return OLS(y_vec, X_mat).fit(cov_type='cluster', 
                                  cov_kwds={'groups': clean[cluster]})

def run_ols_nw(y, X, data, lags=7):
    """OLS regression with Newey-West standard errors"""
    clean = data[[y] + X].dropna()
    X_mat = sm.add_constant(clean[X].astype(float))
    y_vec = clean[y].astype(float)
    return OLS(y_vec, X_mat).fit(cov_type='HAC', 
                                  cov_kwds={'maxlags': lags})

def run_negbin_nw(y, X, data, lags=7):
    """Negative Binomial regression with Newey-West SE"""
    clean = data[[y] + X].dropna()
    X_mat = sm.add_constant(clean[X].astype(float))
    y_vec = clean[y].astype(float)
    return NegativeBinomial(y_vec, X_mat).fit(cov_type='HAC', 
                                               cov_kwds={'maxlags': lags})

# ═══════════════════════════════════════════════════════════════════
# TABLE 3 ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def estimate_panel(data, suffix=''):
    """
    Estimate one panel (7 columns)
    
    Parameters:
    -----------
    suffix : str
        '' for Panel A (corrected)
        '_nc' for Panel B (uncorrected)
    
    Returns:
    --------
    dict : Results for columns 1-7
    """
    # Get fixed effects columns
    fe = [c for c in data.columns if c.startswith(('month_', 'year_', 'dow_'))]
    
    # Lag variables
    lags = [f'lagdaily_woi{i}{suffix}' for i in ['', '2', '3', '4', '5', '6', '7']]
    
    # Palestinian attack controls
    pal = ['occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14']
    
    # News pressure variables
    np_t = f'daily_woi{suffix}'
    np_t1 = f'leaddaily_woi{suffix}'
    
    # Estimate 7 columns
    return {
        # Occurrence regressions
        'c1': run_ols_cluster('occurrence', [np_t] + fe, 'cluster', data),
        'c2': run_ols_nw('occurrence', [np_t, np_t1] + fe, data),
        'c3': run_ols_nw('occurrence', [np_t, np_t1] + lags + pal + fe, data),
        
        # Ln(victims) regressions
        'c4': run_ols_cluster('lnvic', [np_t] + fe, 'cluster', data),
        'c5': run_ols_nw('lnvic', [np_t, np_t1] + fe, data),
        'c6': run_ols_nw('lnvic', [np_t, np_t1] + lags + pal + fe, data),
        
        # Count regression
        'c7': run_negbin_nw('victims_isr', [np_t, np_t1] + lags + pal + fe, data)
    }

# ═══════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_coef(coef, se):
    """Format coefficient with significance stars"""
    p = 2 * (1 - stats.norm.cdf(abs(coef / se)))
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    return f"{coef:.3f}{stars}", f"({se:.3f})"

def create_table(pa, pb):
    """Create formatted output table"""
    # Variables to display
    vars_a = ['daily_woi', 'leaddaily_woi', 'lagdaily_woi',
              'occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14']
    vars_b = [v + '_nc' if 'woi' in v else v for v in vars_a]
    
    rows = []
    
    # ───────────────────────────────────────────────────────────────
    # PANEL A: CORRECTED NEWS PRESSURE
    # ───────────────────────────────────────────────────────────────
    rows.append(['PANEL A: CORRECTED NEWS PRESSURE'] + [''] * 7)
    rows.append([''] * 8)
    
    for var in vars_a:
        # Coefficient row
        row = [var]
        for i in range(1, 8):
            if var in pa[f'c{i}'].params:
                row.append(format_coef(pa[f'c{i}'].params[var], 
                                      pa[f'c{i}'].bse[var])[0])
            else:
                row.append('')
        rows.append(row)
        
        # Standard error row
        row_se = ['']
        for i in range(1, 8):
            if var in pa[f'c{i}'].params:
                row_se.append(format_coef(pa[f'c{i}'].params[var], 
                                         pa[f'c{i}'].bse[var])[1])
            else:
                row_se.append('')
        rows.append(row_se)
    
    # Summary statistics
    rows.append([''] * 8)
    rows.append(['Observations'] + [f"{int(pa[f'c{i}'].nobs):,}" 
                                     for i in range(1, 8)])
    rows.append(['R-squared'] + 
                [f"{pa[f'c{i}'].rsquared:.3f}" if hasattr(pa[f'c{i}'], 'rsquared') 
                 else f"{pa[f'c{i}'].prsquared:.3f}" for i in range(1, 8)])
    
    # ───────────────────────────────────────────────────────────────
    # PANEL B: UNCORRECTED NEWS PRESSURE
    # ───────────────────────────────────────────────────────────────
    rows.append([''] * 8)
    rows.append(['PANEL B: UNCORRECTED NEWS PRESSURE'] + [''] * 7)
    rows.append([''] * 8)
    
    for var in vars_b:
        # Coefficient row
        row = [var]
        for i in range(1, 8):
            if var in pb[f'c{i}'].params:
                row.append(format_coef(pb[f'c{i}'].params[var], 
                                      pb[f'c{i}'].bse[var])[0])
            else:
                row.append('')
        rows.append(row)
        
        # Standard error row
        row_se = ['']
        for i in range(1, 8):
            if var in pb[f'c{i}'].params:
                row_se.append(format_coef(pb[f'c{i}'].params[var], 
                                         pb[f'c{i}'].bse[var])[1])
            else:
                row_se.append('')
        rows.append(row_se)
    
    # Summary statistics
    rows.append([''] * 8)
    rows.append(['Observations'] + [f"{int(pb[f'c{i}'].nobs):,}" 
                                     for i in range(1, 8)])
    rows.append(['R-squared'] + 
                [f"{pb[f'c{i}'].rsquared:.3f}" if hasattr(pb[f'c{i}'], 'rsquared') 
                 else f"{pb[f'c{i}'].prsquared:.3f}" for i in range(1, 8)])
    
    # Create DataFrame
    cols = ['Variable', 'Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7']
    return pd.DataFrame(rows, columns=cols)

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def print_diagnostics(df):
    """Print data diagnostics"""
    print("\n" + "═"*70)
    print("DATA DIAGNOSTICS")
    print("═"*70)
    
    # Basic info
    print(f"\nRows: {len(df):,}")
    print(f"Period: {df['date'].min()} to {df['date'].max()}")
    print(f"Gaza War excluded: {(df['gaza_war']==1).sum():,} observations\n")
    
    # Missing values
    print("Missing values:")
    for v in ['occurrence', 'victims_isr', 'daily_woi', 'leaddaily_woi']:
        if v in df.columns:
            m = df[v].isna().sum()
            print(f"  {v:20s}: {m:6,} ({m/len(df)*100:5.2f}%)")
    
    # Descriptive statistics
    df_clean = df[df['gaza_war'] == 0]
    print("\nDescriptive Statistics:")
    vars = ['occurrence', 'victims_isr', 'daily_woi', 'leaddaily_woi']
    stats_data = []
    for v in vars:
        if v in df_clean.columns:
            stats_data.append({
                'Variable': v,
                'Mean': df_clean[v].mean(),
                'SD': df_clean[v].std(),
                'Min': df_clean[v].min(),
                'Max': df_clean[v].max(),
                'N': df_clean[v].notna().sum()
            })
    print(pd.DataFrame(stats_data).to_string(index=False))

def create_diagnostic_plots(df, save='diagnostics_plots.png'):
    """Create diagnostic plots"""
    df_clean = df[df['gaza_war'] == 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Diagnostic Plots', fontsize=16, fontweight='bold')
    
    # Plot 1: Occurrence distribution
    df_clean['occurrence'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('A. Attack Occurrence', fontweight='bold')
    axes[0,0].set_xlabel('Occurrence')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Victims distribution
    df_clean['victims_isr'].hist(bins=30, ax=axes[0,1], color='coral', edgecolor='black')
    axes[0,1].set_title('B. Number of Victims', fontweight='bold')
    axes[0,1].set_xlabel('Victims')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Plot 3: News pressure at t
    df_clean['daily_woi'].hist(bins=30, ax=axes[1,0], color='lightgreen', edgecolor='black')
    axes[1,0].set_title('C. News Pressure (t)', fontweight='bold')
    axes[1,0].set_xlabel('News Pressure')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Plot 4: News pressure at t+1
    df_clean['leaddaily_woi'].hist(bins=30, ax=axes[1,1], color='plum', edgecolor='black')
    axes[1,1].set_title('D. News Pressure (t+1)', fontweight='bold')
    axes[1,1].set_xlabel('News Pressure')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"✓ Diagnostic plots saved: {save}")

# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def replicate_table3(data_path):
    """
    Main replication function
    
    Steps:
    1. Load and prepare data
    2. Run diagnostics
    3. Estimate Panel A (corrected news pressure)
    4. Estimate Panel B (uncorrected news pressure)
    5. Format and save results
    """
    print("\n" + "═"*70)
    print("TABLE 3 REPLICATION")
    print("Israeli Attacks and News Pressure")
    print("═"*70 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    data = load_data(data_path)
    print(f"✓ Loaded {len(data):,} observations\n")
    
    # Step 2: Diagnostics
    print("Step 2: Running diagnostics...")
    print_diagnostics(pd.read_stata(data_path) if data_path.endswith('.dta') 
                     else pd.read_csv(data_path))
    create_diagnostic_plots(pd.read_stata(data_path) if data_path.endswith('.dta') 
                           else pd.read_csv(data_path))
    print()
    
    # Step 3: Estimate Panel A
    print("\nStep 3: Estimating Panel A (corrected news pressure)...")
    panel_a = estimate_panel(data, suffix='')
    print("✓ Panel A complete (7 columns)\n")
    
    # Step 4: Estimate Panel B
    print("Step 4: Estimating Panel B (uncorrected news pressure)...")
    panel_b = estimate_panel(data, suffix='_nc')
    print("✓ Panel B complete (7 columns)\n")
    
    # Step 5: Format and save
    print("Step 5: Formatting results...")
    table = create_table(panel_a, panel_b)
    print("✓ Formatting complete\n")
    
    print("Step 6: Saving results...")
    table.to_csv('table3_results.csv', index=False)
    table.to_excel('table3_results.xlsx', index=False)
    print("✓ Results saved:")
    print("  - table3_results.csv")
    print("  - table3_results.xlsx")
    print("  - diagnostics_plots.png\n")
    
    # Print results
    print("═"*70)
    print("RESULTS")
    print("═"*70 + "\n")
    print(table.to_string(index=False))
    print("\n" + "═"*70)
    print("REPLICATION COMPLETE")
    print("═"*70 + "\n")
    
    return table, panel_a, panel_b

# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Data file path
    DATA_FILE = "/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta"
    
    # Run replication
    table, panel_a, panel_b = replicate_table3(DATA_FILE)
    
    # Additional analysis examples:
    # print(panel_a['c2'].summary())  # View column 2 details
    # print(panel_a['c2'].params)     # View coefficients
