"""
Table 1 Replication: 2SLS regression (Durante & Zhuravskaya)
"Attack When the World Is Not Watching? US News and the Israeli-Palestinian Conflict"
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
DATA_PATH = '/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta'

# === LOAD DATA ===
df = pd.read_stata(DATA_PATH)
print(f"Loaded: {len(df)} observations\n")

# === HELPER FUNCTIONS ===

def make_dummies(series):
    """Create dummy variables from any dtype (text or numeric)"""
    return pd.get_dummies(series, drop_first=True, dtype=float)

def run_2sls(data, dep_var, filter_attacks=False):
    """
    Run 2SLS regression
    Args:
        data: DataFrame with all variables
        dep_var: dependent variable ('daily_woi' or 'daily_woi_nc')
        filter_attacks: if True, keep only days with attacks (Panel B)
    Returns: dict with regression results
    """
    # Filter for Panel B (days with Israeli or Palestinian attacks)
    if filter_attacks:
        data = data[(data['occurrence_t_y']==1) | (data['occurrence_pal_t_y']==1)]
    
    # Select columns and drop missing values
    cols = ['high_intensity', 'length_conflict_news', dep_var, 'month', 'year', 'dow', 'monthyear']
    d = data[cols].dropna().copy()
    
    # Convert to numeric
    for col in ['high_intensity', 'length_conflict_news', dep_var]:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna()
    
    # Fixed effects (year, month, day-of-week dummies)
    fe = pd.concat([make_dummies(d['year']), make_dummies(d['month']), make_dummies(d['dow'])], axis=1)
    
    # --- FIRST STAGE: regress endogenous var on instrument ---
    y1 = d['length_conflict_news'].values
    X1 = sm.add_constant(pd.concat([d[['high_intensity']].reset_index(drop=True), fe.reset_index(drop=True)], axis=1)).astype(float)
    
    m1 = OLS(y1, X1).fit(cov_type='cluster', cov_kwds={'groups': d['monthyear'].values})
    
    # --- SECOND STAGE: regress outcome on fitted values ---
    y2 = d[dep_var].values
    X2 = sm.add_constant(pd.concat([pd.Series(m1.fittedvalues, name='length_hat'), fe.reset_index(drop=True)], axis=1)).astype(float)
    
    m2 = OLS(y2, X2).fit()

    # --- CORRECT R² CALCULATION (using original X, not fitted) ---
    # Get 2SLS coefficient
    beta_2sls = m2.params['length_hat']
    beta_const = m2.params['const']
    beta_fe = m2.params[2:].values  # FE coefficients

    # Predicted Y using ORIGINAL X (not X_hat)
    x_original = d['length_conflict_news'].values
    y_pred = beta_const + beta_2sls * x_original + np.dot(fe.values, beta_fe)

    # Residuals and R²
    residuals = y2 - y_pred
    SSR = np.sum(residuals ** 2)
    SST = np.sum((y2 - np.mean(y2)) ** 2)
    r2_correct = 1 - SSR / SST

    # F-statistic for instrument strength
    f_stat = (m1.params['high_intensity'] / m1.bse['high_intensity']) ** 2
    
    return {
        'n': int(m1.nobs),
        'coef_1st': m1.params['high_intensity'],
        'se_1st': m1.bse['high_intensity'],
        'r2_1st': m1.rsquared,
        'f_stat': f_stat,
        'coef_2nd': m2.params['length_hat'],
        'se_2nd': m2.bse['length_hat'],
        'pval_2nd': m2.pvalues['length_hat'],
        'r2_2nd': r2_correct
    }

def stars(pval):
    """Return significance stars"""
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.1: return '*'
    return ''

# === RUN REGRESSIONS ===

# Panel A: Full sample
a_corr = run_2sls(df, 'daily_woi')
a_uncorr = run_2sls(df, 'daily_woi_nc')

# Panel B: Days with attacks
b_corr = run_2sls(df, 'daily_woi', filter_attacks=True)
b_uncorr = run_2sls(df, 'daily_woi_nc', filter_attacks=True)

# === OUTPUT RESULTS ===

print("=" * 72)
print("TABLE 1: NEWS PRESSURE AND THE LENGTH OF CONFLICT-RELATED NEWS")
print("=" * 72)

print(f"""
                              (1)           (2)           (3)
Dependent variable:        Length of    Corrected    Uncorrected
                          conflict news  news pressure news pressure
Model:                     1st stage     2nd stage     2nd stage
────────────────────────────────────────────────────────────────────────
PANEL A: FULL SAMPLE
────────────────────────────────────────────────────────────────────────
Intifada, Defensive Shield   {a_corr['coef_1st']:6.3f}***
  and Cast Lead             ({a_corr['se_1st']:5.3f})

Length of conflict news                   {a_corr['coef_2nd']:6.3f}       {a_uncorr['coef_2nd']:6.3f}***
  (minutes)                              ({a_corr['se_2nd']:5.3f})      ({a_uncorr['se_2nd']:5.3f})

Observations                 {a_corr['n']:,}         {a_corr['n']:,}         {a_uncorr['n']:,}
R²                           {a_corr['r2_1st']:.3f}         {a_corr['r2_2nd']:.3f}         {a_uncorr['r2_2nd']:.3f}
F-stat, excl. instr.        {a_corr['f_stat']:6.2f}
────────────────────────────────────────────────────────────────────────
PANEL B: DAYS WITH ATTACK ON SAME/PREVIOUS DAY
────────────────────────────────────────────────────────────────────────
Intifada, Defensive Shield   {b_corr['coef_1st']:6.3f}***
  and Cast Lead             ({b_corr['se_1st']:5.3f})

Length of conflict news                   {b_corr['coef_2nd']:6.4f}       {b_uncorr['coef_2nd']:6.3f}***
  (minutes)                              ({b_corr['se_2nd']:5.3f})      ({b_uncorr['se_2nd']:5.3f})

Observations                 {b_corr['n']:,}         {b_corr['n']:,}         {b_uncorr['n']:,}
R²                           {b_corr['r2_1st']:.3f}         {b_corr['r2_2nd']:.3f}         {b_uncorr['r2_2nd']:.3f}
F-stat, excl. instr.        {b_corr['f_stat']:6.2f}
────────────────────────────────────────────────────────────────────────
Note: *** p<0.01, ** p<0.05, * p<0.1. Clustered SE by month×year.
""")

# === COMPARISON WITH ORIGINAL ===
print("\n" + "=" * 72)
print("COMPARISON WITH ORIGINAL (Panel A)")
print("=" * 72)
print(f"""
                          Original      Replication
────────────────────────────────────────────────────
1st stage coef             5.046         {a_corr['coef_1st']:.3f}
1st stage SE              (1.330)       ({a_corr['se_1st']:.3f})
F-stat                    14.40         {a_corr['f_stat']:.2f}

2nd stage (corrected)      0.002         {a_corr['coef_2nd']:.3f}
2nd stage (uncorrected)   -0.017         {a_uncorr['coef_2nd']:.3f}

Observations               4,003         {a_corr['n']:,}
────────────────────────────────────────────────────
""")
