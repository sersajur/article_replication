"""
Replication of Table 5: Attacks and Next-day News Pressure
Driven by Predictable Political and Sports Events

Structure:
    Panel A: Israeli attacks & predictable events
    Panel B: Palestinian attacks & predictable events  
    Panel C: Placebo - Israeli attacks & disasters
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as LinearIV2SLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load replication data (.dta or .csv)"""
    if filepath.endswith('.dta'):
        return pd.read_stata(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError("File must be .dta or .csv")

# ============================================================================
# VARIABLE CREATION
# ============================================================================

def create_variables(df):
    """Create necessary variables for analysis"""
    
    # Month dummies (mon2-mon12)
    month_dummies = pd.get_dummies(df['month'], prefix='mon', drop_first=True)
    
    # Year dummies (year2-year12)
    year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
    
    # Day of week dummies (day2-day7)
    dow_dummies = pd.get_dummies(df['dow'], prefix='day', drop_first=True)
    
    # Combine
    df = pd.concat([df, month_dummies, year_dummies, dow_dummies], axis=1)
    
    return df

# ============================================================================
# FIRST STAGE REGRESSIONS
# ============================================================================

def first_stage(df, dep_var, instrument, controls, cluster_var):
    """
    OLS first stage
    dep_var: 'leaddaily_woi' or 'leaddaily_woi_nc'
    instrument: 'lead_maj_events' or 'lead_disaster'
    """
    
    # Build formula
    X_vars = [instrument] + controls
    X = sm.add_constant(df[X_vars])
    y = df[dep_var]
    
    # Fit model
    model = sm.OLS(y, X, missing='drop')
    
    # Cluster-robust SE
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': df[cluster_var]})
    
    # F-statistic for instrument
    f_stat = results.f_test(instrument + '=0').fvalue[0][0]
    
    return results, f_stat

# ============================================================================
# SECOND STAGE IV (2SLS)
# ============================================================================

def second_stage_occurrence(df, dep_var, endog_var, instrument, 
                           controls, cluster_var):
    """
    2SLS for binary outcome (occurrence)
    """
    
    # Prepare data
    y = df[dep_var].values
    X_exog = sm.add_constant(df[controls]).values
    X_endog = df[[endog_var]].values
    X_instruments = df[[instrument]].values
    
    # Drop missing
    mask = ~(np.isnan(y) | np.isnan(X_endog).any(axis=1) | 
             np.isnan(X_exog).any(axis=1))
    
    y = y[mask]
    X_exog = X_exog[mask]
    X_endog = X_endog[mask]
    X_instruments = X_instruments[mask]
    clusters = df[cluster_var].values[mask]
    
    # Fit 2SLS
    model = IV2SLS(y, X_exog, X_endog, X_instruments)
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': clusters})
    
    return results

# ============================================================================
# NEGATIVE BINOMIAL (for victims count)
# ============================================================================

def negbin_iv(df, dep_var, endog_var, instrument, controls, cluster_var):
    """
    Negative binomial with IV
    Uses control function approach
    """
    
    # Step 1: First stage residuals
    first_stage_model = sm.OLS(df[endog_var], 
                               sm.add_constant(df[[instrument] + controls]))
    fs_results = first_stage_model.fit()
    residuals = fs_results.resid
    
    # Step 2: Include residuals in NB model
    X_vars = [endog_var] + controls + ['_residuals']
    df_temp = df.copy()
    df_temp['_residuals'] = residuals
    
    X = sm.add_constant(df_temp[X_vars])
    y = df_temp[dep_var]
    
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), 
                   missing='drop')
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': df_temp[cluster_var]})
    
    return results

# ============================================================================
# REDUCED FORM (OLS / GLM)
# ============================================================================

def reduced_form_occurrence(df, dep_var, instrument, controls, cluster_var):
    """OLS reduced form for occurrence"""
    
    X_vars = [instrument] + controls
    X = sm.add_constant(df[X_vars])
    y = df[dep_var]
    
    model = sm.OLS(y, X, missing='drop')
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': df[cluster_var]})
    
    return results

def reduced_form_victims(df, dep_var, instrument, controls, cluster_var):
    """GLM negative binomial reduced form for victims"""
    
    X_vars = [instrument] + controls
    X = sm.add_constant(df[X_vars])
    y = df[dep_var]
    
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), 
                   missing='drop')
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': df[cluster_var]})
    
    return results

# ============================================================================
# PANEL A: ISRAELI ATTACKS & PREDICTABLE EVENTS
# ============================================================================

def panel_a(df):
    """
    Panel A: Israeli Attacks and Predictable Newsworthy Events
    """
    
    print("\n" + "="*70)
    print("PANEL A: ISRAELI ATTACKS & PREDICTABLE EVENTS")
    print("="*70)
    
    # Filter: no Gaza war
    df_a = df[df['gaza_war'] == 0].copy()
    
    # Controls
    controls_isr = ['occurrence_pal_1', 'occurrence_pal_2_7', 
                    'occurrence_pal_8_14']
    
    # Add FE dummies
    month_cols = [c for c in df_a.columns if c.startswith('mon')]
    year_cols = [c for c in df_a.columns if c.startswith('year')]
    dow_cols = [c for c in df_a.columns if c.startswith('day')]
    
    controls_full = controls_isr + month_cols + year_cols + dow_cols
    
    cluster = 'monthyear'
    
    # Column 1: First stage - corrected news pressure
    print("\n[Col 1] First stage: P(t+1) ~ Political/Sports events")
    fs1, f1 = first_stage(df_a, 'leaddaily_woi', 'lead_maj_events', 
                         controls_full, cluster)
    print(f"  Coef: {fs1.params['lead_maj_events']:.3f}")
    print(f"  SE:   {fs1.bse['lead_maj_events']:.3f}")
    print(f"  F-stat: {f1:.2f}")
    
    # Column 2: First stage - uncorrected news pressure
    print("\n[Col 2] First stage: Uncorrected P(t+1) ~ events")
    fs2, f2 = first_stage(df_a, 'leaddaily_woi_nc', 'lead_maj_events',
                         controls_full, cluster)
    print(f"  Coef: {fs2.params['lead_maj_events']:.3f}")
    print(f"  SE:   {fs2.bse['lead_maj_events']:.3f}")
    print(f"  F-stat: {f2:.2f}")
    
    # Column 3: 2SLS - occurrence (corrected)
    print("\n[Col 3] 2SLS: Occurrence ~ P(t+1)")
    ss3 = second_stage_occurrence(df_a, 'occurrence', 'leaddaily_woi',
                                  'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {ss3.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {ss3.bse['leaddaily_woi']:.3f}")
    
    # Column 4: 2SLS - occurrence (uncorrected)
    print("\n[Col 4] 2SLS: Occurrence ~ Uncorrected P(t+1)")
    ss4 = second_stage_occurrence(df_a, 'occurrence', 'leaddaily_woi_nc',
                                  'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {ss4.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {ss4.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 5: Reduced form - occurrence
    print("\n[Col 5] Reduced form: Occurrence ~ events")
    rf5 = reduced_form_occurrence(df_a, 'occurrence', 'lead_maj_events',
                                  controls_full, cluster)
    print(f"  Coef: {rf5.params['lead_maj_events']:.3f}")
    print(f"  SE:   {rf5.bse['lead_maj_events']:.3f}")
    
    # Column 6: 2SLS - victims (corrected) [Negative Binomial]
    print("\n[Col 6] NB-IV: Victims ~ P(t+1)")
    nb6 = negbin_iv(df_a, 'victims_isr', 'leaddaily_woi', 'lead_maj_events',
                    controls_full, cluster)
    print(f"  Coef: {nb6.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {nb6.bse['leaddaily_woi']:.3f}")
    
    # Column 7: 2SLS - victims (uncorrected)
    print("\n[Col 7] NB-IV: Victims ~ Uncorrected P(t+1)")
    nb7 = negbin_iv(df_a, 'victims_isr', 'leaddaily_woi_nc', 
                    'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {nb7.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {nb7.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 8: Reduced form - victims
    print("\n[Col 8] Reduced form: Victims ~ events")
    rf8 = reduced_form_victims(df_a, 'victims_isr', 'lead_maj_events',
                               controls_full, cluster)
    print(f"  Coef: {rf8.params['lead_maj_events']:.3f}")
    print(f"  SE:   {rf8.bse['lead_maj_events']:.3f}")

# ============================================================================
# PANEL B: PALESTINIAN ATTACKS & PREDICTABLE EVENTS
# ============================================================================

def panel_b(df):
    """
    Panel B: Palestinian Attacks and Predictable Newsworthy Events
    """
    
    print("\n" + "="*70)
    print("PANEL B: PALESTINIAN ATTACKS & PREDICTABLE EVENTS")
    print("="*70)
    
    # Filter: no Gaza war
    df_b = df[df['gaza_war'] == 0].copy()
    
    # Controls (different from Panel A)
    controls_pal = ['occurrence_1', 'occurrence_2_7', 'occurrence_8_14']
    
    # No month FE for Palestinian attacks
    year_cols = [c for c in df_b.columns if c.startswith('year')]
    dow_cols = [c for c in df_b.columns if c.startswith('day')]
    
    controls_full = controls_pal + year_cols + dow_cols
    
    cluster = 'monthyear'
    
    # Column 1: First stage
    print("\n[Col 1] First stage: P(t+1) ~ events")
    fs1, f1 = first_stage(df_b, 'leaddaily_woi', 'lead_maj_events',
                         controls_full, cluster)
    print(f"  Coef: {fs1.params['lead_maj_events']:.3f}")
    print(f"  SE:   {fs1.bse['lead_maj_events']:.3f}")
    print(f"  F-stat: {f1:.2f}")
    
    # Column 2: First stage - uncorrected
    print("\n[Col 2] First stage: Uncorrected P(t+1) ~ events")
    fs2, f2 = first_stage(df_b, 'leaddaily_woi_nc', 'lead_maj_events',
                         controls_full, cluster)
    print(f"  Coef: {fs2.params['lead_maj_events']:.3f}")
    print(f"  SE:   {fs2.bse['lead_maj_events']:.3f}")
    print(f"  F-stat: {f2:.2f}")
    
    # Column 3: 2SLS - occurrence
    print("\n[Col 3] 2SLS: Occurrence ~ P(t+1)")
    ss3 = second_stage_occurrence(df_b, 'occurrence_pal', 'leaddaily_woi',
                                  'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {ss3.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {ss3.bse['leaddaily_woi']:.3f}")
    
    # Column 4: 2SLS - occurrence (uncorrected)
    print("\n[Col 4] 2SLS: Occurrence ~ Uncorrected P(t+1)")
    ss4 = second_stage_occurrence(df_b, 'occurrence_pal', 'leaddaily_woi_nc',
                                  'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {ss4.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {ss4.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 5: Reduced form
    print("\n[Col 5] Reduced form: Occurrence ~ events")
    rf5 = reduced_form_occurrence(df_b, 'occurrence_pal', 'lead_maj_events',
                                  controls_full, cluster)
    print(f"  Coef: {rf5.params['lead_maj_events']:.3f}")
    print(f"  SE:   {rf5.bse['lead_maj_events']:.3f}")
    
    # Column 6: 2SLS - victims
    print("\n[Col 6] NB-IV: Victims ~ P(t+1)")
    nb6 = negbin_iv(df_b, 'victims_pal', 'leaddaily_woi', 'lead_maj_events',
                    controls_full, cluster)
    print(f"  Coef: {nb6.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {nb6.bse['leaddaily_woi']:.3f}")
    
    # Column 7: 2SLS - victims (uncorrected)
    print("\n[Col 7] NB-IV: Victims ~ Uncorrected P(t+1)")
    nb7 = negbin_iv(df_b, 'victims_pal', 'leaddaily_woi_nc',
                    'lead_maj_events', controls_full, cluster)
    print(f"  Coef: {nb7.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {nb7.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 8: Reduced form - victims
    print("\n[Col 8] Reduced form: Victims ~ events")
    rf8 = reduced_form_victims(df_b, 'victims_pal', 'lead_maj_events',
                               controls_full, cluster)
    print(f"  Coef: {rf8.params['lead_maj_events']:.3f}")
    print(f"  SE:   {rf8.bse['lead_maj_events']:.3f}")

# ============================================================================
# PANEL C: PLACEBO - ISRAELI ATTACKS & DISASTERS
# ============================================================================

def panel_c(df):
    """
    Panel C (Placebo): Israeli Attacks and Unpredictable Events
    """
    
    print("\n" + "="*70)
    print("PANEL C: PLACEBO - ISRAELI ATTACKS & DISASTERS")
    print("="*70)
    
    # Filter: no Gaza war
    df_c = df[df['gaza_war'] == 0].copy()
    
    # Controls (same as Panel A)
    controls_isr = ['occurrence_pal_1', 'occurrence_pal_2_7',
                    'occurrence_pal_8_14']
    
    month_cols = [c for c in df_c.columns if c.startswith('mon')]
    year_cols = [c for c in df_c.columns if c.startswith('year')]
    dow_cols = [c for c in df_c.columns if c.startswith('day')]
    
    controls_full = controls_isr + month_cols + year_cols + dow_cols
    
    cluster = 'monthyear'
    
    # Column 1: First stage - DISASTER instrument
    print("\n[Col 1] First stage: P(t+1) ~ Disaster onset")
    fs1, f1 = first_stage(df_c, 'leaddaily_woi', 'lead_disaster',
                         controls_full, cluster)
    print(f"  Coef: {fs1.params['lead_disaster']:.3f}")
    print(f"  SE:   {fs1.bse['lead_disaster']:.3f}")
    print(f"  F-stat: {f1:.2f}")
    
    # Column 2: First stage - uncorrected
    print("\n[Col 2] First stage: Uncorrected P(t+1) ~ Disaster")
    fs2, f2 = first_stage(df_c, 'leaddaily_woi_nc', 'lead_disaster',
                         controls_full, cluster)
    print(f"  Coef: {fs2.params['lead_disaster']:.3f}")
    print(f"  SE:   {fs2.bse['lead_disaster']:.3f}")
    print(f"  F-stat: {f2:.2f}")
    
    # Column 3: 2SLS - occurrence
    print("\n[Col 3] 2SLS: Occurrence ~ P(t+1)")
    ss3 = second_stage_occurrence(df_c, 'occurrence', 'leaddaily_woi',
                                  'lead_disaster', controls_full, cluster)
    print(f"  Coef: {ss3.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {ss3.bse['leaddaily_woi']:.3f}")
    
    # Column 4: 2SLS - occurrence (uncorrected)
    print("\n[Col 4] 2SLS: Occurrence ~ Uncorrected P(t+1)")
    ss4 = second_stage_occurrence(df_c, 'occurrence', 'leaddaily_woi_nc',
                                  'lead_disaster', controls_full, cluster)
    print(f"  Coef: {ss4.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {ss4.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 5: Reduced form
    print("\n[Col 5] Reduced form: Occurrence ~ Disaster")
    rf5 = reduced_form_occurrence(df_c, 'occurrence', 'lead_disaster',
                                  controls_full, cluster)
    print(f"  Coef: {rf5.params['lead_disaster']:.3f}")
    print(f"  SE:   {rf5.bse['lead_disaster']:.3f}")
    
    # Column 6: 2SLS - victims
    print("\n[Col 6] NB-IV: Victims ~ P(t+1)")
    nb6 = negbin_iv(df_c, 'victims_isr', 'leaddaily_woi', 'lead_disaster',
                    controls_full, cluster)
    print(f"  Coef: {nb6.params['leaddaily_woi']:.3f}")
    print(f"  SE:   {nb6.bse['leaddaily_woi']:.3f}")
    
    # Column 7: 2SLS - victims (uncorrected)
    print("\n[Col 7] NB-IV: Victims ~ Uncorrected P(t+1)")
    nb7 = negbin_iv(df_c, 'victims_isr', 'leaddaily_woi_nc', 'lead_disaster',
                    controls_full, cluster)
    print(f"  Coef: {nb7.params['leaddaily_woi_nc']:.3f}")
    print(f"  SE:   {nb7.bse['leaddaily_woi_nc']:.3f}")
    
    # Column 8: Reduced form - victims
    print("\n[Col 8] Reduced form: Victims ~ Disaster")
    rf8 = reduced_form_victims(df_c, 'victims_isr', 'lead_disaster',
                               controls_full, cluster)
    print(f"  Coef: {rf8.params['lead_disaster']:.3f}")
    print(f"  SE:   {rf8.bse['lead_disaster']:.3f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main replication function"""
    
    print("\n" + "="*70)
    print("TABLE 5 REPLICATION")
    print("Attacks and Next-day News Pressure Driven by")
    print("Predictable Political and Sports Events")
    print("="*70)
    
    # Load data
    # MODIFY PATH AS NEEDED
    data_path = "/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta"
    
    try:
        df = load_data(data_path)
        print(f"\n✓ Data loaded: {len(df)} observations")
    except FileNotFoundError:
        print(f"\n✗ Data file not found: {data_path}")
        print("Please provide the path to replication_file1.dta")
        return
    
    # Create variables
    df = create_variables(df)
    print("✓ Variables created")
    
    # Run panels
    panel_a(df)
    panel_b(df)
    panel_c(df)
    
    print("\n" + "="*70)
    print("REPLICATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
