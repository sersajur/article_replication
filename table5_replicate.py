"""
Replication of Table 5: Attacks and Next-Day News Pressure 
Driven by Predictable Political and Sports Events

Source: Durante & Zhuravskaya (JPE)

Key methodological notes:
- Columns 1-2: OLS first stage
- Columns 3-4: 2SLS second stage (linear)
- Column 5: OLS reduced form
- Columns 6-7: Control Function Approach for NB IV (equivalent to Stata's qvf)
- Column 8: NB reduced form (GLM)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for paths and variables."""

    DATA_PATH = "replication_file1.dta"
    OUTPUT_PATH = "table5_results.xlsx"

    # Fixed effects
    FE_FULL = ['month', 'year', 'dow']      # Panel A, C
    FE_PARTIAL = ['year', 'dow']            # Panel B (no month FE)

    # Prior attacks (controls)
    PRIOR_PAL = ['occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14']
    PRIOR_ISR = ['occurrence_1', 'occurrence_2_7', 'occurrence_8_14']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load and prepare data."""
    df = pd.read_stata(path)
    df = df.sort_values('date').reset_index(drop=True)

    # Convert key numeric variables
    numeric_cols = [
        'leaddaily_woi', 'leaddaily_woi_nc', 'daily_woi', 'daily_woi_nc',
        'lead_maj_events', 'lead_disaster',
        'occurrence', 'occurrence_pal',
        'occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14',
        'occurrence_1', 'occurrence_2_7', 'occurrence_8_14',
        'victims_isr', 'victims_pal',
        'gaza_war', 'month', 'year'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Note: dow can be text ('Monday', etc.) - pd.get_dummies will handle it

    # Ensure monthyear is suitable for clustering
    if 'monthyear' in df.columns:
        if df['monthyear'].dtype == 'object' or df['monthyear'].dtype.name == 'category':
            df['monthyear'] = pd.Categorical(df['monthyear']).codes

    return df


def filter_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Apply gaza_war==0 filter."""
    return df[df['gaza_war'] == 0].copy()


# =============================================================================
# FIXED EFFECTS GENERATION
# =============================================================================

def create_dummies(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Create dummy variables for fixed effects using pd.get_dummies."""
    result = df.copy()

    for var in cols:
        dummies = pd.get_dummies(result[var], prefix=var, drop_first=True)
        dummies = dummies.astype(float)
        result = pd.concat([result, dummies], axis=1)

    return result


def get_fe_columns(df: pd.DataFrame, fe_vars: list) -> list:
    """Get list of fixed effect dummy column names."""
    fe_cols = []
    for var in fe_vars:
        fe_cols += sorted([c for c in df.columns if c.startswith(f'{var}_')])
    return fe_cols


# =============================================================================
# REGRESSION MODELS
# =============================================================================

def ols_clustered(df: pd.DataFrame, y_var: str, x_vars: list,
                  cluster_var: str) -> dict:
    """OLS with clustered standard errors (Stata-style)."""
    all_vars = [y_var] + x_vars + [cluster_var]
    data = df[all_vars].copy()

    # Ensure numeric types
    for col in [y_var] + x_vars:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    y = data[y_var].astype(float)
    X = sm.add_constant(data[x_vars].astype(float))

    # Fit model with clustered SE
    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': data[cluster_var]}
    )

    # Stata-style R² (uses demeaned y for R² calculation)
    # R² = 1 - RSS/TSS where TSS = sum((y - mean(y))^2)
    y_mean = y.mean()
    tss = ((y - y_mean) ** 2).sum()
    rss = (model.resid ** 2).sum()
    r2_stata = 1 - rss / tss

    return {
        'coef': model.params,
        'se': model.bse,
        'pval': model.pvalues,
        'r2': r2_stata,  # Use Stata-style R²
        'nobs': int(model.nobs),
        'model': model,
        'resid': model.resid
    }


def iv_2sls(df: pd.DataFrame, y_var: str, endog_var: str,
            instrument: str, controls: list, cluster_var: str) -> dict:
    """2SLS IV regression with clustered standard errors."""
    all_vars = [y_var, endog_var, instrument] + controls + [cluster_var]
    data = df[all_vars].copy()

    # Ensure numeric types
    for col in [y_var, endog_var, instrument] + controls:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    y = data[y_var].astype(float)
    endog = data[[endog_var]].astype(float)
    instr = data[[instrument]].astype(float)
    exog = sm.add_constant(data[controls].astype(float))

    model = IV2SLS(y, exog, endog, instr).fit(
        cov_type='clustered',
        clusters=data[cluster_var]
    )

    return {
        'coef': model.params,
        'se': model.std_errors,
        'pval': model.pvalues,
        'r2': model.rsquared,
        'nobs': int(model.nobs),
        'model': model
    }


def first_stage_f(df: pd.DataFrame, endog_var: str, instrument: str,
                  controls: list, cluster_var: str) -> float:
    """Compute F-statistic for excluded instrument."""
    result = ols_clustered(df, endog_var, [instrument] + controls, cluster_var)
    t_stat = result['coef'][instrument] / result['se'][instrument]
    return t_stat ** 2


def nbreg_glm(df: pd.DataFrame, y_var: str, x_vars: list,
              cluster_var: str) -> dict:
    """Negative binomial regression (GLM) - for reduced form."""
    all_vars = [y_var] + x_vars + [cluster_var]
    data = df[all_vars].copy()

    for col in [y_var] + x_vars:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    y = data[y_var].astype(float)
    X = sm.add_constant(data[x_vars].astype(float))

    # NB2 GLM (Stata's default)
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit(
        cov_type='cluster',
        cov_kwds={'groups': data[cluster_var]}
    )

    return {
        'coef': model.params,
        'se': model.bse,
        'pval': model.pvalues,
        'pseudo_r2': 1 - model.deviance / model.null_deviance,
        'nobs': int(model.nobs),
        'model': model
    }


def nbreg_iv_control_function(df: pd.DataFrame, y_var: str, endog_var: str,
                               instrument: str, controls: list,
                               cluster_var: str) -> dict:
    """
    Control Function Approach for IV with Negative Binomial.

    This is the Python equivalent of Stata's qvf command.

    Steps:
    1. First stage: OLS of endogenous var on instrument + controls
    2. Get residuals (v_hat)
    3. Second stage: NB regression with endogenous var + residuals + controls

    The coefficient on the endogenous variable is the IV estimate.
    """
    all_vars = [y_var, endog_var, instrument] + controls + [cluster_var]
    data = df[all_vars].copy()

    for col in [y_var, endog_var, instrument] + controls:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna().reset_index(drop=True)

    # Step 1: First stage OLS
    first_stage_vars = [instrument] + controls
    X_first = sm.add_constant(data[first_stage_vars].astype(float))
    y_first = data[endog_var].astype(float)

    first_stage = sm.OLS(y_first, X_first).fit()

    # Get residuals
    data['v_hat'] = first_stage.resid

    # Step 2: Second stage NB with control function
    second_stage_vars = [endog_var, 'v_hat'] + controls
    X_second = sm.add_constant(data[second_stage_vars].astype(float))
    y_second = data[y_var].astype(float)

    # Fit NB model
    try:
        model = sm.GLM(y_second, X_second,
                       family=sm.families.NegativeBinomial(alpha=1.0)).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_var]}
        )
    except:
        # Fallback: try with different alpha
        model = sm.GLM(y_second, X_second,
                       family=sm.families.NegativeBinomial()).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_var]}
        )

    return {
        'coef': model.params,
        'se': model.bse,
        'pval': model.pvalues,
        'nobs': int(model.nobs),
        'model': model,
        'v_hat_coef': model.params.get('v_hat', np.nan),
        'v_hat_pval': model.pvalues.get('v_hat', np.nan)
    }


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def make_result(col: int, dep_var: str, model: str, var: str,
                result: dict, f_stat: float = np.nan) -> dict:
    """Create standardized result dictionary."""

    # Handle different result structures
    if isinstance(result['coef'], pd.Series):
        coef = result['coef'].get(var, np.nan)
        se = result['se'].get(var, np.nan)
        pval = result['pval'].get(var, np.nan) if 'pval' in result else np.nan
    else:
        coef = result['coef']
        se = result['se']
        pval = result.get('pval', np.nan)

    return {
        'column': col,
        'dep_var': dep_var,
        'model': model,
        'var': var,
        'coef': coef,
        'se': se,
        'pval': pval,
        'r2': result.get('r2', result.get('pseudo_r2', np.nan)),
        'nobs': result['nobs'],
        'f_stat': f_stat
    }


# =============================================================================
# PANEL A: ISRAELI ATTACKS AND PREDICTABLE EVENTS
# =============================================================================

def run_panel_a(df: pd.DataFrame) -> pd.DataFrame:
    """Panel A: Israeli Attacks and Predictable Newsworthy Events."""

    sample = filter_sample(df)
    sample = create_dummies(sample, Config.FE_FULL)
    fe_cols = get_fe_columns(sample, Config.FE_FULL)
    controls = Config.PRIOR_PAL + fe_cols

    results = []

    # Column 1: First stage (P_t+1)
    r1 = ols_clustered(sample, 'leaddaily_woi',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat = first_stage_f(sample, 'leaddaily_woi', 'lead_maj_events',
                           controls, 'monthyear')
    results.append(make_result(1, 'P_t+1', '1st stage', 'lead_maj_events', r1, f_stat))

    # Column 2: First stage (Uncorrected P_t+1)
    r2 = ols_clustered(sample, 'leaddaily_woi_nc',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat_nc = first_stage_f(sample, 'leaddaily_woi_nc', 'lead_maj_events',
                              controls, 'monthyear')
    results.append(make_result(2, 'Uncorr P_t+1', '1st stage', 'lead_maj_events', r2, f_stat_nc))

    # Column 3: 2SLS (Occurrence ~ P_t+1)
    r3 = iv_2sls(sample, 'occurrence', 'leaddaily_woi',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(3, 'Occurrence', '2SLS', 'leaddaily_woi', r3, f_stat))

    # Column 4: 2SLS (Occurrence ~ Uncorr P_t+1)
    r4 = iv_2sls(sample, 'occurrence', 'leaddaily_woi_nc',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(4, 'Occurrence', '2SLS', 'leaddaily_woi_nc', r4, f_stat_nc))

    # Column 5: Reduced form (Occurrence ~ Events)
    r5 = ols_clustered(sample, 'occurrence',
                       ['lead_maj_events'] + controls, 'monthyear')
    results.append(make_result(5, 'Occurrence', 'Reduced form', 'lead_maj_events', r5))

    # Column 6: NB IV using Control Function (Victims ~ P_t+1)
    r6 = nbreg_iv_control_function(sample, 'victims_isr', 'leaddaily_woi',
                                    'lead_maj_events', controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi', r6, f_stat))

    # Column 7: NB IV using Control Function (Victims ~ Uncorr P_t+1)
    r7 = nbreg_iv_control_function(sample, 'victims_isr', 'leaddaily_woi_nc',
                                    'lead_maj_events', controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi_nc', r7, f_stat_nc))

    # Column 8: NB Reduced form (Victims ~ Events)
    r8 = nbreg_glm(sample, 'victims_isr',
                   ['lead_maj_events'] + controls, 'monthyear')
    results.append(make_result(8, 'Num. Victims', 'NB Reduced', 'lead_maj_events', r8))

    return pd.DataFrame(results)


# =============================================================================
# PANEL B: PALESTINIAN ATTACKS AND PREDICTABLE EVENTS
# =============================================================================

def run_panel_b(df: pd.DataFrame) -> pd.DataFrame:
    """Panel B: Palestinian Attacks and Predictable Newsworthy Events."""

    sample = filter_sample(df)
    sample = create_dummies(sample, Config.FE_PARTIAL)  # No month FE!
    fe_cols = get_fe_columns(sample, Config.FE_PARTIAL)
    controls = Config.PRIOR_ISR + fe_cols

    results = []

    # Column 1: First stage (P_t+1)
    r1 = ols_clustered(sample, 'leaddaily_woi',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat = first_stage_f(sample, 'leaddaily_woi', 'lead_maj_events',
                           controls, 'monthyear')
    results.append(make_result(1, 'P_t+1', '1st stage', 'lead_maj_events', r1, f_stat))

    # Column 2: First stage (Uncorrected P_t+1)
    r2 = ols_clustered(sample, 'leaddaily_woi_nc',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat_nc = first_stage_f(sample, 'leaddaily_woi_nc', 'lead_maj_events',
                              controls, 'monthyear')
    results.append(make_result(2, 'Uncorr P_t+1', '1st stage', 'lead_maj_events', r2, f_stat_nc))

    # Column 3: 2SLS (Occurrence ~ P_t+1)
    r3 = iv_2sls(sample, 'occurrence_pal', 'leaddaily_woi',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(3, 'Occurrence', '2SLS', 'leaddaily_woi', r3, f_stat))

    # Column 4: 2SLS (Occurrence ~ Uncorr P_t+1)
    r4 = iv_2sls(sample, 'occurrence_pal', 'leaddaily_woi_nc',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(4, 'Occurrence', '2SLS', 'leaddaily_woi_nc', r4, f_stat_nc))

    # Column 5: Reduced form
    r5 = ols_clustered(sample, 'occurrence_pal',
                       ['lead_maj_events'] + controls, 'monthyear')
    results.append(make_result(5, 'Occurrence', 'Reduced form', 'lead_maj_events', r5))

    # Column 6: NB IV (Victims ~ P_t+1)
    r6 = nbreg_iv_control_function(sample, 'victims_pal', 'leaddaily_woi',
                                    'lead_maj_events', controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi', r6, f_stat))

    # Column 7: NB IV (Victims ~ Uncorr P_t+1)
    r7 = nbreg_iv_control_function(sample, 'victims_pal', 'leaddaily_woi_nc',
                                    'lead_maj_events', controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi_nc', r7, f_stat_nc))

    # Column 8: NB Reduced form
    r8 = nbreg_glm(sample, 'victims_pal',
                   ['lead_maj_events'] + controls, 'monthyear')
    results.append(make_result(8, 'Num. Victims', 'NB Reduced', 'lead_maj_events', r8))

    return pd.DataFrame(results)


# =============================================================================
# PANEL C: PLACEBO - ISRAELI ATTACKS AND UNPREDICTABLE EVENTS
# =============================================================================

def run_panel_c(df: pd.DataFrame) -> pd.DataFrame:
    """Panel C: Placebo - Israeli Attacks and Unpredictable Events."""

    sample = filter_sample(df)
    sample = create_dummies(sample, Config.FE_FULL)
    fe_cols = get_fe_columns(sample, Config.FE_FULL)
    controls = Config.PRIOR_PAL + fe_cols

    results = []

    # Column 1: First stage (P_t+1)
    r1 = ols_clustered(sample, 'leaddaily_woi',
                       ['lead_disaster'] + controls, 'monthyear')
    f_stat = first_stage_f(sample, 'leaddaily_woi', 'lead_disaster',
                           controls, 'monthyear')
    results.append(make_result(1, 'P_t+1', '1st stage', 'lead_disaster', r1, f_stat))

    # Column 2: First stage (Uncorrected P_t+1)
    r2 = ols_clustered(sample, 'leaddaily_woi_nc',
                       ['lead_disaster'] + controls, 'monthyear')
    f_stat_nc = first_stage_f(sample, 'leaddaily_woi_nc', 'lead_disaster',
                              controls, 'monthyear')
    results.append(make_result(2, 'Uncorr P_t+1', '1st stage', 'lead_disaster', r2, f_stat_nc))

    # Column 3: 2SLS (Occurrence ~ P_t+1)
    r3 = iv_2sls(sample, 'occurrence', 'leaddaily_woi',
                 'lead_disaster', controls, 'monthyear')
    results.append(make_result(3, 'Occurrence', '2SLS', 'leaddaily_woi', r3, f_stat))

    # Column 4: 2SLS (Occurrence ~ Uncorr P_t+1)
    r4 = iv_2sls(sample, 'occurrence', 'leaddaily_woi_nc',
                 'lead_disaster', controls, 'monthyear')
    results.append(make_result(4, 'Occurrence', '2SLS', 'leaddaily_woi_nc', r4, f_stat_nc))

    # Column 5: Reduced form
    r5 = ols_clustered(sample, 'occurrence',
                       ['lead_disaster'] + controls, 'monthyear')
    results.append(make_result(5, 'Occurrence', 'Reduced form', 'lead_disaster', r5))

    # Column 6: NB IV (Victims ~ P_t+1)
    r6 = nbreg_iv_control_function(sample, 'victims_isr', 'leaddaily_woi',
                                    'lead_disaster', controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi', r6, f_stat))

    # Column 7: NB IV (Victims ~ Uncorr P_t+1)
    r7 = nbreg_iv_control_function(sample, 'victims_isr', 'leaddaily_woi_nc',
                                    'lead_disaster', controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB IV (CF)', 'leaddaily_woi_nc', r7, f_stat_nc))

    # Column 8: NB Reduced form
    r8 = nbreg_glm(sample, 'victims_isr',
                   ['lead_disaster'] + controls, 'monthyear')
    results.append(make_result(8, 'Num. Victims', 'NB Reduced', 'lead_disaster', r8))

    return pd.DataFrame(results)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def add_stars(coef: float, pval: float) -> str:
    """Add significance stars to coefficient."""
    if pd.isna(coef):
        return "—"
    if pd.isna(pval):
        return f"{coef:.3f}"
    stars = ''
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.10:
        stars = '*'
    return f"{coef:.3f}{stars}"


def format_se(se: float) -> str:
    """Format standard error in parentheses."""
    if pd.isna(se):
        return ""
    return f"({se:.3f})"


def print_table_header():
    """Print table title."""
    width = 120
    print("\n" + "═" * width)
    print("TABLE 5: ATTACKS AND NEXT-DAY NEWS PRESSURE DRIVEN BY")
    print("PREDICTABLE POLITICAL AND SPORTS EVENTS")
    print("═" * width)


def print_panel_results(results_df: pd.DataFrame, panel_name: str,
                        main_var_label: str, fe_label: str, prior_label: str):
    """Print panel in publication format."""

    width = 120
    col_width = 12

    # Panel header
    print("\n" + "─" * width)
    print(f"  {panel_name}")
    print("─" * width)

    # Column numbers
    row = f"{'':32}"
    for i in range(1, 9):
        row += f"({i}){'':{col_width-3}}"
    print(row)

    # Dependent variables
    dep_vars = ['P t+1', 'Uncorr P', 'Occurr.', 'Occurr.',
                'Occurr.', 'Victims', 'Victims', 'Victims']
    row = f"{'Dependent variable:':32}"
    for dv in dep_vars:
        row += f"{dv:>{col_width}}"
    print(row)

    # Models
    models = ['2SLS', '2SLS', '2SLS', '2SLS', 'OLS', 'NB IV', 'NB IV', 'NB']
    row = f"{'Model:':32}"
    for m in models:
        row += f"{m:>{col_width}}"
    print(row)

    # Model types
    types = ['1st stg', '1st stg', '2nd stg', '2nd stg', 'Red.form', '2nd stg', '2nd stg', 'Red.form']
    row = f"{'':32}"
    for t in types:
        row += f"{t:>{col_width}}"
    print(row)

    print("─" * width)

    # Extract data
    data = {}
    for _, r in results_df.iterrows():
        data[r['column']] = r

    # Main variable (instrument) - columns 1, 2, 5, 8
    coefs = [data[1]['coef'], data[2]['coef'], np.nan, np.nan,
             data[5]['coef'], np.nan, np.nan, data[8]['coef']]
    ses = [data[1]['se'], data[2]['se'], np.nan, np.nan,
           data[5]['se'], np.nan, np.nan, data[8]['se']]
    pvals = [data[1]['pval'], data[2]['pval'], np.nan, np.nan,
             data[5]['pval'], np.nan, np.nan, data[8]['pval']]

    row = f"{main_var_label:32}"
    for c, p in zip(coefs, pvals):
        row += f"{add_stars(c, p):>{col_width}}"
    print(row)

    row = f"{'':32}"
    for s in ses:
        row += f"{format_se(s):>{col_width}}"
    print(row)

    # News pressure - columns 3, 6
    print()
    coefs = [np.nan, np.nan, data[3]['coef'], np.nan, np.nan, data[6]['coef'], np.nan, np.nan]
    ses = [np.nan, np.nan, data[3]['se'], np.nan, np.nan, data[6]['se'], np.nan, np.nan]
    pvals = [np.nan, np.nan, data[3]['pval'], np.nan, np.nan, data[6]['pval'], np.nan, np.nan]

    row = f"{'News pressure t+1':32}"
    for c, p in zip(coefs, pvals):
        row += f"{add_stars(c, p):>{col_width}}"
    print(row)

    row = f"{'':32}"
    for s in ses:
        row += f"{format_se(s):>{col_width}}"
    print(row)

    # Uncorrected news pressure - columns 4, 7
    print()
    coefs = [np.nan, np.nan, np.nan, data[4]['coef'], np.nan, np.nan, data[7]['coef'], np.nan]
    ses = [np.nan, np.nan, np.nan, data[4]['se'], np.nan, np.nan, data[7]['se'], np.nan]
    pvals = [np.nan, np.nan, np.nan, data[4]['pval'], np.nan, np.nan, data[7]['pval'], np.nan]

    row = f"{'Uncorrected NP t+1':32}"
    for c, p in zip(coefs, pvals):
        row += f"{add_stars(c, p):>{col_width}}"
    print(row)

    row = f"{'':32}"
    for s in ses:
        row += f"{format_se(s):>{col_width}}"
    print(row)

    # Fixed effects and controls
    print()
    row = f"{fe_label:32}" + f"{'Yes':>{col_width}}" * 8
    print(row)
    row = f"{prior_label:32}" + f"{'Yes':>{col_width}}" * 8
    print(row)

    # Observations
    print()
    row = f"{'Observations':32}"
    for i in range(1, 9):
        row += f"{data[i]['nobs']:>{col_width},}"
    print(row)

    # R-squared
    row = f"{'R-squared':32}"
    for i in range(1, 9):
        r2 = data[i]['r2']
        if pd.isna(r2) or i in [6, 7]:  # IV NB doesn't have R2
            row += f"{'—':>{col_width}}"
        else:
            row += f"{r2:>{col_width}.3f}"
    print(row)

    # F-statistic
    row = f"{'F excl. instr.':32}"
    for i in range(1, 9):
        f = data[i]['f_stat']
        if pd.isna(f) or i == 5 or i == 8:
            row += f"{'—':>{col_width}}"
        else:
            row += f"{f:>{col_width}.2f}"
    print(row)


def print_full_table(panel_a: pd.DataFrame, panel_b: pd.DataFrame,
                     panel_c: pd.DataFrame):
    """Print complete Table 5."""

    print_table_header()

    print_panel_results(
        panel_a,
        "PANEL A: ISRAELI ATTACKS AND PREDICTABLE NEWSWORTHY EVENTS",
        "Political/sports events t+1",
        "FEs (year, month, DOW)",
        "Prior Palestinian attacks"
    )

    print_panel_results(
        panel_b,
        "PANEL B: PALESTINIAN ATTACKS AND PREDICTABLE NEWSWORTHY EVENTS",
        "Political/sports events t+1",
        "FEs (year, DOW)",
        "Prior Israeli attacks"
    )

    print_panel_results(
        panel_c,
        "PANEL C: PLACEBO - ISRAELI ATTACKS AND UNPREDICTABLE EVENTS",
        "Disaster onset t+1",
        "FEs (year, month, DOW)",
        "Prior Palestinian attacks"
    )

    print("\n" + "═" * 120)
    print("Notes: Robust standard errors clustered by month×year in parentheses.")
    print("NB IV estimated using Control Function Approach (equivalent to Stata's qvf).")
    print("*** p<0.01, ** p<0.05, * p<0.1")
    print("═" * 120)


def format_table_publication(panel_a: pd.DataFrame,
                             panel_b: pd.DataFrame,
                             panel_c: pd.DataFrame) -> pd.DataFrame:
    """Format results for Excel export."""

    rows = []
    for panel_df, panel_name in [(panel_a, 'A'), (panel_b, 'B'), (panel_c, 'C')]:
        row = {'Panel': panel_name}
        for _, r in panel_df.iterrows():
            col = r['column']
            row[f'({col}) Coef'] = r['coef']
            row[f'({col}) SE'] = r['se']
            row[f'({col}) pval'] = r['pval']
        row['Observations'] = panel_df.iloc[0]['nobs']
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(data_path: str = None):
    """Main execution function."""

    path = data_path or Config.DATA_PATH

    # Load data
    print(f"\nLoading data from: {path}")
    df = load_data(path)
    print(f"Total observations: {len(df):,}")
    print(f"Sample (gaza_war==0): {len(df[df['gaza_war']==0]):,}")

    # Run panels
    print("\nRunning regressions...")

    print("  → Panel A: Israeli Attacks...")
    panel_a = run_panel_a(df)

    print("  → Panel B: Palestinian Attacks...")
    panel_b = run_panel_b(df)

    print("  → Panel C: Placebo (Disasters)...")
    panel_c = run_panel_c(df)

    # Print formatted table
    print_full_table(panel_a, panel_b, panel_c)

    # Export to Excel
    with pd.ExcelWriter(Config.OUTPUT_PATH, engine='openpyxl') as writer:
        panel_a.to_excel(writer, sheet_name='Panel_A', index=False)
        panel_b.to_excel(writer, sheet_name='Panel_B', index=False)
        panel_c.to_excel(writer, sheet_name='Panel_C', index=False)

        combined = format_table_publication(panel_a, panel_b, panel_c)
        combined.to_excel(writer, sheet_name='Combined', index=False)

    print(f"\nResults exported to: {Config.OUTPUT_PATH}")

    return panel_a, panel_b, panel_c


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    data_path = "/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta"
    main(data_path)