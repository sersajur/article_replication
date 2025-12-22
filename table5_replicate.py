"""
Replication of Table 5: Attacks and Next-Day News Pressure
Driven by Predictable Political and Sports Events

Source: Durante & Zhuravskaya (JPE)
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
    FE_PARTIAL = ['year', 'dow']            # Panel B

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
        'gaza_war', 'month', 'year', 'dow'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
    """Create dummy variables for fixed effects."""
    result = df.copy()
    for col in cols:
        # Convert to numeric codes if categorical
        if result[col].dtype == 'object' or result[col].dtype.name == 'category':
            result[col] = pd.Categorical(result[col]).codes

        # Create dummies with numeric dtype
        dummies = pd.get_dummies(result[col], prefix=col, drop_first=True, dtype=float)
        result = pd.concat([result, dummies], axis=1)
    return result


def get_fe_columns(df: pd.DataFrame, fe_vars: list) -> list:
    """Get list of fixed effect dummy column names."""
    fe_cols = []
    for var in fe_vars:
        fe_cols += [c for c in df.columns if c.startswith(f'{var}_')]
    return fe_cols


def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Ensure all columns are numeric."""
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
    return result


# =============================================================================
# REGRESSION MODELS
# =============================================================================

def ols_clustered(df: pd.DataFrame, y_var: str, x_vars: list,
                  cluster_var: str) -> dict:
    """OLS with clustered standard errors."""
    all_vars = [y_var] + x_vars + [cluster_var]
    data = df[all_vars].copy()

    # Ensure numeric types
    for col in [y_var] + x_vars:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop missing values
    data = data.dropna()

    y = data[y_var].astype(float)
    X = sm.add_constant(data[x_vars].astype(float))

    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': data[cluster_var]}
    )

    return {
        'coef': model.params,
        'se': model.bse,
        'pval': model.pvalues,
        'r2': model.rsquared,
        'nobs': int(model.nobs),
        'model': model
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

    # F = (coef / se)^2
    t_stat = result['coef'][instrument] / result['se'][instrument]
    return t_stat ** 2


def nbreg_glm(df: pd.DataFrame, y_var: str, x_vars: list,
              cluster_var: str) -> dict:
    """Negative binomial regression (GLM approximation)."""
    all_vars = [y_var] + x_vars + [cluster_var]
    data = df[all_vars].copy()

    # Ensure numeric types
    for col in [y_var] + x_vars:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    y = data[y_var].astype(float)
    X = sm.add_constant(data[x_vars].astype(float))

    # Negative binomial GLM
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(
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


# =============================================================================
# PANEL A: ISRAELI ATTACKS AND PREDICTABLE EVENTS
# =============================================================================

def make_result(col: int, dep_var: str, model: str, var: str,
                result: dict, f_stat: float = np.nan) -> dict:
    """Create standardized result dictionary."""
    return {
        'column': col,
        'dep_var': dep_var,
        'model': model,
        'var': var,
        'coef': result['coef'].get(var, np.nan),
        'se': result['se'].get(var, np.nan),
        'pval': result['pval'].get(var, np.nan) if 'pval' in result else np.nan,
        'r2': result.get('r2', result.get('pseudo_r2', np.nan)),
        'nobs': result['nobs'],
        'f_stat': f_stat
    }


def run_panel_a(df: pd.DataFrame) -> pd.DataFrame:
    """Panel A: Israeli Attacks and Predictable Newsworthy Events."""

    sample = filter_sample(df)
    sample = create_dummies(sample, Config.FE_FULL)
    fe_cols = get_fe_columns(sample, Config.FE_FULL)
    controls = Config.PRIOR_PAL + fe_cols

    results = []

    # -------------------------------------------------------------------------
    # Column 1: First stage (P_t+1)
    # -------------------------------------------------------------------------
    r1 = ols_clustered(sample, 'leaddaily_woi',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat = first_stage_f(sample, 'leaddaily_woi', 'lead_maj_events',
                           controls, 'monthyear')
    results.append(make_result(1, 'P_t+1', '1st stage', 'lead_maj_events', r1, f_stat))

    # -------------------------------------------------------------------------
    # Column 2: First stage (Uncorrected P_t+1)
    # -------------------------------------------------------------------------
    r2 = ols_clustered(sample, 'leaddaily_woi_nc',
                       ['lead_maj_events'] + controls, 'monthyear')
    f_stat_nc = first_stage_f(sample, 'leaddaily_woi_nc', 'lead_maj_events',
                              controls, 'monthyear')
    results.append(make_result(2, 'Uncorr P_t+1', '1st stage', 'lead_maj_events', r2, f_stat_nc))

    # -------------------------------------------------------------------------
    # Column 3: 2SLS (Occurrence ~ P_t+1)
    # -------------------------------------------------------------------------
    r3 = iv_2sls(sample, 'occurrence', 'leaddaily_woi',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(3, 'Occurrence', '2SLS', 'leaddaily_woi', r3, f_stat))

    # -------------------------------------------------------------------------
    # Column 4: 2SLS (Occurrence ~ Uncorr P_t+1)
    # -------------------------------------------------------------------------
    r4 = iv_2sls(sample, 'occurrence', 'leaddaily_woi_nc',
                 'lead_maj_events', controls, 'monthyear')
    results.append(make_result(4, 'Occurrence', '2SLS', 'leaddaily_woi_nc', r4, f_stat_nc))

    # -------------------------------------------------------------------------
    # Column 5: Reduced form (Occurrence ~ Events)
    # -------------------------------------------------------------------------
    r5 = ols_clustered(sample, 'occurrence',
                       ['lead_maj_events'] + controls, 'monthyear')
    results.append(make_result(5, 'Occurrence', 'Reduced form', 'lead_maj_events', r5))

    # -------------------------------------------------------------------------
    # Column 6: NB IV (Victims ~ P_t+1)
    # -------------------------------------------------------------------------
    r6 = nbreg_glm(sample, 'victims_isr',
                   ['leaddaily_woi'] + controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB GLM', 'leaddaily_woi', r6, f_stat))

    # -------------------------------------------------------------------------
    # Column 7: NB IV (Victims ~ Uncorr P_t+1)
    # -------------------------------------------------------------------------
    r7 = nbreg_glm(sample, 'victims_isr',
                   ['leaddaily_woi_nc'] + controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB GLM', 'leaddaily_woi_nc', r7, f_stat_nc))

    # -------------------------------------------------------------------------
    # Column 8: NB Reduced form (Victims ~ Events)
    # -------------------------------------------------------------------------
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
    sample = create_dummies(sample, Config.FE_PARTIAL)
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

    # Column 6: NB (Victims ~ P_t+1)
    r6 = nbreg_glm(sample, 'victims_pal',
                   ['leaddaily_woi'] + controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB GLM', 'leaddaily_woi', r6, f_stat))

    # Column 7: NB (Victims ~ Uncorr P_t+1)
    r7 = nbreg_glm(sample, 'victims_pal',
                   ['leaddaily_woi_nc'] + controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB GLM', 'leaddaily_woi_nc', r7, f_stat_nc))

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

    # Column 6: NB (Victims ~ P_t+1)
    r6 = nbreg_glm(sample, 'victims_isr',
                   ['leaddaily_woi'] + controls, 'monthyear')
    results.append(make_result(6, 'Num. Victims', 'NB GLM', 'leaddaily_woi', r6, f_stat))

    # Column 7: NB (Victims ~ Uncorr P_t+1)
    r7 = nbreg_glm(sample, 'victims_isr',
                   ['leaddaily_woi_nc'] + controls, 'monthyear')
    results.append(make_result(7, 'Num. Victims', 'NB GLM', 'leaddaily_woi_nc', r7, f_stat_nc))

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


def format_r2(r2: float) -> str:
    """Format R-squared."""
    if pd.isna(r2):
        return "—"
    return f"{r2:.3f}"


def format_f(f: float) -> str:
    """Format F-statistic."""
    if pd.isna(f):
        return "—"
    return f"{f:.2f}"


def print_table_header():
    """Print table title."""
    width = 120
    print("\n" + "═" * width)
    print("TABLE 5: ATTACKS AND NEXT-DAY NEWS PRESSURE DRIVEN BY")
    print("PREDICTABLE POLITICAL AND SPORTS EVENTS")
    print("═" * width)


def print_panel_header(panel_name: str, dep_vars: list, models: list):
    """Print panel header with column labels."""
    width = 120
    col_width = 12

    print("\n" + "─" * width)
    print(f"  {panel_name}")
    print("─" * width)

    # Column numbers
    header1 = f"{'':30}"
    for i in range(1, 9):
        header1 += f"({i}){'':{col_width-3}}"
    print(header1)

    # Dependent variables
    header2 = f"{'Dependent variable:':30}"
    for dv in dep_vars:
        header2 += f"{dv:>{col_width}}"
    print(header2)

    # Models
    header3 = f"{'Model:':30}"
    for m in models:
        header3 += f"{m:>{col_width}}"
    print(header3)

    print("─" * width)


def print_coef_row(label: str, coefs: list, ses: list, pvals: list):
    """Print coefficient row with standard errors below."""
    col_width = 12

    # Coefficients with stars
    row1 = f"{label:30}"
    for coef, pval in zip(coefs, pvals):
        row1 += f"{add_stars(coef, pval):>{col_width}}"
    print(row1)

    # Standard errors
    row2 = f"{'':30}"
    for se in ses:
        row2 += f"{format_se(se):>{col_width}}"
    print(row2)


def print_info_row(label: str, values: list):
    """Print information row (FE, Obs, R2, F-stat)."""
    col_width = 12
    row = f"{label:30}"
    for v in values:
        if isinstance(v, bool):
            row += f"{'Yes' if v else 'No':>{col_width}}"
        elif isinstance(v, int):
            row += f"{v:>{col_width},}"
        elif isinstance(v, float):
            if pd.isna(v):
                row += f"{'—':>{col_width}}"
            else:
                row += f"{v:>{col_width}.3f}"
        else:
            row += f"{str(v):>{col_width}}"
    print(row)


def format_panel_table(results_df: pd.DataFrame, panel_name: str,
                       main_var: str, fe_label: str, prior_label: str):
    """Format and print a single panel as publication-style table."""

    # Define column structure
    dep_vars = ['P t+1', 'Uncorr P t+1', 'Occurrence', 'Occurrence',
                'Occurrence', 'Num.Victims', 'Num.Victims', 'Num.Victims']
    models = ['2SLS', '2SLS', '2SLS', '2SLS',
              'OLS', 'ML NB', 'ML NB', 'ML NB']
    model_types = ['1st stage', '1st stage', '2nd stage', '2nd stage',
                   'Red. form', 'IV 2nd', 'IV 2nd', 'Red. form']

    print_panel_header(panel_name, dep_vars, models)

    # Model type row
    col_width = 12
    row = f"{'':30}"
    for mt in model_types:
        row += f"{mt:>{col_width}}"
    print(row)
    print()

    # Extract data by column
    coefs = []
    ses = []
    pvals = []
    nobs = []
    r2s = []
    f_stats = []

    for col in range(1, 9):
        row_data = results_df[results_df['column'] == col].iloc[0]
        coefs.append(row_data['coef'])
        ses.append(row_data['se'])
        pvals.append(row_data.get('pval', np.nan))
        nobs.append(row_data['nobs'])
        r2s.append(row_data['r2'])
        f_stats.append(row_data['f_stat'])

    # Main variable row (instrument or endogenous)
    # Columns 1,2,5,8 show instrument; 3,4,6,7 show endogenous
    instr_coefs = [coefs[0], coefs[1], np.nan, np.nan, coefs[4], np.nan, np.nan, coefs[7]]
    instr_ses = [ses[0], ses[1], np.nan, np.nan, ses[4], np.nan, np.nan, ses[7]]
    instr_pvals = [pvals[0], pvals[1], np.nan, np.nan, pvals[4], np.nan, np.nan, pvals[7]]

    endog_coefs = [np.nan, np.nan, coefs[2], coefs[3], np.nan, coefs[5], coefs[6], np.nan]
    endog_ses = [np.nan, np.nan, ses[2], ses[3], np.nan, ses[5], ses[6], np.nan]
    endog_pvals = [np.nan, np.nan, pvals[2], pvals[3], np.nan, pvals[5], pvals[6], np.nan]

    # Print instrument row
    print_coef_row(main_var, instr_coefs, instr_ses, instr_pvals)
    print()

    # Print news pressure row
    print_coef_row("News pressure t+1",
                   [np.nan, np.nan, coefs[2], np.nan, np.nan, coefs[5], np.nan, np.nan],
                   [np.nan, np.nan, ses[2], np.nan, np.nan, ses[5], np.nan, np.nan],
                   [np.nan, np.nan, pvals[2], np.nan, np.nan, pvals[5], np.nan, np.nan])
    print()

    # Print uncorrected news pressure row
    print_coef_row("Uncorrected NP t+1",
                   [np.nan, np.nan, np.nan, coefs[3], np.nan, np.nan, coefs[6], np.nan],
                   [np.nan, np.nan, np.nan, ses[3], np.nan, np.nan, ses[6], np.nan],
                   [np.nan, np.nan, np.nan, pvals[3], np.nan, np.nan, pvals[6], np.nan])

    print()
    print_info_row(fe_label, [True] * 8)
    print_info_row(prior_label, [True] * 8)
    print()
    print_info_row("Observations", nobs)

    # R-squared row
    r2_display = []
    for i, r in enumerate(r2s):
        if i in [5, 6]:  # IV NB columns don't have R2
            r2_display.append("—")
        elif pd.isna(r):
            r2_display.append("—")
        else:
            r2_display.append(f"{r:.3f}")

    row = f"{'(Pseudo) R-squared':30}"
    for v in r2_display:
        row += f"{v:>{12}}"
    print(row)

    # F-statistic row
    f_display = []
    for f in f_stats:
        if pd.isna(f):
            f_display.append("—")
        else:
            f_display.append(f"{f:.2f}")

    row = f"{'F-stat, excl. instr.':30}"
    for v in f_display:
        row += f"{v:>{12}}"
    print(row)


def print_full_table(panel_a: pd.DataFrame, panel_b: pd.DataFrame,
                     panel_c: pd.DataFrame):
    """Print complete Table 5 in publication format."""

    print_table_header()

    # Panel A
    format_panel_table(
        panel_a,
        "PANEL A: ISRAELI ATTACKS AND PREDICTABLE NEWSWORTHY EVENTS",
        "Political/sports events t+1",
        "FEs (year, month, DOW)",
        "Prior Palestinian attacks"
    )

    # Panel B
    format_panel_table(
        panel_b,
        "PANEL B: PALESTINIAN ATTACKS AND PREDICTABLE NEWSWORTHY EVENTS",
        "Political/sports events t+1",
        "FEs (year, DOW)",
        "Prior Israeli attacks"
    )

    # Panel C
    format_panel_table(
        panel_c,
        "PANEL C: PLACEBO - ISRAELI ATTACKS AND UNPREDICTABLE NEWSWORTHY EVENTS",
        "Disaster onset t+1",
        "FEs (year, month, DOW)",
        "Prior Palestinian attacks"
    )

    print("\n" + "═" * 120)
    print("Notes: Robust standard errors clustered by month×year in parentheses.")
    print("*** p<0.01, ** p<0.05, * p<0.1")
    print("═" * 120)


def format_table_publication(panel_a: pd.DataFrame,
                             panel_b: pd.DataFrame,
                             panel_c: pd.DataFrame) -> pd.DataFrame:
    """Format results in publication-style DataFrame for Excel export."""

    rows = []

    for panel_df, panel_name in [(panel_a, 'A'), (panel_b, 'B'), (panel_c, 'C')]:
        row = {'Panel': panel_name}
        for _, r in panel_df.iterrows():
            col = r['column']
            row[f'({col}) Coef'] = r['coef']
            row[f'({col}) SE'] = r['se']

        # Add summary stats from first row (all same within panel)
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
    print(f"Observations: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sample after gaza_war==0 filter: {len(df[df['gaza_war']==0]):,}")

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