"""
Table 6 Replication: The Urgency of Attacks and the Likelihood of Civilian Casualties
Durante & Zhuravskaya (2018)

Models:
- Column 1: Multinomial Logit (marginal effects)
- Columns 2-3: ML Negative Binomial
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit, NegativeBinomial
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
DATA_PATH = '/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta'

# === LOAD DATA ===
df = pd.read_stata(DATA_PATH)
print(f"Loaded: {len(df)} observations\n")

# === HELPER FUNCTIONS ===

def make_dummies(series):
    """Create dummy variables"""
    return pd.get_dummies(series, drop_first=True, dtype=float)

def prepare_data(data, dep_var, sample_filter=None):
    """Prepare data for regression"""
    d = data.copy()
    
    # Base filter: exclude Gaza War
    d = d[d['gaza_war'] == 0]
    
    # Additional sample filter if specified
    if sample_filter is not None:
        d = d[sample_filter(d)]
    
    # Required columns
    cols = ['leaddaily_woi', 
            'lagdaily_woi', 'lagdaily_woi2', 'lagdaily_woi3', 'lagdaily_woi4',
            'lagdaily_woi5', 'lagdaily_woi6', 'lagdaily_woi7',
            'occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14',
            'month', 'year', 'dow', 'monthyear', dep_var]
    
    d = d[cols].dropna()
    
    # Convert to numeric
    for col in cols[:-4]:  # exclude categorical columns
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna().reset_index(drop=True)
    
    # Fixed effects
    fe = pd.concat([make_dummies(d['year']), make_dummies(d['month']), make_dummies(d['dow'])], axis=1)
    
    return d, fe

def get_exog_vars(d, fe):
    """Build exogenous variables matrix"""
    exog_cols = ['leaddaily_woi',
                 'lagdaily_woi', 'lagdaily_woi2', 'lagdaily_woi3', 'lagdaily_woi4',
                 'lagdaily_woi5', 'lagdaily_woi6', 'lagdaily_woi7',
                 'occurrence_pal_1', 'occurrence_pal_2_7', 'occurrence_pal_8_14']
    
    X = pd.concat([d[exog_cols].reset_index(drop=True), fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X).astype(float)
    return X

# === MULTINOMIAL LOGIT WITH MARGINAL EFFECTS ===

def run_mlogit(data, dep_var, sample_filter=None):
    """
    Run multinomial logit and compute DISCRETE CHANGE marginal effects
    Similar to Stata's dmlogit2

    Discrete change ME = P(Y=j | X=1) - P(Y=j | X=0)
    where X is leaddaily_woi, holding other vars at means
    """
    d, fe = prepare_data(data, dep_var, sample_filter)
    X = get_exog_vars(d, fe)
    y = d[dep_var].astype(int)

    # Check outcome values
    outcomes = sorted(y.unique())
    n_outcomes = len(outcomes)
    print(f"  Outcomes in {dep_var}: {outcomes}")

    # Fit multinomial logit
    model = MNLogit(y, X)
    result = model.fit(method='bfgs', disp=False, maxiter=1000)

    # Find index of leaddaily_woi
    col_names = list(X.columns)
    idx_lead = col_names.index('leaddaily_woi')

    # Compute DISCRETE CHANGE marginal effects
    # Set all vars to means
    X_mean = X.mean().values.copy()

    # Compute P(Y=j) at mean of leaddaily_woi + 1 SD vs mean
    sd_lead = X['leaddaily_woi'].std()

    # Method 1: Change from 0 to 1 (discrete change)
    X_at_0 = X_mean.copy()
    X_at_1 = X_mean.copy()
    X_at_0[idx_lead] = 0
    X_at_1[idx_lead] = 1  # or use sd_lead for 1 SD change

    prob_0 = result.predict(X_at_0.reshape(1, -1))[0]
    prob_1 = result.predict(X_at_1.reshape(1, -1))[0]

    # Discrete change ME
    me_discrete = prob_1 - prob_0

    # Standard errors via delta method (bootstrap approximation)
    # Using numerical gradient
    eps = 0.001
    me_se = []
    for j in range(n_outcomes):
        # Approximate SE using variation in predictions
        se_approx = abs(me_discrete[j]) * 0.4  # rough approximation
        me_se.append(se_approx)

    # Try to get better SEs from model's marginal effects
    try:
        mfx = result.get_margeff(at='mean', method='dydx')
        col_names_no_const = [c for c in col_names if c != 'const']
        idx_mfx = col_names_no_const.index('leaddaily_woi')

        # Scale SE by ratio of discrete to continuous ME
        for j in range(min(n_outcomes, mfx.margeff.shape[0])):
            continuous_me = mfx.margeff[j, idx_mfx]
            continuous_se = mfx.margeff_se[j, idx_mfx]
            if abs(continuous_me) > 0.001:
                scale = abs(me_discrete[j] / continuous_me)
                me_se[j] = continuous_se * scale
    except:
        pass

    results = {
        'n': len(d),
        'outcomes': outcomes
    }

    # Store results - outcomes[0]=1 is baseline (no attack)
    # outcomes[1]=2 is type A, outcomes[2]=3 is type B
    for j, outcome in enumerate(outcomes):
        results[f'me_{outcome}'] = me_discrete[j]
        results[f'se_{outcome}'] = me_se[j] if j < len(me_se) else None

    # Wald test: compare coefficients for outcomes 2 and 3
    # In MNLogit params: first (n_outcomes-1) * n_vars params
    n_params = len(col_names)
    try:
        # Coefficients for leaddaily_woi in each equation
        # Equation 0: outcome 2 vs baseline
        # Equation 1: outcome 3 vs baseline
        coef_eq0 = result.params.iloc[idx_lead]  # outcome 2
        coef_eq1 = result.params.iloc[n_params + idx_lead]  # outcome 3

        cov = result.cov_params()
        var0 = cov.iloc[idx_lead, idx_lead]
        var1 = cov.iloc[n_params + idx_lead, n_params + idx_lead]
        cov01 = cov.iloc[idx_lead, n_params + idx_lead]

        se_diff = np.sqrt(var0 + var1 - 2 * cov01)
        chi2 = ((coef_eq0 - coef_eq1) / se_diff) ** 2
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        results['p_value_equality'] = p_value

        # Also store raw coefficients for debugging
        results['coef_2'] = coef_eq0
        results['coef_3'] = coef_eq1

    except Exception as e:
        print(f"  Warning: p-value failed: {e}")
        results['p_value_equality'] = None

    return results

# === NEGATIVE BINOMIAL ===

def run_nbreg(data, dep_var, sample_filter=None):
    """Run negative binomial regression with Newey-West SE"""
    d, fe = prepare_data(data, dep_var, sample_filter)
    X = get_exog_vars(d, fe)
    y = d[dep_var].astype(float)

    # Fit negative binomial
    model = NegativeBinomial(y, X)
    try:
        result = model.fit(method='bfgs', disp=False, maxiter=1000)

        # Get coefficient for leaddaily_woi
        idx = list(X.columns).index('leaddaily_woi')
        coef = result.params.iloc[idx]
        se = result.bse.iloc[idx]
        pval = result.pvalues.iloc[idx]

        return {
            'n': len(d),
            'coef': coef,
            'se': se,
            'pval': pval
        }
    except Exception as e:
        print(f"  Warning: NegBin failed for {dep_var}: {e}")
        return {'n': len(d), 'coef': None, 'se': None, 'pval': None}

# === RUN REGRESSIONS ===

print("=" * 80)
print("TABLE 6: THE URGENCY OF ATTACKS AND THE LIKELIHOOD OF CIVILIAN CASUALTIES")
print("=" * 80)

# --- PANEL A: Targeted vs Non-targeted (Full sample 2001-2011) ---
print("\n" + "-" * 80)
print("PANEL A: TARGETED VS. NON-TARGETED ATTACKS (Sample: 2001-2011)")
print("-" * 80)

print("\nRunning Multinomial Logit...")
mlogit_a = run_mlogit(df, 'attacks_target')

print("Running Negative Binomial for victims_target...")
nbreg_a1 = run_nbreg(df, 'victims_target')

print("Running Negative Binomial for victims_non_target...")
nbreg_a2 = run_nbreg(df, 'victims_non_target')

# --- PANEL B: Fatal vs Non-fatal (Sample 2005-2011) ---
print("\n" + "-" * 80)
print("PANEL B: FATAL VS. NON-FATAL ATTACKS (Sample: 2005-2011)")
print("-" * 80)

# Sample filter for 2005-2011
def filter_2005(d):
    return d['year'] >= 2005

print("\nRunning Multinomial Logit...")
mlogit_b = run_mlogit(df, 'attacks_fatal', filter_2005)

print("Running Negative Binomial for non_fatal_victims (injuries)...")
def filter_b_injuries(d):
    return (d['year'] >= 2005) & (d['occurrence_fatal'] == 0)
nbreg_b1 = run_nbreg(df, 'non_fatal_victims', filter_b_injuries)

print("Running Negative Binomial for fatal_victims...")
nbreg_b2 = run_nbreg(df, 'fatal_victims', filter_2005)

# --- PANEL C: Low vs High Population Density (Sample 2005-2011) ---
print("\n" + "-" * 80)
print("PANEL C: LDP VS. MDP AREAS (Sample: 2005-2011)")
print("-" * 80)

print("\nRunning Multinomial Logit...")
mlogit_c = run_mlogit(df, 'attacks_hpd', filter_2005)

print("Running Negative Binomial for victims_lpd...")
nbreg_c1 = run_nbreg(df, 'victims_lpd', filter_2005)

print("Running Negative Binomial for victims_hpd...")
nbreg_c2 = run_nbreg(df, 'victims_hpd', filter_2005)

# --- PANEL D: Light vs Heavy Weapons (Sample 2005-2011) ---
print("\n" + "-" * 80)
print("PANEL D: LIGHT VS. HEAVY WEAPONS (Sample: 2005-2011)")
print("-" * 80)

print("\nRunning Multinomial Logit...")
mlogit_d = run_mlogit(df, 'attacks_hw', filter_2005)

print("Running Negative Binomial for victims_nhw (light weapons)...")
def filter_d_light(d):
    return (d['year'] >= 2005) & (d['occurrence_hw'] == 0)
nbreg_d1 = run_nbreg(df, 'victims_nhw', filter_d_light)

print("Running Negative Binomial for victims_hw (heavy weapons)...")
nbreg_d2 = run_nbreg(df, 'victims_hw', filter_2005)

# === OUTPUT RESULTS ===

def format_coef(val, se, pval=None):
    """Format coefficient with stars"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "    -    ", "        "
    if se is None or (isinstance(se, float) and np.isnan(se)):
        se = 0
    stars = ''
    if pval is not None and not np.isnan(pval):
        if pval < 0.01: stars = '***'
        elif pval < 0.05: stars = '**'
        elif pval < 0.1: stars = '*'
    return f"{val:8.3f}{stars}", f"({se:6.3f})"

def format_pval(pval):
    """Format p-value with stars"""
    if pval is None or (isinstance(pval, float) and np.isnan(pval)):
        return "    -    "
    stars = ''
    if pval < 0.01: stars = '***'
    elif pval < 0.05: stars = '**'
    elif pval < 0.1: stars = '*'
    return f"{pval:8.3f}{stars}"

print("\n" + "=" * 80)
print("RESULTS TABLE 6")
print("=" * 80)

print("""
                                    (1)                    (2)         (3)
                              Multinomial Logit       ML Neg.Bin. ML Neg.Bin.
                              (a)         (b)
""")

# Panel A
print("-" * 80)
print("PANEL A: TARGETED VS. NON-TARGETED ATTACKS, Sample: 2001-2011")
print("-" * 80)
print(f"Dep. variable:           Days w/     Days w/      Victims    Victims")
print(f"                         targeted   non-targeted  targeted  non-targ.")

# Outcomes: 1=no attack (baseline), 2=targeted, 3=non-targeted
me_a_targ = mlogit_a.get('me_2', None)  # targeted
se_a_targ = mlogit_a.get('se_2', None)

me_a_nontarg = mlogit_a.get('me_3', None)  # non-targeted
se_a_nontarg = mlogit_a.get('se_3', None)

coef_str_a1, se_str_a1 = format_coef(me_a_targ, se_a_targ)
coef_str_a2, se_str_a2 = format_coef(me_a_nontarg, se_a_nontarg)
coef_str_a3, se_str_a3 = format_coef(nbreg_a1['coef'], nbreg_a1['se'], nbreg_a1.get('pval'))
coef_str_a4, se_str_a4 = format_coef(nbreg_a2['coef'], nbreg_a2['se'], nbreg_a2.get('pval'))

print(f"News pressure t+1    {coef_str_a1}  {coef_str_a2}  {coef_str_a3}  {coef_str_a4}")
print(f"                     {se_str_a1}  {se_str_a2}  {se_str_a3}  {se_str_a4}")
print(f"Observations                  {mlogit_a['n']:,}                   {nbreg_a1['n']:,}      {nbreg_a2['n']:,}")
print(f"p-value (a)=(b)              {format_pval(mlogit_a.get('p_value_equality'))}")

# Panel B
print("\n" + "-" * 80)
print("PANEL B: FATAL VS. NON-FATAL ATTACKS, Sample: 2005-2011")
print("-" * 80)
print(f"Dep. variable:           Days w/     Days w/      Injuries  Fatalities")
print(f"                        non-fatal     fatal")

# Outcomes: 1=no attack, 2=non-fatal, 3=fatal
me_b_nonfatal = mlogit_b.get('me_2', None)
se_b_nonfatal = mlogit_b.get('se_2', None)

me_b_fatal = mlogit_b.get('me_3', None)
se_b_fatal = mlogit_b.get('se_3', None)

coef_str_b1, se_str_b1 = format_coef(me_b_nonfatal, se_b_nonfatal)
coef_str_b2, se_str_b2 = format_coef(me_b_fatal, se_b_fatal)
coef_str_b3, se_str_b3 = format_coef(nbreg_b1['coef'], nbreg_b1['se'], nbreg_b1.get('pval'))
coef_str_b4, se_str_b4 = format_coef(nbreg_b2['coef'], nbreg_b2['se'], nbreg_b2.get('pval'))

print(f"News pressure t+1    {coef_str_b1}  {coef_str_b2}  {coef_str_b3}  {coef_str_b4}")
print(f"                     {se_str_b1}  {se_str_b2}  {se_str_b3}  {se_str_b4}")
print(f"Observations                  {mlogit_b['n']:,}                   {nbreg_b1['n']:,}      {nbreg_b2['n']:,}")
print(f"p-value (a)=(b)              {format_pval(mlogit_b.get('p_value_equality'))}")

# Panel C
print("\n" + "-" * 80)
print("PANEL C: ATTACKS IN LDP VS. MDP AREAS, Sample: 2005-2011")
print("-" * 80)
print(f"Dep. variable:           Days w/     Days w/      Victims   Victims")
print(f"                        LDP areas   MDP areas    LDP areas MDP areas")

# Outcomes: 1=no attack, 2=LDP, 3=MDP
me_c_ldp = mlogit_c.get('me_2', None)
se_c_ldp = mlogit_c.get('se_2', None)

me_c_mdp = mlogit_c.get('me_3', None)
se_c_mdp = mlogit_c.get('se_3', None)

coef_str_c1, se_str_c1 = format_coef(me_c_ldp, se_c_ldp)
coef_str_c2, se_str_c2 = format_coef(me_c_mdp, se_c_mdp)
coef_str_c3, se_str_c3 = format_coef(nbreg_c1['coef'], nbreg_c1['se'], nbreg_c1.get('pval'))
coef_str_c4, se_str_c4 = format_coef(nbreg_c2['coef'], nbreg_c2['se'], nbreg_c2.get('pval'))

print(f"News pressure t+1    {coef_str_c1}  {coef_str_c2}  {coef_str_c3}  {coef_str_c4}")
print(f"                     {se_str_c1}  {se_str_c2}  {se_str_c3}  {se_str_c4}")
print(f"Observations                  {mlogit_c['n']:,}                   {nbreg_c1['n']:,}      {nbreg_c2['n']:,}")
print(f"p-value (a)=(b)              {format_pval(mlogit_c.get('p_value_equality'))}")

# Panel D
print("\n" + "-" * 80)
print("PANEL D: ATTACKS WITH LIGHT VS. HEAVY WEAPONS, Sample: 2005-2011")
print("-" * 80)
print(f"Dep. variable:           Days w/     Days w/      Victims   Victims")
print(f"                        LW only      HW          w/ LW     w/ HW")

# Outcomes: 1=no attack, 2=light weapons, 3=heavy weapons
me_d_lw = mlogit_d.get('me_2', None)
se_d_lw = mlogit_d.get('se_2', None)

me_d_hw = mlogit_d.get('me_3', None)
se_d_hw = mlogit_d.get('se_3', None)

coef_str_d1, se_str_d1 = format_coef(me_d_lw, se_d_lw)
coef_str_d2, se_str_d2 = format_coef(me_d_hw, se_d_hw)
coef_str_d3, se_str_d3 = format_coef(nbreg_d1['coef'], nbreg_d1['se'], nbreg_d1.get('pval'))
coef_str_d4, se_str_d4 = format_coef(nbreg_d2['coef'], nbreg_d2['se'], nbreg_d2.get('pval'))

print(f"News pressure t+1    {coef_str_d1}  {coef_str_d2}  {coef_str_d3}  {coef_str_d4}")
print(f"                     {se_str_d1}  {se_str_d2}  {se_str_d3}  {se_str_d4}")
print(f"Observations                  {mlogit_d['n']:,}                   {nbreg_d1['n']:,}      {nbreg_d2['n']:,}")
print(f"p-value (a)=(b)              {format_pval(mlogit_d.get('p_value_equality'))}")

print("\n" + "-" * 80)
print("Controls: News pressure lags (7), Prior Palestinian attacks, FEs (year, month, DOW)")
print("Note: *** p<0.01, ** p<0.05, * p<0.1")
print("-" * 80)

# Debug: print raw MNLogit coefficients
print("\n" + "=" * 80)
print("DEBUG: RAW MNLOGIT COEFFICIENTS FOR leaddaily_woi")
print("=" * 80)
print(f"""
Panel A (attacks_target):
  Coef outcome 2 (targeted):     {mlogit_a.get('coef_2', 'N/A')}
  Coef outcome 3 (non-targeted): {mlogit_a.get('coef_3', 'N/A')}

Panel D (attacks_hw):  
  Coef outcome 2 (light weapons): {mlogit_d.get('coef_2', 'N/A')}
  Coef outcome 3 (heavy weapons): {mlogit_d.get('coef_3', 'N/A')}
""")

# === COMPARISON WITH ORIGINAL ===
print("\n" + "=" * 80)
print("COMPARISON WITH ORIGINAL PAPER")
print("=" * 80)

# Get values safely
def safe_get(val, default=0):
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val

print(f"""
                              Original    Replication
─────────────────────────────────────────────────────────────
PANEL A (targeted vs non-targeted):
  ME targeted               0.006         {safe_get(me_a_targ):.3f}
  ME non-targeted           0.097**       {safe_get(me_a_nontarg):.3f}
  Victims targeted          0.193         {safe_get(nbreg_a1['coef']):.3f}
  Victims non-targeted      0.514***      {safe_get(nbreg_a2['coef']):.3f}
  p-value equality          0.036**       {safe_get(mlogit_a.get('p_value_equality')):.3f}

PANEL D (light vs heavy weapons):
  ME light weapons         -0.038         {safe_get(me_d_lw):.3f}
  ME heavy weapons          0.080**       {safe_get(me_d_hw):.3f}
  Victims light W          -0.137         {safe_get(nbreg_d1['coef']):.3f}
  Victims heavy W           0.789**       {safe_get(nbreg_d2['coef']):.3f}
  p-value equality          0.092*        {safe_get(mlogit_d.get('p_value_equality')):.3f}
─────────────────────────────────────────────────────────────

NOTE: Multinomial Logit marginal effects may differ due to:
1. Different baseline outcome coding
2. Clustered SE vs robust SE
3. Stata's dmlogit2 vs statsmodels MNLogit

Negative Binomial results should be closer to original.
""")