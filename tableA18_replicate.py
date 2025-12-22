"""
═══════════════════════════════════════════════════════════════════════════════
REPLICATION: TABLE A18
Content of News Stories that Appear on the Same Day and on the Next Day
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ═══════════════════════════════════════════════════════════════════════════
# 1. ПІДГОТОВКА ДАНИХ
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(df):
    """Створює необхідні змінні"""
    
    df = df.copy()
    
    # Convert categorical to numeric (0/1) if needed
    binary_vars = [
        'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20',
        'q4', 'q5', 'q6', 'q7', 'q8'
    ]

    for var in binary_vars:
        if var in df.columns:
            if df[var].dtype.name == 'category':
                # Convert category to numeric (assuming 0/1 or Yes/No)
                df[var] = pd.Categorical(df[var]).codes
                # If codes are -1, 0, 1, map to 0, 1
                if df[var].min() == -1:
                    df[var] = df[var].replace(-1, 0)

    # Create q21_q22 if not exists (interview with relatives OR witnesses)
    if 'q21_q22' not in df.columns:
        if 'q21' in df.columns and 'q22' in df.columns:
            # Convert if needed
            for var in ['q21', 'q22']:
                if df[var].dtype.name == 'category':
                    df[var] = pd.Categorical(df[var]).codes
                    if df[var].min() == -1:
                        df[var] = df[var].replace(-1, 0)
            # q21_q22 = 1 if q21==1 OR q22==1
            df['q21_q22'] = ((df['q21'] == 1) | (df['q22'] == 1)).astype(int)

    # Create composite variables
    # Info on attack severity (average of q11, q13, q14)
    df['q_info'] = (df['q11'] + df['q13'] + df['q14']) / 3

    # Images of victims (average of q16, q17, q18, q20)
    df['q_images'] = (df['q16'] + df['q17'] + df['q18'] + df['q20']) / 4

    # Personal stories (average of q19, q21_q22)
    df['q_personal_stories'] = (df['q19'] + df['q21_q22']) / 2

    # Create otherdays variable if not exists
    # otherdays = story NOT about same-day or previous-day attack
    if 'otherdays' not in df.columns:
        df['otherdays'] = ((df['q7'] == 0) & (df['q8'] == 0)).astype(int)

    # Create q8_palestine (interaction) if not exists
    # This is next-day coverage × Palestinian attack
    if 'q8_palestine' not in df.columns:
        df['q8_palestine'] = df['q8'] * df['q6']

    # Create monthyear if not exists
    if 'monthyear' not in df.columns:
        # Try to extract from date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['monthyear'] = df['date'].dt.year * 100 + df['date'].dt.month
        else:
            # Fallback: create a dummy grouping variable
            df['monthyear'] = 1

    # Check for coder_name, if not exists try alternatives
    if 'coder_name' not in df.columns:
        # Try alternative names
        for alt_name in ['coder', 'analyst', 'q1']:
            if alt_name in df.columns:
                df['coder_name'] = df[alt_name]
                break
        if 'coder_name' not in df.columns:
            # Create dummy coder
            df['coder_name'] = 1

    # Convert coder_name to string for categorical encoding
    if df['coder_name'].dtype.name == 'category':
        df['coder_name'] = df['coder_name'].astype(str)

    # Check for network
    if 'network' not in df.columns:
        # Try alternative names
        for alt_name in ['q3']:
            if alt_name in df.columns:
                df['network'] = df[alt_name]
                break
        if 'network' not in df.columns:
            # Create dummy network
            df['network'] = 1

    # Convert network to string for categorical encoding
    if df['network'].dtype.name == 'category':
        df['network'] = df['network'].astype(str)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. PANEL A: ТІЛЬКИ ІЗРАЇЛЬСЬКІ АТАКИ (БЕЗ ПАЛЕСТИНСЬКИХ)
# ═══════════════════════════════════════════════════════════════════════════

def panel_a_model(df, outcome_var):
    """
    Panel A: News about Israeli attacks only
    Sample: Israeli attacks, same or previous day, no Palestinian attacks
    """

    # Filter: q5==1 (conflict news), q4==1 (Israeli attack),
    #         (q7==1 | q8==1) (same or previous day), q6==0 (no Pal attack)
    df_sample = df[
        (df['q5'] == 1) &
        (df['q4'] == 1) &
        ((df['q7'] == 1) | (df['q8'] == 1)) &
        (df['q6'] == 0)
    ].copy()

    # Calculate mean for same-day coverage
    mean_same_day = df_sample[df_sample['q7'] == 1][outcome_var].mean()

    # Variables needed for regression
    model_vars = [outcome_var, 'q8', 'network', 'q6', 'coder_name', 'monthyear']

    # Drop rows with missing values in model variables
    df_sample = df_sample[model_vars].dropna()

    # Regression: outcome ~ q8 + network FE + q6 + coder FE
    formula = f'{outcome_var} ~ q8 + C(network) + q6 + C(coder_name)'

    model = smf.ols(formula, data=df_sample).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_sample['monthyear']}
    )

    return model, mean_same_day, int(model.nobs)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PANEL B: ВСІ НОВИНИ ПРО КОНФЛІКТ
# ═══════════════════════════════════════════════════════════════════════════

def panel_b_model(df, outcome_var):
    """
    Panel B: All conflict news stories
    Additional controls: q8_palestine, otherdays
    """

    # Full sample: all conflict news
    df_sample = df.copy()

    # Variables needed for regression
    model_vars = [outcome_var, 'q8', 'q8_palestine', 'otherdays',
                  'network', 'q4', 'q6', 'coder_name', 'monthyear']

    # Drop rows with missing values in model variables
    df_sample = df_sample[model_vars].dropna()

    # Regression with additional controls
    # q8_palestine = q8 × q6 (already created in prepare_data)
    formula = f'{outcome_var} ~ q8 + q8_palestine + otherdays + C(network) + q4 + q6 + C(coder_name)'

    model = smf.ols(formula, data=df_sample).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_sample['monthyear']}
    )

    return model, int(model.nobs)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ВИТЯГУВАННЯ РЕЗУЛЬТАТІВ
# ═══════════════════════════════════════════════════════════════════════════

def extract_results_panel_a(model, outcome_var, mean_same_day, n_obs):
    """Витягує результати для Panel A"""

    if 'q8' in model.params.index:
        return {
            'outcome': outcome_var,
            'next_day_coef': model.params['q8'],
            'next_day_se': model.bse['q8'],
            'next_day_pval': model.pvalues['q8'],
            'n_obs': n_obs,
            'r2': model.rsquared,
            'mean_same_day': mean_same_day
        }
    else:
        return None


def extract_results_panel_b(model, outcome_var, n_obs):
    """Витягує результати для Panel B"""

    results = {'outcome': outcome_var, 'n_obs': n_obs, 'r2': model.rsquared}

    key_vars = ['q8', 'q8_palestine', 'otherdays', 'q4', 'q6']

    for var in key_vars:
        if var in model.params.index:
            results[f'{var}_coef'] = model.params[var]
            results[f'{var}_se'] = model.bse[var]
            results[f'{var}_pval'] = model.pvalues[var]

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. КОМПАКТНА ТАБЛИЦЯ - PANEL A
# ═══════════════════════════════════════════════════════════════════════════

def print_panel_a_table(results_dict):
    """Форматує Panel A"""

    print("\n" + "═" * 95)
    print("PANEL A: Israeli attacks (same or previous day), no Palestinian attacks")
    print("═" * 95)
    print(f"{'Content':<40} {'Next-day':<11} {'N':<8} {'R²':<8} {'Same-day mean':<12}")
    print("─" * 95)

    content_labels = {
        'q12': 'Info on location',
        'q14': 'Info on civilian victims',
        'q19': 'Personal info',
        'q20': 'Burials/mourning',
        'q21_q22': 'Interviews',
        'q_info': 'Attack severity',
        'q_images': 'Images',
        'q_personal_stories': 'Personal touch'
    }

    for outcome, label in content_labels.items():
        if outcome in results_dict:
            res = results_dict[outcome]
            coef = res['next_day_coef']
            se = res['next_day_se']
            pval = res['next_day_pval']

            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

            print(f"{label:<40} {coef:>5.3f}{stars:<3}  {res['n_obs']:<8,} {res['r2']:<8.3f} {res['mean_same_day']:<12.3f}")
            print(f"{'':<40} ({se:>5.3f})")

    print("═" * 95)
    print("\nNote: *** p<0.01, ** p<0.05, * p<0.1")
    print("Standard errors in parentheses")
    print("All regressions include network and analyst fixed effects")
    print("Sample: News stories about Israeli attacks (same or previous day), excluding stories mentioning Palestinian attacks")


# ═══════════════════════════════════════════════════════════════════════════
# 6. КОМПАКТНА ТАБЛИЦЯ - PANEL B
# ═══════════════════════════════════════════════════════════════════════════

def print_panel_b_table(results_dict):
    """Форматує Panel B"""

    print("\n" + "═" * 120)
    print("PANEL B: All news stories about the Israeli-Palestinian conflict")
    print("═" * 120)
    print(f"{'Content':<28} {'Next-day':<11} {'N×Pal':<11} {'Other days':<11} {'Particular':<11} {'Palestinian':<11}")
    print(f"{'':28} {'(q8)':<11} {'(q8×q6)':<11} {'(otherdays)':<11} {'attack(q4)':<11} {'attack(q6)':<11}")
    print("─" * 120)

    content_labels = {
        'q12': 'Info on location',
        'q14': 'Info on civilian victims',
        'q19': 'Personal info',
        'q20': 'Burials/mourning',
        'q21_q22': 'Interviews',
        'q_info': 'Attack severity',
        'q_images': 'Images',
        'q_personal_stories': 'Personal touch'
    }

    for outcome, label in content_labels.items():
        if outcome in results_dict:
            res = results_dict[outcome]

            # Extract coefficients and format
            vars_order = ['q8', 'q8_palestine', 'otherdays', 'q4', 'q6']
            coeffs = []
            ses = []

            for var in vars_order:
                if f'{var}_coef' in res:
                    coef = res[f'{var}_coef']
                    se = res[f'{var}_se']
                    pval = res[f'{var}_pval']
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    coeffs.append(f"{coef:>6.3f}{stars:<3}")
                    ses.append(f"({se:>5.3f})")
                else:
                    coeffs.append("     ")
                    ses.append("       ")

            # Print coefficient row
            print(f"{label:<28} {coeffs[0]:11} {coeffs[1]:11} {coeffs[2]:11} {coeffs[3]:11} {coeffs[4]:11}")
            # Print SE row
            print(f"{'':<28} {ses[0]:11} {ses[1]:11} {ses[2]:11} {ses[3]:11} {ses[4]:11}")

    print("─" * 120)

    # Show sample statistics from first outcome
    first_outcome = list(content_labels.keys())[0]
    if first_outcome in results_dict:
        res = results_dict[first_outcome]
        print(f"{'Observations':<28} {res['n_obs']:<11,}")
        print(f"{'Average R²':<28} {np.mean([results_dict[k]['r2'] for k in content_labels if k in results_dict]):<11.3f}")

    print("═" * 120)
    print("\nNote: *** p<0.01, ** p<0.05, * p<0.1")
    print("Standard errors in parentheses")
    print("All regressions control for: network FE, analyst FE")
    print("q8 = Next-day coverage; q8×q6 = Next-day × Palestinian attack")
    print("otherdays = Story not about same/previous day; q4 = Story about particular attack")
    print("q6 = Story about Palestinian attack")


# ═══════════════════════════════════════════════════════════════════════════
# 7. ВИКОНАННЯ
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Основна функція"""

    # Load data - replication_file2.dta містить контент-аналіз новин
    df = pd.read_stata('/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file2.dta')
    df = prepare_data(df)

    # Diagnostic info
    print(f"\nДані завантажено: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print(f"monthyear: {df['monthyear'].notna().sum()} non-null values")
    print(f"network: {df['network'].notna().sum()} non-null values")
    print(f"coder_name: {df['coder_name'].notna().sum()} non-null values")

    print("\nФітую моделі...")

    # Content variables to analyze
    content_vars = [
        'q12',              # Exact location
        'q14',              # Number of civilian victims
        'q19',              # Personal info
        'q20',              # Burials/mourning
        'q21_q22',          # Interviews with relatives
        'q_info',           # Attack severity index
        'q_images',         # Images index
        'q_personal_stories' # Personal stories index
    ]

    # PANEL A
    print("\n[Panel A] Фітую моделі для ізраїльських атак...")
    results_a = {}

    for i, var in enumerate(content_vars, 1):
        print(f"  [{i}/{len(content_vars)}] {var}...")
        try:
            model, mean_same, n_obs = panel_a_model(df, var)
            results_a[var] = extract_results_panel_a(model, var, mean_same, n_obs)
        except Exception as e:
            print(f"    Помилка: {e}")
            continue

    # PANEL B
    print("\n[Panel B] Фітую моделі для всіх конфліктних новин...")
    results_b = {}

    for i, var in enumerate(content_vars, 1):
        print(f"  [{i}/{len(content_vars)}] {var}...")
        try:
            model, n_obs = panel_b_model(df, var)
            results_b[var] = extract_results_panel_b(model, var, n_obs)
        except Exception as e:
            print(f"    Помилка: {e}")
            # Print more diagnostic info
            if 'df' in locals():
                print(f"    Debug: df shape = {df.shape}")
                print(f"    Debug: {var} has {df[var].notna().sum()} non-null values")
                print(f"    Debug: monthyear has {df['monthyear'].notna().sum()} non-null values")
            continue

    # Print results
    print("═" * 79)
    print("TABLE A18: CONTENT OF NEWS STORIES (SAME DAY vs NEXT DAY)")
    print("═" * 79)
    print_panel_a_table(results_a)
    print_panel_b_table(results_b)

    print("\n✓ Всі моделі виконано успішно")


if __name__ == "__main__":
    main()