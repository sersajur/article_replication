"""
═══════════════════════════════════════════════════════════════════════════════
REPLICATION: TABLE 2
Coverage of Conflict, News Pressure, and Google Searches
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

# ═══════════════════════════════════════════════════════════════════════════
# 1. ПІДГОТОВКА ЗМІННИХ
# ═══════════════════════════════════════════════════════════════════════════

def extract_results(model, key_vars):
    """Витягує тільки ключові коефіцієнти з моделі"""
    results = {}

    for var in key_vars:
        if var in model.params.index:
            results[var] = {
                'coef': model.params[var],
                'se': model.bse[var],
                'pval': model.pvalues[var]
            }

    # Додаємо статистики
    results['N'] = int(model.nobs)

    if hasattr(model, 'rsquared'):
        results['R2'] = model.rsquared
    elif hasattr(model, 'prsquared'):
        results['Pseudo_R2'] = model.prsquared

    return results


def print_compact_table(results_dict):
    """Виводить результати у компактному форматі таблиці"""
    print("\n" + "═" * 110)
    print(f"{'Variable':<25} {'(1)':<15} {'(2)':<15} {'(3)':<15} {'(4)':<15} {'(5)':<15} {'(6)':<15}")
    print(f"{'':25} {'Any news':<15} {'Length':<15} {'Any news':<15} {'Length':<15} {'Google':<15} {'Google':<15}")
    print("─" * 110)

    # Ключові змінні для кожної колонки
    all_vars = [
        'occurrence_t_y',
        'occurrence_pal_t_y',
        'lnvic_t_y',
        'lnvic_pal_y',
        'daily_woi',
        'length_conflict_news_t_t_1'
    ]

    # Виводимо коефіцієнти
    for var in all_vars:
        row = f"{var:<25}"
        for i in range(1, 7):
            model_name = f'Model_{i}'
            if model_name in results_dict and var in results_dict[model_name]:
                coef = results_dict[model_name][var]['coef']
                se = results_dict[model_name][var]['se']
                pval = results_dict[model_name][var]['pval']

                # Форматування зірочок значимості
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row += f"{coef:>6.3f}{stars:<3} ({se:>5.3f})  "
            else:
                row += f"{'':>15}"
        print(row)

    print("─" * 110)

    # Статистики
    print(f"{'Fixed Effects':<25} {'Yes':<15} {'Yes':<15} {'Yes':<15} {'Yes':<15} {'Yes':<15} {'Yes':<15}")

    # Observations
    obs_row = f"{'Observations':<25}"
    for i in range(1, 7):
        model_name = f'Model_{i}'
        if model_name in results_dict:
            obs_row += f"{results_dict[model_name]['N']:<15,}"
    print(obs_row)

    # R-squared
    r2_row = f"{'R-squared / Pseudo R2':<25}"
    for i in range(1, 7):
        model_name = f'Model_{i}'
        if model_name in results_dict:
            if 'R2' in results_dict[model_name]:
                r2_row += f"{results_dict[model_name]['R2']:<15.3f}"
            elif 'Pseudo_R2' in results_dict[model_name]:
                r2_row += f"{results_dict[model_name]['Pseudo_R2']:<15.3f}"
    print(r2_row)

    print("═" * 110)
    print("\nNote: *** p<0.01, ** p<0.05, * p<0.1")
    print("Standard errors in parentheses")
    print("All models include year, month, and day-of-week fixed effects")


def prepare_data(df):
    """Створює необхідні змінні та дамі-змінні"""

    # Дефрагментація DataFrame
    df = df.copy()

    # Fixed Effects Dummies
    df['month_fe'] = pd.Categorical(df['month'])
    df['year_fe']  = pd.Categorical(df['year'])
    df['dow_fe']   = pd.Categorical(df['dow'])

    # Підвибірка з атаками
    df['attack_sample'] = (df['occurrence_t_y'] == 1) | (df['occurrence_pal_t_y'] == 1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. МОДЕЛЬ 1: OLS - ANY NEWS ON CONFLICT (ALL DAYS)
# ═══════════════════════════════════════════════════════════════════════════

def model_1(df):
    """Column 1: Any news on conflict ~ Israeli attack + Palestinian attack"""

    formula = 'any_conflict_news ~ occurrence_t_y + occurrence_pal_t_y + C(month) + C(year) + C(dow)'

    # Спочатку фітуємо без кластеризації, щоб знайти використані спостереження
    temp_fit = smf.ols(formula, data=df).fit()

    # Отримуємо індекси використаних спостережень
    used_idx = temp_fit.model.data.row_labels
    groups = df.loc[used_idx, 'monthyear']

    # Фітуємо з правильною кластеризацією
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': groups})

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 3. МОДЕЛЬ 2: NEGATIVE BINOMIAL - LENGTH OF NEWS (ALL DAYS)
# ═══════════════════════════════════════════════════════════════════════════

def model_2(df):
    """Column 2: Length of conflict news ~ Israeli attack + Palestinian attack"""

    formula = 'length_conflict_news ~ occurrence_t_y + occurrence_pal_t_y + C(month) + C(year) + C(dow)'

    # Спочатку фітуємо без кластеризації з кращими параметрами
    temp_fit = smf.negativebinomial(formula, data=df).fit(
        method='bfgs',
        maxiter=1000,
        disp=False
    )

    # Отримуємо індекси використаних спостережень
    used_idx = temp_fit.model.data.row_labels
    groups = df.loc[used_idx, 'monthyear']

    # Фітуємо з правильною кластеризацією
    model = smf.negativebinomial(formula, data=df).fit(
        method='bfgs',
        maxiter=1000,
        cov_type='cluster',
        cov_kwds={'groups': groups},
        disp=False
    )

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. МОДЕЛЬ 3: OLS - ANY NEWS (ATTACK DAYS ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def model_3(df):
    """Column 3: Any news ~ victims + victims_pal + news_pressure | attacks"""

    df_attack = df[df['attack_sample']].copy()
    formula   = 'any_conflict_news ~ lnvic_t_y + lnvic_pal_y + daily_woi + C(month) + C(year) + C(dow)'

    # Спочатку фітуємо без кластеризації
    temp_fit = smf.ols(formula, data=df_attack).fit()

    # Отримуємо індекси використаних спостережень
    used_idx = temp_fit.model.data.row_labels
    groups = df_attack.loc[used_idx, 'monthyear']

    # Фітуємо з правильною кластеризацією
    model = smf.ols(formula, data=df_attack).fit(cov_type='cluster', cov_kwds={'groups': groups})

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 5. МОДЕЛЬ 4: NEGATIVE BINOMIAL - LENGTH (ATTACK DAYS ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def model_4(df):
    """Column 4: Length ~ victims + victims_pal + news_pressure | attacks"""

    df_attack = df[df['attack_sample']].copy()
    formula   = 'length_conflict_news ~ lnvic_t_y + lnvic_pal_y + daily_woi + C(month) + C(year) + C(dow)'

    # Спочатку фітуємо без кластеризації з кращими параметрами
    temp_fit = smf.negativebinomial(formula, data=df_attack).fit(
        method='bfgs',
        maxiter=1000,
        disp=False
    )

    # Отримуємо індекси використаних спостережень
    used_idx = temp_fit.model.data.row_labels
    groups = df_attack.loc[used_idx, 'monthyear']

    # Фітуємо з правильною кластеризацією
    model = smf.negativebinomial(formula, data=df_attack).fit(
        method='bfgs',
        maxiter=1000,
        cov_type='cluster',
        cov_kwds={'groups': groups},
        disp=False
    )

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 6. МОДЕЛЬ 5: NEWEY-WEST - GOOGLE SEARCHES
# ═══════════════════════════════════════════════════════════════════════════

def model_5(df):
    """Column 5: Google searches ~ victims + victims_pal + time_trend"""

    df_valid = df[df['length_conflict_news_t_t_1'].notna()]
    formula  = 'conflict_searches ~ lnvic_t_y + lnvic_pal_y + monthyear + C(month) + C(year) + C(dow)'
    model    = smf.ols(formula, data=df_valid).fit(cov_type='HAC', cov_kwds={'maxlags': 7})

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 7. МОДЕЛЬ 6: NEWEY-WEST - GOOGLE SEARCHES + NEWS LENGTH
# ═══════════════════════════════════════════════════════════════════════════

def model_6(df):
    """Column 6: Google searches ~ victims + victims_pal + news_length + time"""

    formula = 'conflict_searches ~ lnvic_t_y + lnvic_pal_y + length_conflict_news_t_t_1 + monthyear + C(month) + C(year) + C(dow)'
    model   = smf.ols(formula, data=df).fit(cov_type='HAC', cov_kwds={'maxlags': 7})

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 8. ВИКОНАННЯ ТА ВИВЕДЕННЯ РЕЗУЛЬТАТІВ
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Основна функція для запуску всіх моделей"""

    df = pd.read_stata('/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta')
    df = prepare_data(df)

    print("═" * 79)
    print("TABLE 2: COVERAGE OF CONFLICT, NEWS PRESSURE, AND GOOGLE SEARCHES")
    print("═" * 79)
    print("\nFitting models...")

    # Витягуємо ключові змінні для кожної моделі
    key_vars_map = {
        'Model_1': ['occurrence_t_y', 'occurrence_pal_t_y'],
        'Model_2': ['occurrence_t_y', 'occurrence_pal_t_y'],
        'Model_3': ['lnvic_t_y', 'lnvic_pal_y', 'daily_woi'],
        'Model_4': ['lnvic_t_y', 'lnvic_pal_y', 'daily_woi'],
        'Model_5': ['lnvic_t_y', 'lnvic_pal_y'],
        'Model_6': ['lnvic_t_y', 'lnvic_pal_y', 'length_conflict_news_t_t_1']
    }

    # Фітуємо моделі
    models = {}
    results_summary = {}

    print("  [1/6] Model 1: OLS (Any news, all days)...")
    models['Model_1'] = model_1(df)
    results_summary['Model_1'] = extract_results(models['Model_1'], key_vars_map['Model_1'])

    print("  [2/6] Model 2: Negative Binomial (Length, all days)...")
    models['Model_2'] = model_2(df)
    results_summary['Model_2'] = extract_results(models['Model_2'], key_vars_map['Model_2'])

    print("  [3/6] Model 3: OLS (Any news, attack days)...")
    models['Model_3'] = model_3(df)
    results_summary['Model_3'] = extract_results(models['Model_3'], key_vars_map['Model_3'])

    print("  [4/6] Model 4: Negative Binomial (Length, attack days)...")
    models['Model_4'] = model_4(df)
    results_summary['Model_4'] = extract_results(models['Model_4'], key_vars_map['Model_4'])

    print("  [5/6] Model 5: Newey-West (Google searches)...")
    models['Model_5'] = model_5(df)
    results_summary['Model_5'] = extract_results(models['Model_5'], key_vars_map['Model_5'])

    print("  [6/6] Model 6: Newey-West (Google + news length)...")
    models['Model_6'] = model_6(df)
    results_summary['Model_6'] = extract_results(models['Model_6'], key_vars_map['Model_6'])

    # Виводимо компактну таблицю
    print_compact_table(results_summary)

if __name__ == "__main__":
    main()