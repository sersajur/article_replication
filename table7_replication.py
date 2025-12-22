"""
═══════════════════════════════════════════════════════════════════════════════
REPLICATION: TABLE 7
Google Search Volume, Conflict-related News, and Timing of Attacks
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# 1. ПІДГОТОВКА ДАНИХ
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(df):
    """Створює interaction terms та лаги"""
    
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create lagged variables
    df['occurrence_lag1'] = df['occurrence'].shift(1)
    df['occurrence_lag2'] = df['occurrence'].shift(2)
    df['lnvic_lag1'] = df['lnvic'].shift(1)
    df['lnvic_lag2'] = df['lnvic'].shift(2)
    df['lnvic_pal_lag1'] = df['lnvic_pal'].shift(1)
    df['lnvic_pal_lag2'] = df['lnvic_pal'].shift(2)
    df['occurrence_pal_lag1'] = df['occurrence_pal'].shift(1)
    df['occurrence_pal_lag2'] = df['occurrence_pal'].shift(2)
    
    # Interaction terms: Any conflict news × Israeli attack
    df['news_attack_same'] = df['any_conflict_news'] * df['occurrence']
    df['news_attack_prev'] = df['any_conflict_news'] * df['occurrence_lag1']
    df['news_no_attack'] = df['any_conflict_news'] * (
        ~df['occurrence'].astype(bool) & ~df['occurrence_lag1'].astype(bool)
    )
    
    # Interaction terms: Length of conflict news × Israeli attack
    df['length_attack_same'] = df['length_conflict_news'] * df['occurrence']
    df['length_attack_prev'] = df['length_conflict_news'] * df['occurrence_lag1']
    df['length_no_attack'] = df['length_conflict_news'] * (
        ~df['occurrence'].astype(bool) & ~df['occurrence_lag1'].astype(bool)
    )
    
    # Lagged interaction terms
    for var in ['news_attack_same', 'news_attack_prev', 'news_no_attack',
                'length_attack_same', 'length_attack_prev', 'length_no_attack']:
        df[f'{var}_lag1'] = df[var].shift(1)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. МОДЕЛЬ 1: ANY NEWS × ATTACK (базова)
# ═══════════════════════════════════════════════════════════════════════════

def model_1(df):
    """Column 1: Any conflict news × Israeli attack interactions"""
    
    formula = '''
    conflict_searches ~ 
    news_attack_same + news_attack_prev + news_no_attack +
    lnvic + lnvic_lag1 + lnvic_pal + lnvic_pal_lag1 +
    occurrence + occurrence_lag1 + occurrence_pal + occurrence_pal_lag1 +
    monthyear + C(dow) + C(month)
    '''
    
    model = smf.ols(formula, data=df).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 7}
    )
    
    # Test: same-day vs previous-day coverage
    test_result = model.wald_test('news_attack_same = news_attack_prev')
    p_value = test_result.pvalue
    
    return model, {'p_same_vs_prev_t': p_value}


# ═══════════════════════════════════════════════════════════════════════════
# 3. МОДЕЛЬ 2: ANY NEWS × ATTACK (з лагами)
# ═══════════════════════════════════════════════════════════════════════════

def model_2(df):
    """Column 2: Any conflict news × Israeli attack + lagged interactions"""
    
    formula = '''
    conflict_searches ~ 
    news_attack_same + news_attack_prev + news_no_attack +
    news_attack_same_lag1 + news_attack_prev_lag1 + news_no_attack_lag1 +
    lnvic + lnvic_lag1 + lnvic_lag2 +
    lnvic_pal + lnvic_pal_lag1 + lnvic_pal_lag2 +
    occurrence + occurrence_lag1 + occurrence_lag2 +
    occurrence_pal + occurrence_pal_lag1 + occurrence_pal_lag2 +
    monthyear + C(dow) + C(month)
    '''
    
    model = smf.ols(formula, data=df).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 7}
    )
    
    # Tests
    test1 = model.wald_test('news_attack_same = news_attack_prev')
    test2 = model.wald_test('news_attack_same_lag1 = news_attack_prev_lag1')
    test3 = model.wald_test(
        'news_attack_same + news_attack_same_lag1 = ' +
        'news_attack_prev + news_attack_prev_lag1'
    )
    
    return model, {
        'p_same_vs_prev_t': test1.pvalue,
        'p_same_vs_prev_t_lag': test2.pvalue,
        'p_same_vs_prev_sum': test3.pvalue
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. МОДЕЛЬ 3: LENGTH × ATTACK (базова)
# ═══════════════════════════════════════════════════════════════════════════

def model_3(df):
    """Column 3: Length of conflict news × Israeli attack interactions"""
    
    formula = '''
    conflict_searches ~ 
    length_attack_same + length_attack_prev + length_no_attack +
    lnvic + lnvic_lag1 + lnvic_pal + lnvic_pal_lag1 +
    occurrence + occurrence_lag1 + occurrence_pal + occurrence_pal_lag1 +
    monthyear + C(dow) + C(month)
    '''
    
    model = smf.ols(formula, data=df).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 7}
    )
    
    # Test
    test_result = model.wald_test('length_attack_same = length_attack_prev')
    
    return model, {'p_same_vs_prev_t': test_result.pvalue}


# ═══════════════════════════════════════════════════════════════════════════
# 5. МОДЕЛЬ 4: LENGTH × ATTACK (з лагами)
# ═══════════════════════════════════════════════════════════════════════════

def model_4(df):
    """Column 4: Length of conflict news × Israeli attack + lagged interactions"""
    
    formula = '''
    conflict_searches ~ 
    length_attack_same + length_attack_prev + length_no_attack +
    length_attack_same_lag1 + length_attack_prev_lag1 + length_no_attack_lag1 +
    lnvic + lnvic_lag1 + lnvic_lag2 +
    lnvic_pal + lnvic_pal_lag1 + lnvic_pal_lag2 +
    occurrence + occurrence_lag1 + occurrence_lag2 +
    occurrence_pal + occurrence_pal_lag1 + occurrence_pal_lag2 +
    monthyear + C(dow) + C(month)
    '''
    
    model = smf.ols(formula, data=df).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 7}
    )
    
    # Tests
    test1 = model.wald_test('length_attack_same = length_attack_prev')
    test2 = model.wald_test('length_attack_same_lag1 = length_attack_prev_lag1')
    test3 = model.wald_test(
        'length_attack_same + length_attack_same_lag1 = ' +
        'length_attack_prev + length_attack_prev_lag1'
    )
    
    return model, {
        'p_same_vs_prev_t': test1.pvalue,
        'p_same_vs_prev_t_lag': test2.pvalue,
        'p_same_vs_prev_sum': test3.pvalue
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. ВИТЯГУВАННЯ РЕЗУЛЬТАТІВ
# ═══════════════════════════════════════════════════════════════════════════

def extract_results(model, key_vars, tests):
    """Витягує коефіцієнти для компактної таблиці"""
    
    results = {}
    
    for var in key_vars:
        if var in model.params.index:
            results[var] = {
                'coef': model.params[var],
                'se': model.bse[var],
                'pval': model.pvalues[var]
            }
    
    results['N'] = int(model.nobs)
    results['R2'] = model.rsquared
    results.update(tests)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. КОМПАКТНА ТАБЛИЦЯ
# ═══════════════════════════════════════════════════════════════════════════

def print_compact_table(results_dict):
    """Форматує результати як академічну таблицю"""
    
    print("\n" + "═" * 110)
    print(f"{'Variable':<50} {'(1)':<15} {'(2)':<15} {'(3)':<15} {'(4)':<15}")
    print(f"{'':50} {'OLS':<15} {'OLS':<15} {'OLS':<15} {'OLS':<15}")
    print("─" * 110)
    
    # Змінні для виводу
    vars_display = [
        ('news_attack_same', 'Any conflict news × Israeli attack, same day'),
        ('news_attack_prev', 'Any conflict news × Israeli attack, previous day'),
        ('news_attack_same_lag1', 'Any conflict news, t-1 × Israeli attack, same day'),
        ('news_attack_prev_lag1', 'Any conflict news, t-1 × Israeli attack, prev day'),
        ('length_attack_same', 'Length of conflict news × Israeli attack, same day'),
        ('length_attack_prev', 'Length of conflict news × Israeli attack, prev day'),
        ('length_attack_same_lag1', 'Length of conflict news, t-1 × Israeli attack, same'),
        ('length_attack_prev_lag1', 'Length of conflict news, t-1 × Israeli attack, prev')
    ]
    
    for var_name, var_label in vars_display:
        row = f"{var_label:<50}"
        
        for i in range(1, 5):
            model_name = f'Model_{i}'
            if model_name in results_dict and var_name in results_dict[model_name]:
                coef = results_dict[model_name][var_name]['coef']
                se = results_dict[model_name][var_name]['se']
                pval = results_dict[model_name][var_name]['pval']
                
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row += f"{coef:>6.3f}{stars:<3}     "
            else:
                row += f"{'':>15}"
        
        # Виводимо тільки ненульові рядки
        if any(model in results_dict and var_name in results_dict[model] 
               for model in [f'Model_{i}' for i in range(1, 5)]):
            print(row)
            
            # Standard errors
            row_se = f"{'':<50}"
            for i in range(1, 5):
                model_name = f'Model_{i}'
                if model_name in results_dict and var_name in results_dict[model_name]:
                    se = results_dict[model_name][var_name]['se']
                    row_se += f"({se:>5.3f})       "
                else:
                    row_se += f"{'':>15}"
            print(row_se)
    
    print("─" * 110)
    
    # Controls
    print(f"{'Occurrence & log victims (t, t-1)':<50} {'Yes':<15} {'Yes':<15} {'Yes':<15} {'Yes':<15}")
    print(f"{'Occurrence & log victims (t-2)':<50} {'No':<15} {'Yes':<15} {'No':<15} {'Yes':<15}")
    print(f"{'FEs (DOW, month); linear time trend':<50} {'Yes':<15} {'Yes':<15} {'Yes':<15} {'Yes':<15}")
    
    # Observations
    obs_row = f"{'Observations':<50}"
    for i in range(1, 5):
        model_name = f'Model_{i}'
        if model_name in results_dict:
            obs_row += f"{results_dict[model_name]['N']:<15,}"
    print(obs_row)
    
    # R-squared
    r2_row = f"{'R-squared':<50}"
    for i in range(1, 5):
        model_name = f'Model_{i}'
        if model_name in results_dict:
            r2_row += f"{results_dict[model_name]['R2']:<15.3f}"
    print(r2_row)
    
    # P-values
    print("─" * 110)
    pval_row1 = f"{'p-value: same-day vs. previous-day coverage, t':<50}"
    for i in range(1, 5):
        model_name = f'Model_{i}'
        if model_name in results_dict and 'p_same_vs_prev_t' in results_dict[model_name]:
            pval = results_dict[model_name]['p_same_vs_prev_t']
            stars = '*' if pval < 0.1 else ''
            pval_row1 += f"{pval:<6.3f}{stars:<9}"
    print(pval_row1)
    
    pval_row2 = f"{'p-value: same-day vs. previous-day coverage, t-1':<50}"
    for i in range(1, 5):
        model_name = f'Model_{i}'
        if model_name in results_dict and 'p_same_vs_prev_t_lag' in results_dict[model_name]:
            pval = results_dict[model_name]['p_same_vs_prev_t_lag']
            stars = '*' if pval < 0.1 else ''
            pval_row2 += f"{pval:<6.3f}{stars:<9}"
        else:
            pval_row2 += f"{'':>15}"
    print(pval_row2)
    
    pval_row3 = f"{'p-value: same-day vs. previous-day coverage, (t-1)+(t)':<50}"
    for i in range(1, 5):
        model_name = f'Model_{i}'
        if model_name in results_dict and 'p_same_vs_prev_sum' in results_dict[model_name]:
            pval = results_dict[model_name]['p_same_vs_prev_sum']
            stars = '*' if pval < 0.1 else ''
            pval_row3 += f"{pval:<6.3f}{stars:<9}"
        else:
            pval_row3 += f"{'':>15}"
    print(pval_row3)
    
    print("═" * 110)
    print("\nNote: *** p<0.01, ** p<0.05, * p<0.1")
    print("Standard errors (Newey-West) in parentheses")


# ═══════════════════════════════════════════════════════════════════════════
# 8. ВИКОНАННЯ
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Основна функція"""
    
    # Load data
    df = pd.read_stata('/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta')
    df = prepare_data(df)
    
    # Filter: Google data available from 2004
    df = df[df['conflict_searches'].notna()].copy()
    
    print("═" * 79)
    print("TABLE 7: GOOGLE SEARCH VOLUME, CONFLICT-RELATED NEWS, AND TIMING OF ATTACKS")
    print("═" * 79)
    print("\nФітую моделі...")
    
    # Key variables for extraction
    key_vars_map = {
        'Model_1': ['news_attack_same', 'news_attack_prev'],
        'Model_2': ['news_attack_same', 'news_attack_prev', 
                    'news_attack_same_lag1', 'news_attack_prev_lag1'],
        'Model_3': ['length_attack_same', 'length_attack_prev'],
        'Model_4': ['length_attack_same', 'length_attack_prev',
                    'length_attack_same_lag1', 'length_attack_prev_lag1']
    }
    
    # Fit models
    results_summary = {}
    
    print("  [1/4] Model 1: Any news × attack (base)...")
    m1, tests1 = model_1(df)
    results_summary['Model_1'] = extract_results(m1, key_vars_map['Model_1'], tests1)
    
    print("  [2/4] Model 2: Any news × attack (with lags)...")
    m2, tests2 = model_2(df)
    results_summary['Model_2'] = extract_results(m2, key_vars_map['Model_2'], tests2)
    
    print("  [3/4] Model 3: Length × attack (base)...")
    m3, tests3 = model_3(df)
    results_summary['Model_3'] = extract_results(m3, key_vars_map['Model_3'], tests3)
    
    print("  [4/4] Model 4: Length × attack (with lags)...")
    m4, tests4 = model_4(df)
    results_summary['Model_4'] = extract_results(m4, key_vars_map['Model_4'], tests4)
    
    # Print compact table
    print_compact_table(results_summary)
    
    print("\n✓ Всі моделі виконано успішно")


if __name__ == "__main__":
    main()
