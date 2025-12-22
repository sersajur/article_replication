"""
Table 4 Replication: Palestinian Attacks and News Pressure
Replicates results from Durante & Zhuravskaya (JPE)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from scipy import stats


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load replication dataset"""
    df = pd.read_stata(filepath)
    df = df[df['gaza_war'] == 0].copy()  # Exclude Gaza War period
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ============================================================================
# MODEL ESTIMATION
# ============================================================================

def estimate_ols_cluster(formula, data, cluster_var='monthyear'):
    """OLS with clustered standard errors"""
    # Step 1: Fit without covariance to identify the sample
    temp_model = smf.ols(formula, data=data).fit()

    # Step 2: Get indices of rows used in the model
    # The model stores fitted values which have the same index as used data
    used_index = temp_model.fittedvalues.index

    # Step 3: Extract cluster groups for used observations only
    cluster_groups = data.loc[used_index, cluster_var]

    # Step 4: Fit again with cluster-robust covariance on same data
    # Use the subset of data that was actually used
    data_subset = data.loc[used_index]

    model = smf.ols(formula, data=data_subset).fit(
        cov_type='cluster',
        cov_kwds={'groups': cluster_groups}
    )

    return model


def estimate_newey_west(formula, data, lags=7):
    """OLS with Newey-West HAC standard errors"""
    # Step 1: Fit without covariance to identify the sample
    temp_model = smf.ols(formula, data=data).fit()

    # Step 2: Use only the data subset that was actually used
    used_index = temp_model.fittedvalues.index
    data_subset = data.loc[used_index]

    # Step 3: Fit with Newey-West covariance
    model = smf.ols(formula, data=data_subset).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': lags}
    )

    return model


def estimate_negbin_ml(formula, data, lags=7):
    """Negative Binomial ML with HAC standard errors"""
    # Step 1: Fit without covariance to identify the sample
    temp_model = smf.negativebinomial(formula, data=data).fit()

    # Step 2: Use only the data subset that was actually used
    used_index = temp_model.fittedvalues.index
    data_subset = data.loc[used_index]

    # Step 3: Fit with HAC covariance
    model = smf.negativebinomial(formula, data=data_subset).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': lags}
    )

    return model


# ============================================================================
# TABLE 4 SPECIFICATIONS
# ============================================================================

class Table4Replication:
    """Replicates Table 4: Palestinian Attacks and News Pressure"""

    def __init__(self, data):
        self.data = data
        self.results = {}
        self.base_controls = 'C(month) + C(year) + C(dow)'

    # ------------------------------------------------------------------------
    # Column 1: Occurrence ~ News Pressure (t)
    # ------------------------------------------------------------------------
    def estimate_col1(self):
        """Column 1: OLS with clustered SE"""
        formula = f'occurrence_pal ~ daily_woi + {self.base_controls}'
        model = estimate_ols_cluster(formula, self.data)
        self.results['col1'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 2: Occurrence ~ News Pressure (t, t+1)
    # ------------------------------------------------------------------------
    def estimate_col2(self):
        """Column 2: Newey-West (7 lags)"""
        formula = f'occurrence_pal ~ daily_woi + leaddaily_woi + {self.base_controls}'
        model = estimate_newey_west(formula, self.data)
        self.results['col2'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 3: Occurrence ~ Full specification
    # ------------------------------------------------------------------------
    def estimate_col3(self):
        """Column 3: Newey-West with lags + Israeli attacks"""
        # Lag variables: lagdaily_woi, lagdaily_woi2, ..., lagdaily_woi7
        lag_vars = 'lagdaily_woi + ' + ' + '.join([f'lagdaily_woi{i}' for i in range(2, 8)])
        israel_vars = 'occurrence_1 + occurrence_2_7 + occurrence_8_14'

        formula = (f'occurrence_pal ~ daily_woi + leaddaily_woi + '
                   f'{lag_vars} + {israel_vars} + {self.base_controls}')

        model = estimate_newey_west(formula, self.data)
        self.results['col3'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 4: Ln(1+victims) ~ News Pressure (t)
    # ------------------------------------------------------------------------
    def estimate_col4(self):
        """Column 4: OLS with clustered SE"""
        formula = f'lnvic_pal ~ daily_woi + {self.base_controls}'
        model = estimate_ols_cluster(formula, self.data)
        self.results['col4'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 5: Ln(1+victims) ~ News Pressure (t, t+1)
    # ------------------------------------------------------------------------
    def estimate_col5(self):
        """Column 5: Newey-West (7 lags)"""
        formula = f'lnvic_pal ~ daily_woi + leaddaily_woi + {self.base_controls}'
        model = estimate_newey_west(formula, self.data)
        self.results['col5'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 6: Ln(1+victims) ~ Full specification
    # ------------------------------------------------------------------------
    def estimate_col6(self):
        """Column 6: Newey-West with lags + Israeli attacks"""
        # Lag variables: lagdaily_woi, lagdaily_woi2, ..., lagdaily_woi7
        lag_vars = 'lagdaily_woi + ' + ' + '.join([f'lagdaily_woi{i}' for i in range(2, 8)])
        israel_vars = 'occurrence_1 + occurrence_2_7 + occurrence_8_14'

        formula = (f'lnvic_pal ~ daily_woi + leaddaily_woi + '
                   f'{lag_vars} + {israel_vars} + {self.base_controls}')

        model = estimate_newey_west(formula, self.data)
        self.results['col6'] = model
        return model

    # ------------------------------------------------------------------------
    # Column 7: Number of victims ~ Full specification
    # ------------------------------------------------------------------------
    def estimate_col7(self):
        """Column 7: ML Negative Binomial with HAC"""
        # Lag variables: lagdaily_woi, lagdaily_woi2, ..., lagdaily_woi7
        lag_vars = 'lagdaily_woi + ' + ' + '.join([f'lagdaily_woi{i}' for i in range(2, 8)])
        israel_vars = 'occurrence_1 + occurrence_2_7 + occurrence_8_14'

        formula = (f'victims_pal ~ daily_woi + leaddaily_woi + '
                   f'{lag_vars} + {israel_vars} + {self.base_controls}')

        model = estimate_negbin_ml(formula, self.data)
        self.results['col7'] = model
        return model

    # ------------------------------------------------------------------------
    # Estimate all models
    # ------------------------------------------------------------------------
    def estimate_all(self):
        """Estimate all 7 columns"""
        print("Estimating Table 4...")
        print("=" * 60)

        for i in range(1, 8):
            print(f"\nColumn {i}...")
            getattr(self, f'estimate_col{i}')()

        print("\n" + "=" * 60)
        print("Estimation complete.")
        return self.results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_results_table(results):
    """Format results as publication-ready table"""

    # Define coefficient names
    coef_names = {
        'daily_woi': 'News pressure t',
        'leaddaily_woi': 'News pressure t+1',
        'lagdaily_woi': 'News pressure t-1',
        'occurrence_1': 'Israeli attacks (previous day)',
        'occurrence_2_7': 'Israeli attacks (previous week)',
        'occurrence_8_14': 'Israeli attacks (week before previous)'
    }

    # Build table structure
    table_data = []

    for coef_key, coef_name in coef_names.items():
        row = {'Variable': coef_name}

        for col_num in range(1, 8):
            model = results[f'col{col_num}']

            if coef_key in model.params.index:
                # Coefficient
                coef = model.params[coef_key]
                # Robust standard error (already computed in model)
                se = model.bse[coef_key]
                # Significance stars
                p_val = model.pvalues[coef_key]
                stars = ('***' if p_val < 0.01 else
                        '**' if p_val < 0.05 else
                        '*' if p_val < 0.1 else '')

                row[f'Col{col_num}'] = f'{coef:.3f}{stars}\n({se:.3f})'
            else:
                row[f'Col{col_num}'] = '—'

        table_data.append(row)

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Add model info
    model_info = {
        'Variable': 'Model',
        'Col1': 'OLS', 'Col2': 'OLS', 'Col3': 'OLS',
        'Col4': 'OLS', 'Col5': 'OLS', 'Col6': 'OLS',
        'Col7': 'ML Neg. Bin.'
    }

    # Add observations
    obs_info = {
        'Variable': 'Observations',
        **{f'Col{i}': str(results[f'col{i}'].nobs)
           for i in range(1, 8)}
    }

    # Add R-squared
    r2_info = {
        'Variable': '(Pseudo) R-squared',
        **{f'Col{i}': f"{results[f'col{i}'].rsquared:.3f}"
           for i in range(1, 7)}
    }
    r2_info['Col7'] = f"{results['col7'].prsquared:.3f}"

    # Combine all
    table_df = pd.concat([
        table_df,
        pd.DataFrame([model_info, obs_info, r2_info])
    ], ignore_index=True)

    return table_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main replication workflow"""

    print("\n" + "=" * 60)
    print("TABLE 4 REPLICATION")
    print("Palestinian Attacks and News Pressure")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    DATA_FILE = "/Users/sersajur/Documents/PhD/econometric/replication/dta/replication_file1.dta"
    data = load_data(DATA_FILE)
    print(f"Observations: {len(data)}\n")

    # Estimate models
    replicator = Table4Replication(data)
    results = replicator.estimate_all()

    # Format output
    print("\nFormatting results...")
    table = format_results_table(results)

    # Display
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60 + "\n")
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()