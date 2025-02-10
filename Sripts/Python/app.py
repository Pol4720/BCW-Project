import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import confusion_matrix



def analyze_logistic_regression(data_path, not_include, output_dir, significance_level=0.05):
    """
    Perform logistic regression analysis on the given dataset.

    Parameters:
    data_path (str): Path to the dataset CSV file.
    not_include (list): List of columns to exclude from the analysis.
    output_dir (str): Path to the directory where plots will be saved.
    significance_level (float): Significance level for backward elimination process.

    Returns:
    dict: A dictionary containing model analysis results and paths to generated plots.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Define the dependent variable and independent variables
    y = df['diagnosis']
    X = df.drop(columns=['diagnosis'] + not_include)

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Check for multicollinearity and remove highly correlated variables
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X = X.drop(columns=to_drop)

    # Perform logistic regression with all variables
    model = sm.Logit(y, X).fit()

    # Backward elimination process
    variables_removed = []

    while True:
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_variable = p_values.idxmax()
            X = X.drop(columns=[excluded_variable])
            model = sm.Logit(y, X).fit()
            variables_removed.append((excluded_variable, max_p_value))
        else:
            break

    # Validate the model
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    residuals = y - y_pred_prob

    # Check if the mean and sum of residuals are close to 0
    mean_residuals = np.mean(residuals)
    sum_residuals = np.sum(residuals)

    # Kolmogorov-Smirnov test for normality
    ks_test = kstest(residuals, 'norm')

    # Durbin-Watson test for independence
    durbin_watson_test = durbin_watson(residuals)

    # Breusch-Pagan test for homoscedasticity
    _, bp_pvalue, _, _ = het_breuschpagan(residuals, model.model.exog)

    # Generate plots
    os.makedirs(output_dir, exist_ok=True)

    # Histogram of residuals with adjusted scale
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.xlim(residuals.min() - 0.00000000001, residuals.max() + 0.00000000001)  # Adjust the x-axis limits for small residuals
    plt.savefig(os.path.join(output_dir, 'histogram_residuals.png'))
    plt.close()

    # QQ plot of residuals
    qqplot(residuals, line='s')
    plt.title('QQ Plot of Residuals')
    plt.savefig(os.path.join(output_dir, 'qqplot_residuals.png'))
    plt.close()

    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    # Export results
    validation_stats = {
        'mean_residuals': mean_residuals,
        'sum_residuals': sum_residuals,
        'ks_test_statistic': ks_test.statistic,
        'ks_test_pvalue': ks_test.pvalue,
        'durbin_watson_statistic': durbin_watson_test,
        'breusch_pagan_pvalue': bp_pvalue,
        'variables_removed': variables_removed,
        'histogram_plot': os.path.join(output_dir, 'histogram_residuals.png'),
        'qqplot': os.path.join(output_dir, 'qqplot_residuals.png')
    }

    results = {
        'validation_stats': validation_stats,
        'confusion_matrix': conf_matrix,
        'model_summary': model.summary(),
        'model_summary2': model.summary2()
    }

    return results

