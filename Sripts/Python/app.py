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

# Load the dataset
data_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"
df = pd.read_csv(data_path)
not_include = ["id", "diagnosis2"]

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
significance_level = 0.05
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
y_pred = model.predict(X)
residuals = y - y_pred

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
output_dir = 'model_plots'
os.makedirs(output_dir, exist_ok=True)

# Histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.savefig(os.path.join(output_dir, 'histogram_residuals.png'))
plt.close()

# QQ plot of residuals
qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.savefig(os.path.join(output_dir, 'qqplot_residuals.png'))
plt.close()

# Export results
results = {
    'mean_residuals': mean_residuals,
    'sum_residuals': sum_residuals,
    'ks_test_statistic': ks_test.statistic,
    'ks_test_pvalue': ks_test.pvalue,
    'durbin_watson_statistic': durbin_watson_test,
    'breusch_pagan_pvalue': bp_pvalue,
    'model_equation': model.summary2().tables[1].to_string(),
    'variables_removed': variables_removed
}

with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: {value}\n')

print("Model analysis completed. Results and plots are saved in the 'model_plots' directory.")