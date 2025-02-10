import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_vs_diagnosis(data_path, variable):
    """
    Plots the distribution of a specified variable against the diagnosis from a dataset and saves the plot as a PNG file.
    Parameters:
    data_path (str): The file path to the CSV dataset.
    variable (str): The name of the variable to plot against the diagnosis.
    Returns:
    str: The file path to the saved plot image.
    """
    df = pd.read_csv(data_path)
    output_dir = 'plots_diagnosis'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=variable, hue='diagnosis', element='step', stat='density', common_norm=False)
    plt.title(f'Distribución de {variable} por Diagnosis', fontsize=16)
    plt.xlabel(variable, fontsize=14)
    plt.ylabel('Densidad', fontsize=14)
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'distribucion_{variable}_por_diagnosis.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path