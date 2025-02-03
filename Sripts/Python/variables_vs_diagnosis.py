import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

variables = ['radius_mean', 'texture_mean', 'perimeter_mean',
             'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean',
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'symmetry_worst']

data_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"  # Cambia esto por la ruta real del dataset
df = pd.read_csv(data_path)

output_dir = 'plots_diagnosis'
os.makedirs(output_dir, exist_ok=True)

for var in variables:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=var, hue='diagnosis', element='step', stat='density', common_norm=False)
    plt.title(f'Distribución de {var} por Diagnosis', fontsize=16)
    plt.xlabel(var, fontsize=14)
    plt.ylabel('Densidad', fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'distribucion_{var}_por_diagnosis.png'))
    plt.close()

    # output_dir = os.path.join('Plots', 'variables_vs_diagnosis')