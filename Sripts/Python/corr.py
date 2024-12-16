import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from pandas.plotting import table

def create_correlation_analysis(data_path, variables, significance_level = 0.5):
    # Cargar el dataset
    df = pd.read_csv(data_path)
    
    # Crear carpeta para los plots de correlación
    output_dir = 'plots_corr'
    os.makedirs(output_dir, exist_ok=True)

    # Calcular el coeficiente de correlación de Pearson y generar combinaciones de variables
    correlations = df[variables].corr(method='pearson')
    combinations = [(variables[i], variables[j]) for i in range(len(variables)) for j in range(i + 1, len(variables))]

    for var1, var2 in combinations:
        # Coeficiente de correlación de Pearson
        pearson_corr = correlations.loc[var1, var2]
        if abs(pearson_corr) >= 0.7:
             # Crear carpeta para cada par de variables
             pair_dir = os.path.join(output_dir, f'{var1}_vs_{var2}')
             os.makedirs(pair_dir, exist_ok=True)

             # Gráfico de dispersión para ilustrar la correlación
             plt.figure(figsize=(10, 6))
             sns.scatterplot(x=df[var1], y=df[var2], color='blue', alpha=0.6)
             plt.title(f'Dispersión: {var1} vs {var2}', fontsize=16)
             plt.xlabel(var1, fontsize=14)
             plt.ylabel(var2, fontsize=14)
             plt.grid(True)
             plt.savefig(os.path.join(pair_dir, f'dispersión_{var1}_vs_{var2}.png'))
             plt.close()

            # Prueba de hipótesis Chi-cuadrado
             contingency_table = pd.crosstab(df[var1], df[var2])
             chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

             conclusion = "Rechazamos H0" if p_value < significance_level else "No rechazamos H0"
        
            # Crear tabla de resultados de Chi-cuadrado
             chi_table = pd.DataFrame({
                        'Estadístico': [chi2_stat],
                        'p-value': [p_value],
                        'Conclusión': [conclusion]
                     })

             chi_table_path = os.path.join(pair_dir, f'chi_cuadrado_{var1}_vs_{var2}.png')
             plt.figure(figsize=(8, 4))
             guardar_tabla(chi_table)                                                
             plt.title(f'Prueba Chi-cuadrado: {var1} vs {var2}', fontsize=16)
             plt.savefig(chi_table_path)
             plt.close()

        
             correlation_type = "Fuerte" if abs(pearson_corr) > 0.7 else "Moderada" if abs(pearson_corr) > 0.3 else "Débil"

                 # Tabla del coeficiente de correlación
             corr_info = pd.DataFrame({
                     'Coeficiente': [pearson_corr],
                     'Tipo de Correlación': [correlation_type]
                   })

             corr_info_path = os.path.join(pair_dir, f'coeficiente_correlacion_{var1}_vs_{var2}.png')
             plt.figure(figsize=(8, 4))
             #sns.heatmap(corr_info.set_index('Coeficiente'), annot=True, cmap='coolwarm', cbar=False)
             guardar_tabla(corr_info)
             plt.title(f'Coeficiente de Correlación: {var1} vs {var2}', fontsize=16)
             plt.savefig(corr_info_path)
             plt.close()

             # Gráfico de regresión
             plt.figure(figsize=(10, 6))
             sns.regplot(x=df[var1], y=df[var2], scatter_kws={'alpha':0.6}, line_kws={"color":"red"})
             plt.title(f'Regresión: {var1} vs {var2}', fontsize=16)
             plt.xlabel(var1, fontsize=14)
             plt.ylabel(var2, fontsize=14)
             plt.grid(True)
             plt.savefig(os.path.join(pair_dir, f'regresion_{var1}_vs_{var2}.png'))
             plt.close()


def guardar_tabla(df):
    fig, ax = plt.subplots(figsize=(8, 4))  # Ajusta el tamaño de la figura
    ax.axis('off')  # Oculta los ejes

    # Crear la tabla
    tabla = table(ax, df, loc='center', cellLoc='center')

    # Estilizar la tabla
    tabla.auto_set_font_size(False)  # Desactiva el ajuste automático del tamaño de fuente
    tabla.set_fontsize(12)  # Establece un tamaño de fuente específico
    tabla.scale(1.2, 1.2)  # Escala la tabla

    # Colores pastel para las celdas
    for (i, j), cell in tabla.get_celld().items():
        if i == 0:
            cell.set_facecolor('#FFDDC1')  # Color pastel para el encabezado
        else:
            cell.set_facecolor('#E6F7FF' if i % 2 == 0 else '#FFF3E6')  # Colores alternos para las filas
        cell.set_edgecolor('lightgrey')  # Color del borde de las celdas



# Ruta del dataset y lista de variables a analizar
data_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"  # Cambia esto por la ruta real del dataset
variables = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'symmetry_worst']  

create_correlation_analysis(data_path, variables)
