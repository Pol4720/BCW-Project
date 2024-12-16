import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_analysis_plots(data_path, variables):
    # Cargar el dataset
    df = pd.read_csv(data_path)
    
    # Crear carpeta para los plots
    output_dir = 'plots_stats'
    os.makedirs(output_dir, exist_ok=True)

    for var in variables:
        # Crear carpeta para cada variable
        var_dir = os.path.join(output_dir, var)
        os.makedirs(var_dir, exist_ok=True)

        # Histograma
        plt.figure(figsize=(10, 6))
        sns.histplot(df[var], kde=True, color='skyblue')
        plt.title(f'Histograma de {var}', fontsize=16)
        plt.xlabel(var, fontsize=14)
        plt.ylabel('Frecuencia', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(var_dir, f'histograma_{var}.png'))
        plt.close()

        # Gráfico de pastel si la variable es cualitativa o discreta
        if df[var].dtype == 'object' or df[var].nunique() < 20:
            plt.figure(figsize=(8, 8))
            df[var].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            plt.title(f'Gráfico de Pastel de {var}', fontsize=16)
            plt.ylabel('')
            plt.savefig(os.path.join(var_dir, f'grafico_pastel_{var}.png'))
            plt.close()

        # Estadísticas descriptivas
        stats_desc = {
            'Media': df[var].mean(),
            'Moda': df[var].mode()[0],
            'Mediana': df[var].median(),
            'Primer Cuartil': df[var].quantile(0.25),
            'Tercer Cuartil': df[var].quantile(0.75)
        }

        stats_table = pd.DataFrame(list(stats_desc.items()), columns=['Estadístico', 'Valor'])
        stats_table_path = os.path.join(var_dir, f'estadisticas_centro_{var}.png')
        plt.figure(figsize=(10, 4))
        sns.heatmap(stats_table.set_index('Estadístico'), annot=True, cmap='coolwarm', cbar=False)
        plt.title(f'Estadísticos de Centro de {var}', fontsize=16)
        plt.savefig(stats_table_path)
        plt.close()

        # Estadísticas de dispersión
        dispersion_stats = {
            'Máximo': df[var].max(),
            'Mínimo': df[var].min(),
            'Rango': df[var].max() - df[var].min(),
            'Varianza': df[var].var(),
            'Desviación estándar': df[var].std(),
            'CV': (df[var].std() / df[var].mean()) * 100
        }

        dispersion_table = pd.DataFrame(list(dispersion_stats.items()), columns=['Estadístico', 'Valor'])
        dispersion_table_path = os.path.join(var_dir, f'estadisticos_dispersión_{var}.png')
        plt.figure(figsize=(8, 4))
        sns.heatmap(dispersion_table.set_index('Estadístico'), annot=True, cmap='coolwarm', cbar=False)
        plt.title(f'Estadísticos de Dispersión de {var}', fontsize=16)
        plt.savefig(dispersion_table_path)
        plt.close()

        # Gráfico de cajas y bigotes
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[var], color='lightgreen')
        plt.title(f'Gráfico de Cajas y Bigotes de {var}', fontsize=16)
        plt.xlabel(var, fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(var_dir, f'boxplot_{var}.png'))
        plt.close()

        # Contraste con distribuciones conocidas
        plt.figure(figsize=(10, 6))
        sns.histplot(df[var], kde=True, stat="density", color='skyblue', label='Datos Observados')
        
        # Distribución Normal
        mu, std = stats.norm.fit(df[var])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label='Normal')

        # Distribución Exponencial
        loc, scale = stats.expon.fit(df[var])
        p_exp = stats.expon.pdf(x, loc=loc, scale=scale)
        plt.plot(x, p_exp, 'r', linewidth=2, label='Exponencial')

        # Distribución Chi-cuadrado (grados de libertad = n-1)
        df_chi = len(df[var]) - 1
        p_chi = stats.chi2.pdf(x, df_chi)
        plt.plot(x, p_chi, 'g', linewidth=2, label='Chi-cuadrado')

        # Distribución Gamma (parámetros ajustables )
        a, loc, scale = stats.gamma.fit(df[var])
        p_gamma = stats.gamma.pdf(x, a, loc=loc, scale=scale)
        plt.plot(x, p_gamma, 'b', linewidth=2, label='Gamma')

        plt.title(f'Contraste de Distribuciones para {var}', fontsize=16)
        plt.xlabel(var, fontsize=14)
        plt.ylabel('Densidad', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(var_dir, f'distribuciones_conocidas_{var}.png'))
        plt.close()

# Ruta del dataset y variables a analizar
data_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"  # Cambia esto por la ruta real del dataset
variables = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'symmetry_worst']  

create_analysis_plots(data_path, variables)
