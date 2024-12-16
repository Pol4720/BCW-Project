import pandas as pd
from scipy.stats import kstest, norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_normalidad_kolmogorov_smirnov(file_path, significance_level=0.05, output_dir="plots"):
    """
    Realiza la prueba de Kolmogorov-Smirnov para evaluar la normalidad de cada columna numérica en un archivo CSV
    y genera una tabla resumen con los resultados.

    Args:
        file_path (str): Ruta al archivo CSV.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.
        output_dir (str): Directorio donde se guardan las salidas.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cargar el dataset
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta '{file_path}'.")
        return
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_cols) == 0:
        print("El dataset no contiene columnas numéricas.")
        return

    print(f"Realizando la prueba de Kolmogorov-Smirnov para {len(numeric_cols)} columnas numéricas...\n")

    # Resultados
    results = []
    for col in numeric_cols:
        values = data[col].dropna()  # Eliminar valores nulos

        # Ajustar la media y desviación estándar de los datos a una distribución normal
        mean, std = values.mean(), values.std()
        standardized_values = (values - mean) / std

        # Prueba de K-S para la distribución normal
        stat, p_value = kstest(standardized_values, 'norm')

        # Conclusión
        conclusion = 'Normal' if p_value > significance_level else 'No Normal'
        symbol = '✔️' if p_value > significance_level else '❌'

        # Almacenar resultados
        results.append({
            'Variable': col,
            'Estadístico': f"{stat:.4f}",
            'P-Value': f"{p_value:.4f}",
            'Conclusión': f"{conclusion} {symbol}"
        })

    # Crear tabla resumen en DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    # Aplicar estilo de seaborn
    sns.set(style="whitegrid")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, len(results_df) * 0.7))
    ax.axis('off')  # Ocultar ejes
    # Crear una lista de colores para cada fila y columna
    n_rows = len(results_df)
    n_cols = len(results_df.columns)
    color_palette = sns.color_palette("coolwarm", n_rows)

    # Expandir los colores a todas las columnas
    cell_colours = [[color_palette[i]] * n_cols for i in range(n_rows)]

    # Agregar la tabla al gráfico
    table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     loc='center',
                     cellLoc='center',
                     cellColours=cell_colours)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Guardar tabla como imagen
    table_img_path = os.path.join(output_dir, 'resumen_normalidad.png')
    plt.savefig(table_img_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabla resumen guardada en: {table_img_path}")

# Ruta al archivo CSV
file_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"

# Llamar a la función con un nivel de significancia del 5%
test_normalidad_kolmogorov_smirnov(file_path)
