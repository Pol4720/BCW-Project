import pandas as pd
import numpy as np
from scipy.stats import anderson, norm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calcular_p_value(statistic):
    """
    Aproxima el p-value para el estadístico de Anderson-Darling en distribuciones normales.

    Args:
        statistic (float): Estadístico de Anderson-Darling (A^2).

    Returns:
        float: p-value aproximado.
    """
    if statistic < 0.2:
        p_value = 1 - np.exp(-13.436 + 101.14 * statistic - 223.73 * statistic**2)
    elif statistic < 0.34:
        p_value = 1 - np.exp(-8.318 + 42.796 * statistic - 59.938 * statistic**2)
    elif statistic < 0.6:
        p_value = np.exp(0.9177 - 4.279 * statistic - 1.38 * statistic**2)
    else:
        p_value = np.exp(1.2937 - 5.709 * statistic + 0.0186 * statistic**2)

    return p_value

def test_normalidad_anderson_darling(file_path, variables, significance_level=0.05, output_dir="plots"):
    """
    Realiza la prueba de Anderson-Darling para evaluar la normalidad de cada columna numérica en un archivo CSV
    y genera una tabla resumen con los resultados.

    Args:
        file_path (str): Ruta al archivo CSV.
        variables (list): Lista de nombres de columnas a evaluar.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.
        output_dir (str): Directorio donde se guardan las salidas.

    Returns:
        None
    """
    # Crear el directorio de salida si no existe
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

    # Filtrar las variables proporcionadas que sean numéricas
    variables = [var for var in variables if var in numeric_cols]

    if len(variables) == 0:
        print("No se proporcionaron columnas numéricas válidas.")
        return

    print(f"Realizando la prueba de Anderson-Darling para {len(variables)} columnas numéricas...\n")

    # Resultados
    results = []
    for col in variables:
        values = data[col].dropna()  # Eliminar valores nulos

        if len(values) == 0:
            print(f"⚠ '{col}': No tiene suficientes datos.")
            continue

        # Test de Anderson-Darling
        result = anderson(values, dist='norm')
        statistic = result.statistic
        p_value = calcular_p_value(statistic)
        critical_values = result.critical_values
        significance_levels = result.significance_level

        # Determinar si es normal según el nivel de significancia proporcionado
        significance_index = next((i for i, sl in enumerate(significance_levels) if sl <= significance_level * 100), -1)

        if significance_index != -1 and statistic < critical_values[significance_index]:
            conclusion = 'Normal'
            symbol = '✔️'
        else:
            conclusion = 'No Normal'
            symbol = '❌'

        # Almacenar resultados
        results.append({
            'Variable': col,
            'Estadístico': f"{statistic:.4f}",
            'Valor Crítico': f"{critical_values[significance_index]:.4f}" if significance_index != -1 else "N/A",
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
    table_img_path = os.path.join(output_dir, 'resumen_anderson_darling.png')
    plt.savefig(table_img_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabla resumen guardada en: {table_img_path}")

