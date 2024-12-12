import pandas as pd
import numpy as np
from scipy.stats import anderson, norm
import matplotlib.pyplot as plt
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
def test_normalidad_anderson_darling(file_path, significance_level=0.05, output_dir="plots"):
    """
    Realiza el test de normalidad de Anderson-Darling para cada columna numérica en un archivo CSV.
    Genera gráficos para las variables que pueden considerarse normales.

    Args:
        file_path (str): Ruta al archivo CSV.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.
        output_dir (str): Directorio donde se guardarán los gráficos. Default = "plots".

    Returns:
        None: Imprime un resumen de los resultados.
    """
    # Crear el directorio para los gráficos si no existe
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

    print(f"Realizando el test de normalidad Anderson-Darling para {len(numeric_cols)} columnas numéricas...\n")

    normal_columns = []
    for col in numeric_cols:
        values = data[col].dropna()  # Eliminar valores nulos

        if len(values) == 0:
            print(f"⚠ '{col}': No tiene suficientes datos.")
            continue

        # Test de Anderson-Darling
        result = anderson(values, dist='norm')

        # Estadístico y p-valor aproximado
        statistic = result.statistic
        p_value = calcular_p_value(statistic)
        critical_values = result.critical_values
        significance_levels = result.significance_level

        # Determinar si es normal según el nivel de significancia proporcionado
        significance_index = next((i for i, sl in enumerate(significance_levels) if sl <= significance_level * 100), -1)

        if significance_index != -1 and statistic < critical_values[significance_index]:
            normal_columns.append(col)
            print(f"✅ '{col}': Puede considerarse como normal (Estadístico: {statistic:.4f}, "
                  f"Valor crítico: {critical_values[significance_index]:.4f}, "
                  f"Nivel de significancia: {significance_levels[significance_index]}%),"
                  f"P_value: {p_value:.4f}")

            # Generar gráfico para esta variable
            plt.figure(figsize=(8, 6))
            mu, sigma = np.mean(values), np.std(values)
            x = np.linspace(min(values), max(values), 100)
            y = norm.pdf(x, mu, sigma)

            # Histograma y distribución normal
            plt.hist(values, bins=20, density=True, alpha=0.6, color='blue', label="Distribución empírica")
            plt.plot(x, y, 'r-', label=f"Normal($\\mu={mu:.2f}, \\sigma={sigma:.2f}$)")

            plt.title(f"'{col}' - Comparación con distribución normal")
            plt.xlabel("Valor")
            plt.ylabel("Densidad")
            plt.legend()
            plt.grid()

            # Guardar el gráfico
            plot_path = os.path.join(output_dir, f"{col}_normalidad.png")
            plt.savefig(plot_path)
            plt.close()
        else:
            print(f"❌ '{col}': No es normal (Estadístico: {statistic:.4f}).",
            f"Valor crítico: {critical_values[significance_index]:.4f}, "
            f"Nivel de significancia: {significance_levels[significance_index]}%),"
            f"P_value: {p_value:.4f}")

    # Resumen final
    print("\nResumen:")
    if normal_columns:
        print(f"Las siguientes columnas pueden considerarse normales: {', '.join(normal_columns)}.")
        print(f"Se han generado gráficos para las columnas normales en el directorio '{output_dir}'.")
    else:
        print("Ninguna columna numérica puede considerarse normal según el test de Anderson-Darling.")

file_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"
test_normalidad_anderson_darling(file_path)