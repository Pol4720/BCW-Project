import pandas as pd
from scipy.stats import kstest, norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def test_normalidad_kolmogorov_smirnov(file_path, significance_level=0.05,output_dir="plots"):
    """
    Realiza la prueba de Kolmogorov-Smirnov para evaluar la normalidad de cada columna numérica en un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.

    Returns:
        None: Imprime un resumen de los resultados y genera gráficos para las variables normales.
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

    # Realizar la prueba de Kolmogorov-Smirnov para cada columna
    normal_columns = []
    for col in numeric_cols:
        values = data[col].dropna()  # Eliminar valores nulos

        # Ajustar la media y desviación estándar de los datos a una distribución normal
        mean, std = values.mean(), values.std()
        standardized_values = (values - mean) / std

        # Prueba de K-S para la distribución normal
        stat, p_value = kstest(standardized_values, 'norm')

        # Interpretar el resultado
        if p_value > significance_level:
            normal_columns.append(col)
            print(f"✅ '{col}': Puede considerarse normal (Estadístico: {stat:.4f}, p-value: {p_value:.4f}).")

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
            print(f"❌ '{col}': No es normal (Estadístico: {stat:.4f}, p-value: {p_value:.4f}).")

    # Resumen final
    print("\nResumen:")
    if normal_columns:
        print(f"Las siguientes columnas pueden considerarse normales: {', '.join(normal_columns)}.")
    else:
        print("Ninguna columna numérica puede considerarse normal según la prueba de Kolmogorov-Smirnov.")


# Ruta al archivo CSV
file_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"

# Llamar a la función con un nivel de significancia del 5%
test_normalidad_kolmogorov_smirnov(file_path)
