import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

def test_normalidad_shapiro_wilk(file_path, significance_level=0.05):
    """
    Realiza la prueba de normalidad Shapiro-Wilk para cada columna numérica en un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.

    Returns:
        None: Imprime un resumen de los resultados y genera gráficos para las variables normales.
    """
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

    print(f"Realizando el test de Shapiro-Wilk para {len(numeric_cols)} columnas numéricas...\n")

    # Realizar el test de Shapiro-Wilk para cada columna
    normal_columns = []
    for col in numeric_cols:
        values = data[col].dropna()  # Eliminar valores nulos
        stat, p_value = shapiro(values)  # Test de Shapiro-Wilk

        # Interpretar el resultado
        if p_value > significance_level:
            normal_columns.append(col)
            print(f"✅ '{col}': Puede considerarse normal (Estadístico: {stat:.4f}, p-value: {p_value:.4f}).")
            # Graficar comparación entre la distribución empírica y la normal
            plt.figure(figsize=(8, 6))
            sns.histplot(values, kde=True, stat="density", color="blue", label="Distribución empírica")
            sns.kdeplot(values, color="red", linestyle="--", label="Aproximación normal")
            plt.title(f"Comparación: Distribución empírica vs. Normal ({col})")
            plt.legend()
            plt.xlabel(col)
            plt.ylabel("Densidad")
            plt.show()
        else:
            print(f"❌ '{col}': No es normal (Estadístico: {stat:.4f}, p-value: {p_value:.4f}).")

    # Resumen final
    print("\nResumen:")
    if normal_columns:
        print(f"Las siguientes columnas pueden considerarse normales: {', '.join(normal_columns)}.")
    else:
        print("Ninguna columna numérica puede considerarse normal según la prueba de Shapiro-Wilk.")

# Ruta al archivo CSV
file_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"

# Llamar a la función con un nivel de significancia del 5%
test_normalidad_shapiro_wilk(file_path)
