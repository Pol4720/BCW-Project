import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_normalidad_shapiro_wilk(file_path, significance_level=0.05, output_dir="plots"):
    """
    Realiza la prueba de normalidad Shapiro-Wilk para cada columna numérica en un archivo CSV y genera
    una tabla resumen con los resultados en formato de imagen.

    Args:
        file_path (str): Ruta al archivo CSV.
        significance_level (float): Nivel de significancia para el test. Default = 0.05.
        output_dir (str): Directorio donde se guardará la imagen de la tabla.

    Returns:
        None
    """
    # Crear el directorio si no existe
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

    print(f"Realizando el test de Shapiro-Wilk para {len(numeric_cols)} columnas numéricas...\n")

    # Lista para almacenar resultados
    results = []

    # Realizar el test de Shapiro-Wilk para cada columna
    for col in numeric_cols:
        values = data[col].dropna()  # Eliminar valores nulos
        stat, p_value = shapiro(values)  # Test de Shapiro-Wilk

        # Interpretar el resultado
        if p_value > significance_level:
            conclusion = 'Normal ✔️'  # ✔️
        else:
            conclusion = 'No Normal ❌'  # ❌

        # Almacenar resultados
        results.append([col, f"{stat:.4f}", f"{p_value:.4f}", conclusion])

    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame(results, columns=["Variable", "Estadístico", "P-Value", "Conclusión"])
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
    table_img_path = os.path.join(output_dir, 'resumen_shapiro_wilk.png')
    plt.savefig(table_img_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabla resumen guardada en: {table_img_path}")

# Ruta al archivo CSV
file_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"

# Llamar a la función con un nivel de significancia del 5%
test_normalidad_shapiro_wilk(file_path)