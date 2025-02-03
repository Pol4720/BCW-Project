import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from pandas.plotting import table
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def create_correlation_analysis(data_path, variables, significance_level=0.5):
    # Cargar el dataset
    df = pd.read_csv(data_path)

    # Crear carpeta para los plots de correlación
    output_dir = 'plots_corr'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 12))
    sns.heatmap(df[variables].corr(method='pearson'), annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    plt.title('Matriz de Correlación de Pearson', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'matriz_correlacion_pearson.png'))
    plt.close()

    plt.figure(figsize=(12, 12))
    sns.heatmap(df[variables].corr(method='spearman'), annot=True, cmap='viridis', cbar=True, vmin=-1, vmax=1)
    plt.title('Matriz de Correlación de Spearman', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'matriz_correlacion_spearman.png'))
    plt.close()

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
            plt.savefig(os.path.join(pair_dir, f'dispersion_{var1}_vs_{var2}.png'))
            plt.close()

            # Ajuste de regresión lineal
            X = df[var1].values.reshape(-1, 1)
            y = df[var2].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Calcular r2 y mse
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # Verificar si la relación es lineal
            residuals = y - y_pred
            if np.all(np.abs(residuals) < 2 * np.std(residuals)) and np.isclose(r2, 1, atol=0.1) and np.isclose(np.sum(residuals), 0, atol=0.1) and np.isclose(np.mean(residuals), 0, atol=0.1):
                # Gráfico de regresión
                plt.figure(figsize=(10, 6))
                sns.regplot(x=df[var1], y=df[var2], scatter_kws={'alpha': 0.6}, line_kws={"color": "red"})
                plt.title(f'Regresión: {var1} vs {var2}', fontsize=16)
                plt.xlabel(var1, fontsize=14)
                plt.ylabel(var2, fontsize=14)
                plt.grid(True)
                plt.savefig(os.path.join(pair_dir, f'regresion_{var1}_vs_{var2}.png'))
                plt.close()

                # Guardar la ecuación del modelo y la justificación
                equation = f'{var2} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {var1}'
                justification = (
                    f'El modelo cumple con los supuestos de linealidad porque:\n'
                    f'1. La relación entre {var1} y {var2} es aproximadamente lineal.\n'
                    f'2. Los residuos se distribuyen aleatoriamente alrededor de cero.\n'
                    f'3. El coeficiente de determinación (R^2) es {r2:.2f}, lo que indica un buen ajuste.\n'
                    f'4. El error cuadrático medio (MSE) es {mse:.2f}, lo que indica que los errores son pequeños.'
                )
            else:
                equation = f'{var2} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {var1}'
                justification = (
                    f'El modelo no cumple con los supuestos de linealidad porque:\n'
                    f'2. Los residuos no se distribuyen aleatoriamente alrededor de cero.\n'
                    f'3. El coeficiente de determinación (R^2) es {r2:.2f}, lo que indica un mal ajuste.\n'
                    f'4. El error cuadrático medio (MSE) es {mse:.2f}, lo que indica que los errores son grandes.'
                )

            with open(os.path.join(pair_dir, f'modelo_{var1}_vs_{var2}.txt'), 'w') as f:
                f.write(f'Ecuación del modelo: {equation}\n\n')
                f.write(f'Justificación:\n{justification}\n')
            print(f'Modelo lineal guardado: {var1} vs {var2}')

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