import numpy as np
import streamlit as st
import pandas as pd
import os
import sys
import json

# Agregar la ruta absoluta de los módulos
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python/Test')
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python')

from shapiro import test_normalidad_shapiro_wilk
from normalTest import test_normalidad_anderson_darling
from kolmogorov import test_normalidad_kolmogorov_smirnov
from stats import create_analysis_plots
from corr import create_correlation_analysis
from variables_vs_diagnosis import plot_variable_vs_diagnosis
from app import analyze_logistic_regression


import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import kstest, norm, chi2, f
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import chi2_contingency


# Cargar la información descriptiva de las variables
with open('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/UI/variable_info.json', 'r', encoding='utf-8') as f:
    variable_info = json.load(f)

variables = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'symmetry_worst']  

# Configuración de la página
st.set_page_config(page_title="Proyecto Estadística", layout="wide")

# Título del proyecto
st.title("Proyecto Final de Estadística")
st.subheader("Análisis del dataset Breast Cancer Wisconsin (Diagnostics)")

# Cargar el dataset
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset', 'data.csv')
df = pd.read_csv(data_path)

# Secciones del informe
sections = ["Introducción", "Análisis Descriptivo de los datos", "Análisis de la distribución", "Estimación de parámetros", "Pruebas de Hipótesis", "Correlación e Independencia", "Modelo Logístico"]

# Sidebar para navegación
st.sidebar.title("Navegación")
selected_section = st.sidebar.radio("Ir a", sections)

# Introducción
if selected_section == "Introducción":
    st.header("Introducción")
    st.write("""
    Este proyecto de análisis estadístico, realizado como parte del plan de estudios de Ciencia de la Computación, se centra en el conjunto de datos Breast Cancer Wisconsin (Diagnostics). 
    Nuestro objetivo es ir más allá de una simple clasificación y profundizar en un análisis estadístico exhaustivo, explorando las características de los datos y sus relaciones intrínsecas.
    """)

# Análisis Descriptivo de los datos
elif selected_section == "Análisis Descriptivo de los datos":
    st.header("Análisis Descriptivo de los datos")
    st.write("A continuación se muestra un cuadro con todas las variables presentes en el dataset, de conjunto con su clasificación estadística y su escala de medición.")
    st.dataframe(df.describe())
    st.write("Seleccione una variable para ver su análisis descriptivo:")
    variable = st.selectbox("Variable", variables)
    
    # Mostrar información descriptiva de la variable
    st.subheader(f"Información descriptiva de {variable}")
    st.write(f"**Descripción:** {variable_info[variable]['descripcion']}")
    st.write(f"**Clasificación Estadística:** {variable_info[variable]['clasificacion']}")
    st.write(f"**Escala de Medición:** {variable_info[variable]['escala']}")
    st.write(f"**Medición:** {variable_info[variable]['medicion']}")
    if 'unidades' in variable_info[variable]:
        st.write(f"**Unidades:** {variable_info[variable]['unidades']}")
    if 'interpretacion' in variable_info[variable]:
        st.write(f"**Interpretación:** {variable_info[variable]['interpretacion']}")
    
    create_analysis_plots(data_path, [variable])
    if variable == 'diagnosis':
        st.image(f'plots_stats/{variable}/grafico_pastel_diagnosis.png')
    else:
        st.image(f'plots_stats/{variable}/histograma_{variable}.png')
        st.image(f'plots_stats/{variable}/distribuciones_conocidas_{variable}.png')

    st.image(f'plots_stats/{variable}/estadisticas_centro_{variable}.png')
    st.image(f'plots_stats/{variable}/estadisticos_dispersión_{variable}.png')
    st.image(f'plots_stats/{variable}/boxplot_{variable}.png')

    if variable != 'diagnosis':
        if st.button(f"Mostrar análisis de {variable} para tumores malignos y benignos"):
            plot_variable_vs_diagnosis(data_path, variable)
            st.image(f'plots_diagnosis/distribucion_{variable}_por_diagnosis.png')

# Análisis de la distribución
elif selected_section == "Análisis de la distribución":
    st.header("Análisis de la distribución")
    st.write("Pruebas de Normalidad")
    st.write("Seleccione una prueba de normalidad para ver los resultados:")
    test = st.selectbox("Prueba de Normalidad", ["Kolmogorov-Smirnov", "Shapiro-Wilk", "Anderson-Darling"])
    if test == "Kolmogorov-Smirnov":
        test_normalidad_kolmogorov_smirnov(data_path, variables)
        st.image('plots/resumen_normalidad.png')
    elif test == "Shapiro-Wilk":
        test_normalidad_shapiro_wilk(data_path)
        st.image('plots/resumen_shapiro_wilk.png')
    elif test == "Anderson-Darling":
        test_normalidad_anderson_darling(data_path, variables)
        st.image('plots/resumen_anderson_darling.png')

# Estimación de parámetros
elif selected_section == "Estimación de parámetros":
    st.header("Estimación de parámetros")
    st.write("Seleccione una variable para ver su estimación de parámetros (Asumiendo Normalidad):")
    variable = st.selectbox("Variable", variables)
    
    st.write("Seleccione el nivel de significancia:")
    alpha = st.selectbox("Nivel de Significancia", [0.01, 0.05, 0.025, 0.1, 0.005])
    
    st.write("Seleccione el tipo de estimación:")
    estimation_type = st.radio("Tipo de Estimación", ["Estimación Puntual", "Estimación por Intervalos"])

    sample_size = st.number_input(f"Seleccione el tamaño de la muestra para {variable}", min_value=1, max_value=len(df), value=30)
    
    if estimation_type == "Estimación Puntual":
        st.write(f"Estimación Puntual para {variable}")
        sample = df[variable].sample(n=sample_size, random_state=1)
        mean = sample.mean()
        variance = sample.var()
        st.write(f"Media muestral de {variable}: {mean}")
        st.write(f"Varianza muestral de {variable}: {variance}")
    elif estimation_type == "Estimación por Intervalos":
        st.write(f"Estimación por Intervalos para {variable} con un nivel de significancia de {alpha}")
        sample = df[variable].sample(n=sample_size, random_state=1)
        mean = sample.mean()
        variance = sample.var()
        st.write(f"Media muestral de {variable}: {mean}")
        st.write(f"Varianza muestral de {variable}: {variance}")

        if variable == 'diagnosis':
            # Intervalo de confianza para la proporción (variable diagnosis)
            p_hat = mean
            z = np.abs(np.percentile(np.random.standard_normal(1000000), [(1-alpha/2)*100, alpha/2*100]))
            ci_lower = p_hat - z[0] * np.sqrt((p_hat * (1 - p_hat)) / sample_size)
            ci_upper = p_hat + z[0] * np.sqrt((p_hat * (1 - p_hat)) / sample_size)
            st.write(f"Intervalo de confianza para la proporción (p) con un nivel de significancia de {alpha}:")
            st.write(f"Fórmula: p̂ ± Z_(1-α/2) * sqrt((p̂ * (1 - p̂)) / n)")
            st.write(f"Intervalo de confianza: ({ci_lower}, {ci_upper})")
        else:
            # Intervalo de confianza para la media
            z = np.abs(np.percentile(np.random.standard_normal(1000000), [(1-alpha/2)*100, alpha/2*100]))
            ci_lower = mean - z[0] * (np.sqrt(variance) / np.sqrt(sample_size))
            ci_upper = mean + z[0] * (np.sqrt(variance) / np.sqrt(sample_size))
            st.write(f"Intervalo de confianza para la media con un nivel de significancia de {alpha}:")
            st.write(f"Fórmula: x̄ ± Z_(1-α/2) * (σ / sqrt(n))")
            st.write(f"Intervalo de confianza: ({ci_lower}, {ci_upper})")

            # Intervalo de confianza para la varianza
            chi2_lower = np.percentile(np.random.chisquare(sample_size-1, 1000000), alpha/2*100)
            chi2_upper = np.percentile(np.random.chisquare(sample_size-1, 1000000), (1-alpha/2)*100)
            ci_lower_var = (sample_size - 1) * variance / chi2_upper
            ci_upper_var = (sample_size - 1) * variance / chi2_lower
            st.write(f"Intervalo de confianza para la varianza con un nivel de significancia de {alpha}:")
            st.write(f"Fórmula: ((n-1)s² / χ²_(1-α/2), (n-1)s² / χ²_(α/2))")
            st.write(f"Intervalo de confianza: ({ci_lower_var}, {ci_upper_var})")

# Pruebas de Hipótesis
elif selected_section == "Pruebas de Hipótesis":
    st.header("Pruebas de Hipótesis")
    st.write("Seleccione una prueba de hipótesis para ver los resultados:")
    hypothesis_test = st.selectbox("Prueba de Hipótesis", ["Media de una población", "Proporción de una población", "Varianza de una población", "Igualdad de medias", "Igualdad de proporciones", "Igualdad de varianzas"])
    st.write("Seleccione el nivel de significancia:")
    alpha = st.selectbox("Nivel de Significancia", [0.01, 0.05, 0.025, 0.1, 0.005])
    st.write("Seleccione el tipo de prueba:")
    tail_type = st.radio("Tipo de Prueba", ["Bilateral", "Unilateral (cola inferior)", "Unilateral (cola superior)"])

    # Pruebas de Hipótesis para una población
    if hypothesis_test in ["Media de una población", "Proporción de una población", "Varianza de una población"]:
        st.subheader("Pruebas de Hipótesis para una población")
        variable = st.selectbox("Variable", variables)
        sample_size = st.number_input(f"Seleccione el tamaño de la muestra para {variable}", min_value=1, max_value=len(df), value=30)
        sample = df[variable].sample(n=sample_size, random_state=1)
        mean = sample.mean()
        variance = sample.var()
        st.write(f"Media muestral de {variable}: {mean}")
        st.write(f"Varianza muestral de {variable}: {variance}")

        if hypothesis_test == "Media de una población":
            mu = st.number_input("Ingrese el valor de la media poblacional (H0):", value=20.0)
            st.write(f"Prueba de Hipótesis para la media de {variable}")
            st.write(f"H0: μ = {mu}")
            if tail_type == "Bilateral":
                st.write(f"H1: μ ≠ {mu}")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: μ < {mu}")
            else:
                st.write(f"H1: μ > {mu}")
            z = (mean - mu) / (np.sqrt(variance) / np.sqrt(sample_size))
            if tail_type == "Bilateral":
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.norm.cdf(z)
            else:
                p_value = 1 - stats.norm.cdf(z)
            st.write(f"Estadístico Z: {z}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")

        elif hypothesis_test == "Proporción de una población":
            p = st.number_input("Ingrese el valor de la proporción poblacional (H0):", value=0.5)
            st.write(f"Prueba de Hipótesis para la proporción de {variable}")
            st.write(f"H0: p = {p}")
            if tail_type == "Bilateral":
                st.write(f"H1: p ≠ {p}")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: p < {p}")
            else:
                st.write(f"H1: p > {p}")
            p_hat = mean
            z = (p_hat - p) / np.sqrt((p * (1 - p)) / sample_size)
            if tail_type == "Bilateral":
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.norm.cdf(z)
            else:
                p_value = 1 - stats.norm.cdf(z)
            st.write(f"Estadístico Z: {z}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")

        elif hypothesis_test == "Varianza de una población":
            sigma2 = st.number_input("Ingrese el valor de la varianza poblacional (H0):", value=0.001)
            st.write(f"Prueba de Hipótesis para la varianza de {variable}")
            st.write(f"H0: σ² = {sigma2}")
            if tail_type == "Bilateral":
                st.write(f"H1: σ² ≠ {sigma2}")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: σ² < {sigma2}")
            else:
                st.write(f"H1: σ² > {sigma2}")
            chi2 = (sample_size - 1) * variance / sigma2
            if tail_type == "Bilateral":
                p_value = 2 * min(stats.chi2.cdf(chi2, sample_size - 1), 1 - stats.chi2.cdf(chi2, sample_size - 1))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.chi2.cdf(chi2, sample_size - 1)
            else:
                p_value = 1 - stats.chi2.cdf(chi2, sample_size - 1)
            st.write(f"Estadístico Chi-cuadrado: {chi2}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")

    # Pruebas de Hipótesis para dos poblaciones
    elif hypothesis_test in ["Igualdad de medias", "Igualdad de proporciones", "Igualdad de varianzas"]:
        st.subheader("Pruebas de Hipótesis para dos poblaciones")
        variable = st.selectbox("Variable", variables)
        sample_size = st.number_input(f"Seleccione el tamaño de la muestra para {variable}", min_value=1, max_value=len(df), value=30)
        group1 = st.selectbox("Grupo 1", df['diagnosis'].unique())
        group2 = st.selectbox("Grupo 2", df['diagnosis'].unique())
        sample1 = df[df['diagnosis'] == group1][variable].sample(n=sample_size, random_state=1)
        sample2 = df[df['diagnosis'] == group2][variable].sample(n=sample_size, random_state=1)
        mean1 = sample1.mean()
        mean2 = sample2.mean()
        var1 = sample1.var()
        var2 = sample2.var()
        st.write(f"Media muestral de {variable} para {group1}: {mean1}")
        st.write(f"Media muestral de {variable} para {group2}: {mean2}")
        st.write(f"Varianza muestral de {variable} para {group1}: {var1}")
        st.write(f"Varianza muestral de {variable} para {group2}: {var2}")

        if hypothesis_test == "Igualdad de medias":
            st.write(f"Prueba de Hipótesis para la igualdad de medias de {variable}")
            st.write(f"H0: μ1 = μ2")
            if tail_type == "Bilateral":
                st.write(f"H1: μ1 ≠ μ2")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: μ1 < μ2")
            else:
                st.write(f"H1: μ1 > μ2")
            t, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
            if tail_type == "Bilateral":
                p_value = 2 * (1 - stats.t.cdf(abs(t), df=sample_size-1))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.t.cdf(t, df=sample_size-1)
            else:
                p_value = 1 - stats.t.cdf(t, df=sample_size-1)
            st.write(f"Estadístico t: {t}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")

        elif hypothesis_test == "Igualdad de proporciones":
            st.write(f"Prueba de Hipótesis para la igualdad de proporciones de {variable}")
            st.write(f"H0: p1 = p2")
            if tail_type == "Bilateral":
                st.write(f"H1: p1 ≠ p2")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: p1 < p2")
            else:
                st.write(f"H1: p1 > p2")
            p1 = mean1
            p2 = mean2
            p_combined = (p1 * sample_size + p2 * sample_size) / (2 * sample_size)
            z = (p1 - p2) / np.sqrt(p_combined * (1 - p_combined) * (2 / sample_size))
            if tail_type == "Bilateral":
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.norm.cdf(z)
            else:
                p_value = 1 - stats.norm.cdf(z)
            st.write(f"Estadístico Z: {z}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")

        elif hypothesis_test == "Igualdad de varianzas":
            st.write(f"Prueba de Hipótesis para la igualdad de varianzas de {variable}")
            st.write(f"H0: σ1² = σ2²")
            if tail_type == "Bilateral":
                st.write(f"H1: σ1² ≠ σ2²")
            elif tail_type == "Unilateral (cola inferior)":
                st.write(f"H1: σ1² < σ2²")
            else:
                st.write(f"H1: σ1² > σ2²")
            f = var1 / var2
            dfn = sample_size - 1
            dfd = sample_size - 1
            if tail_type == "Bilateral":
                p_value = 2 * min(stats.f.cdf(f, dfn, dfd), 1 - stats.f.cdf(f, dfn, dfd))
            elif tail_type == "Unilateral (cola inferior)":
                p_value = stats.f.cdf(f, dfn, dfd)
            else:
                p_value = 1 - stats.f.cdf(f, dfn, dfd)
            st.write(f"Estadístico F: {f}")
            st.write(f"Valor p: {p_value}")
            if p_value < alpha:
                st.write("Resultado: Rechazamos la hipótesis nula (H0)")
            else:
                st.write("Resultado: No rechazamos la hipótesis nula (H0)")


# Correlación e Independencia
elif selected_section == "Correlación e Independencia":
    st.header("Correlación e Independencia")
    st.write("Seleccione un tipo de análisis de correlación:")
    correlation_type = st.selectbox("Tipo de Correlación", ["Pearson", "Spearman"])
    create_correlation_analysis(data_path, variables)
    if correlation_type == "Pearson":
        st.image('plots_corr/matriz_correlacion_pearson.png')
    elif correlation_type == "Spearman":
        st.image('plots_corr/matriz_correlacion_spearman.png')
    st.write("Seleccione un par de variables para ver su análisis de correlación detallado:")
    var1 = st.selectbox("Variable 1", variables)
    var2 = st.selectbox("Variable 2", variables)
    # Aquí se pueden agregar funciones para mostrar gráficos de dispersión y regresión para el par de variables seleccionado
    # Mostrar gráfico de dispersión
    st.subheader(f"Análisis de Correlación entre {var1} y {var2}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[var1], y=df[var2])
    plt.title(f"Gráfico de Dispersión entre {var1} y {var2}")
    scatter_path = os.path.join('plots_corr', f'scatter_{var1}_{var2}.png')
    plt.savefig(scatter_path)
    plt.close()
    st.image(scatter_path)

    # Calcular coeficiente de correlación de Pearson
    pearson_corr = df[[var1, var2]].corr(method='pearson').iloc[0, 1]
    st.write(f"Coeficiente de Correlación de Pearson entre {var1} y {var2}: {pearson_corr}")

    # Prueba de independencia de Chi-cuadrado
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    st.write(f"Prueba de Independencia de Chi-cuadrado entre {var1} y {var2}")
    st.write(f"Valor de p: {p_value}")

    # Resultado de la prueba de independencia
    alpha = 0.05  # Nivel de significancia
    if p_value < alpha:
        st.write("Resultado: Las variables son dependientes (rechazamos H0)")
    else:
        st.write("Resultado: Las variables son independientes (no rechazamos H0)")

    # Mostrar gráfico de regresión si la correlación es fuerte
    if abs(pearson_corr) > 0.7:
        st.write("La correlación es fuerte. Mostrando gráfico de regresión.")
        X = df[var1].values.reshape(-1, 1)
        y = df[var2].values
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        y_pred_linear = linear_model.predict(X)
        linear_r2 = r2_score(y, y_pred_linear)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue')
        plt.plot(X, y_pred_linear, color='red')
        plt.title(f"Regresión Lineal entre {var1} y {var2}")
        plt.xlabel(var1)
        plt.ylabel(var2)
        regression_path = os.path.join('plots_corr', f'regression_{var1}_{var2}.png')
        plt.savefig(regression_path)
        plt.close()
        st.image(regression_path)
        st.write(f"Ecuación del Modelo de Regresión Lineal: y = {linear_model.coef_[0]}*x + {linear_model.intercept_}")

# Modelo Logístico
elif selected_section == "Modelo Logístico":
    st.header("Modelo Logístico para la variable diagnosis")
    
    
    not_include = ['id', 'diagnosis2']

    st.write("Seleccione el nivel de significancia:")
    alpha = st.selectbox("Nivel de Significancia", [0.01, 0.05, 0.025, 0.1, 0.005])
    
    results = analyze_logistic_regression(data_path, not_include, 'model_plots', alpha)
    
    # Mostrar resumen del modelo
    st.subheader("Resumen del Modelo")
    st.table(results['model_summary'].tables[1])
    
    # Mostrar ecuación del modelo
    st.subheader("Ecuación del Modelo Logístico")
    coefs = results['model_summary'].tables[1].data
    equation = "logit(p) = " + " + ".join([f"{coef[1]}*{coef[0]}" for coef in coefs[1:]])
    st.write(f"Ecuación del Modelo: {equation}")
    
    # Mostrar estadísticas de validación
    st.subheader("Estadísticas de Validación")
    validation_stats = results['validation_stats']
    st.write(f"Media de los residuos: {validation_stats['mean_residuals']}")
    st.write(f"Suma de los residuos: {validation_stats['sum_residuals']}")
    st.write(f"Estadístico de la prueba KS: {validation_stats['ks_test_statistic']}")
    st.write(f"Valor p de la prueba KS: {validation_stats['ks_test_pvalue']}")
    st.write(f"Estadístico de Durbin-Watson: {validation_stats['durbin_watson_statistic']}")
    st.write(f"Valor p de Breusch-Pagan: {validation_stats['breusch_pagan_pvalue']}")
    st.write(f"Variables eliminadas: {', '.join(map(str, validation_stats['variables_removed']))}")
    
    # Mostrar gráficos generados
    st.subheader("Gráficos de Validación")
    st.image(validation_stats['histogram_plot'])
    st.image(validation_stats['qqplot'])
    
    # Mostrar matriz de confusión
    st.subheader("Matriz de Confusión")
    confusion_matrix = results['confusion_matrix']
    st.table(pd.DataFrame(confusion_matrix, index=["Actual Negativo", "Actual Positivo"], columns=["Predicho Negativo", "Predicho Positivo"]).style.hide(axis="index").hide(axis="columns"))
    
    # Interpretación de la matriz de confusión
    st.write("**Interpretación de la Matriz de Confusión:**")
    st.write(f"Verdaderos Positivos (TP): {confusion_matrix[1, 1]}")
    st.write(f"Verdaderos Negativos (TN): {confusion_matrix[0, 0]}")
    st.write(f"Falsos Positivos (FP): {confusion_matrix[0, 1]}")
    st.write(f"Falsos Negativos (FN): {confusion_matrix[1, 0]}")
    
    # Calcular y mostrar el porcentaje de éxito (accuracy)
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
    st.subheader("Porcentaje de Éxito del Modelo (Accuracy)")
    st.write(f"El porcentaje de éxito del modelo es: {accuracy * 100:.2f}%")


# Footer
st.sidebar.markdown("---")
st.sidebar.write("Proyecto realizado por Richard Alejandro Matos Arderí y Mauricio Sunde Jiménez")
