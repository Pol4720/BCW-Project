import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python')


from corr import create_correlation_analysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency

def mostrar_correlacion_independencia(df, data_path, variables):
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
