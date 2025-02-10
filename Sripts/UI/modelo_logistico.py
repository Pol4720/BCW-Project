import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python')
from app import analyze_logistic_regression

def mostrar_modelo_logistico(df, data_path):
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
