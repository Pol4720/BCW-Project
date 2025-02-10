import numpy as np
import streamlit as st
import pandas as pd
import os
import sys
import json

# Agregar la ruta absoluta de los módulos
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python/Test')
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python')
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/UI')

from introduccion import mostrar_introduccion
from analisis_descriptivo import mostrar_analisis_descriptivo
from analisis_distribucion import mostrar_analisis_distribucion
from estimacion_parametros import mostrar_estimacion_parametros
from pruebas_hipotesis import mostrar_pruebas_hipotesis
from correlacion_independencia import mostrar_correlacion_independencia
from modelo_logistico import mostrar_modelo_logistico

from scipy.stats import  f

# Cargar la información descriptiva de las variables
with open('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/UI/variable_info.json', 'r', encoding='utf-8') as f:
    variable_info = json.load(f)

variables = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'symmetry_worst']  

# Configuración de la página
st.set_page_config(page_title="BCW", layout="wide")

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
    mostrar_introduccion()

# Análisis Descriptivo de los datos
elif selected_section == "Análisis Descriptivo de los datos":
    mostrar_analisis_descriptivo(df, data_path, variable_info, variables)

# Análisis de la distribución
elif selected_section == "Análisis de la distribución":
    mostrar_analisis_distribucion(data_path, variables)

# Estimación de parámetros
elif selected_section == "Estimación de parámetros":
    mostrar_estimacion_parametros(df, variables)

# Pruebas de Hipótesis
elif selected_section == "Pruebas de Hipótesis":
    mostrar_pruebas_hipotesis(df, variables)

# Correlación e Independencia
elif selected_section == "Correlación e Independencia":
    mostrar_correlacion_independencia(df, data_path, variables)

# Modelo Logístico
elif selected_section == "Modelo Logístico":
    mostrar_modelo_logistico(df, data_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Proyecto realizado por Richard Alejandro Matos Arderí y Mauricio Sunde Jiménez")
