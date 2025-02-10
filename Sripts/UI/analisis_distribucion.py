import streamlit as st

import sys
sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python/Test')


from shapiro import test_normalidad_shapiro_wilk
from normalTest import test_normalidad_anderson_darling
from kolmogorov import test_normalidad_kolmogorov_smirnov

def mostrar_analisis_distribucion(data_path, variables):
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
