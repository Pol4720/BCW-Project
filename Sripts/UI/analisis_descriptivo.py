import streamlit as st
import sys

sys.path.append('c:/Users/Richard/Desktop/Estadística/Proyecto/BCW-Project/Sripts/Python')

from stats import create_analysis_plots
from variables_vs_diagnosis import plot_variable_vs_diagnosis

def mostrar_analisis_descriptivo(df, data_path, variable_info, variables):
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
