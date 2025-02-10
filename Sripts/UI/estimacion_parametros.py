import streamlit as st
import numpy as np

def mostrar_estimacion_parametros(df, variables):
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
