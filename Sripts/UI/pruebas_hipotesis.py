import streamlit as st
import numpy as np
import scipy.stats as stats

def mostrar_pruebas_hipotesis(df, variables):
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


