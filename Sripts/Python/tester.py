import numpy as np
import pandas as pd
# Coeficientes del modelo
coef = {
    'const': -75.407292,
    'radius_mean': 1.761751,
    'smoothness_mean': 134.234402,
    'compactness_mean': -62.300596,
    'concavity_mean': 64.440480,
    'radius_se': 24.795382,
    'texture_se': -3.116882,
    'smoothness_se': 404.965775,
    'fractal_dimension_se': -2122.164836,
    'texture_worst': 0.560933,
    'fractal_dimension_worst': 277.330243
}

def AutomaticTester():
    data_path = "/Users/mauriciosundejimenez/Downloads/ProyectoEstadistica/BCW-Project/Dataset/data.csv"
    df = pd.read_csv(data_path)
    # Verificar el modelo para todas las observaciones del dataset
    for index, row in df.iterrows():
        input_data = {key: row[key] for key in coef if key != 'const'}
        diagnosis, prob = logistic_regression_predict(coef, input_data)
        if diagnosis != row['diagnosis']:
            print(f"El modelo no se cumple para la observación con ID {row['id']} (Diagnóstico real: {row['diagnosis']}, Diagnóstico predicho: {diagnosis})")
        # else:
        #     print("El modelo se cumple para la observacion con ID {row['id']}")

def logistic_regression_predict(coef, input_data):
    # Calcular el valor de z
    z = coef['const']
    for key in input_data:
        z += coef[key] * input_data[key]

    # Calcular la probabilidad usando la función sigmoide
    prob = 1 / (1 + np.exp(-z))

    # Determinar el diagnóstico
    diagnosis = 1 if prob >= 0.5 else 0
    return diagnosis, prob

def main():
    # Solicitar valores de las variables al usuario
    AutomaticTester()
    input_data = {}

    for key in coef:
        if key != 'const':
            value = float(input(f"Ingrese el valor de {key}: "))
            input_data[key] = value

    # Realizar la predicción
    diagnosis, prob = logistic_regression_predict(coef, input_data)

    # Mostrar el resultado
    diagnosis_str = "maligno" if diagnosis == 1 else "benigno"
    print(f"El diagnóstico es {diagnosis_str} con una probabilidad de {prob:.4f}")

if __name__ == "__main__":
    main()