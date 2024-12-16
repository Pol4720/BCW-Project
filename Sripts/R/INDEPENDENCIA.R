library(MASS)

# Cargar el dataset
data(ALFA)

# Obtener todas las combinaciones posibles de variables
var_combinations <- combn(names(ALFA), 2)

# Realizar el análisis de chi cuadrado para cada combinación de variables
for (i in 1:ncol(var_combinations)) {
  var1 <- var_combinations[1, i]
  var2 <- var_combinations[2, i]
  
  # Crear una tabla de contingencia entre las dos variables
  contingency_table <- table(ALFA[[var1]], ALFA[[var2]])
  
  # Realizar la prueba chi cuadrado de independencia
  chi_square_test <- chisq.test(contingency_table)
  
  # Obtener los resultados numéricos de la prueba chi cuadrado
  chi_square_value <- chi_square_test$statistic
  p_value <- chi_square_test$p.value
  
  
  
  # Imprimir los resultados
  cat("Variables:", var1, "y", var2, "\n")
  cat("Valor de chi cuadrado:", chi_square_value, "\n")
  cat("Probabilidad de obtener un valor mayor o igual (p-valor): ", p_value, "\n")
  
  # Determinar si las variables son independientes o dependientes
  if (p_value < 0.05) {
    cat("Las variables", var1, "y", var2, "son dependientes\n\n")
  } else {
    cat("Las variables", var1, "y", var2, "son independientes\n\n")
  }
}



# Evaluar si existe una correlación
correlation_evaluation <- function(correlation) {
  if (abs(correlation) >= 0.7) {
    return("Existe una correlación fuerte.")
  } else if (abs(correlation) >= 0.4) {
    return("Existe una correlación moderada.")
  } else {
    return("No existe una correlación significativa.")
  }
}










