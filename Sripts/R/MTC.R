library(MASS)

# Cargar el dataset
data(data.csv)

# Función para calcular la moda
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Función para graficar la distribución de probabilidad y las distribuciones de variables aleatorias conocidas
plot_distribution <- function(data, var_name) {
  hist(data, freq = FALSE, main = paste("Distribución de", var_name), xlab = var_name, col = "lightblue")
  curve(dnorm(x, mean = mean(data), sd = sd(data)), col = "red", lwd = 2, add = TRUE)
  curve(dexp(x, rate = 1/mean(data)), col = "green", lwd = 2, add = TRUE)
  curve(dgamma(x, shape = 2, rate = 2/mean(data)), col = "blue", lwd = 2, add = TRUE)
  curve(dunif(x, min = min(data), max = max(data)), col = "orange", lwd = 2, add = TRUE)
  curve(dchisq(x, df = 3), col = "purple", lwd = 2, add = TRUE)
  lines(density(data), col = "black", lwd = 2, add = TRUE)
  
  legend("topright", legend = c("Normal", "Exponencial", "Gamma", "Uniforme", "Chi-cuadrado",var_name), fill = c("red", "green", "blue", "orange", "purple","black"))
}

# Iterar sobre cada variable del dataset
for (var in names(data.csv)) {
  data <- data.csv[[var]]
  
  # Medidas de tendencia central
  central_measures <- c(mean = mean(data), mode = get_mode(data), median = median(data),
                        Q1 = quantile(data, 0.25), Q3 = quantile(data, 0.75))
  
  # Medidas de variabilidad
  variability_measures <- c(max = max(data), min = min(data), range = max(data) - min(data),
                            variance = var(data), sd = sd(data), cv = sd(data) / mean(data))
  
  # Imprimir las medidas
  cat("Variable:", var, "\n")
  cat("Medidas de tendencia central:\n")
  print(central_measures)
  cat("Medidas de variabilidad:\n")
  print(variability_measures)
  
  # Gráfico de cajas y bigotes
  boxplot(data, main = paste("Boxplot de", var), col = "beige", border = "brown")
  
  # Histograma
  hist(data, main = paste("Histograma de", var), xlab = var, col = "blue4")
  
  # Gráfico poligonal
  plot(density(data), main = paste("Gráfico poligonal de", var), col = "purple")
  
  # Gráfico de distribución de probabilidad y distribuciones conocidas
  plot_distribution(data, var)
}
