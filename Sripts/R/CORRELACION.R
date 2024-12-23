rm(list=ls())

data("data")

library(corrplot)
cor_matrix_Pearson <- cor(data)
cor_matrix_Spearman <- cor(data,method = "spearman")
corrplot(cor_matrix_Pearson,method = "circle",type ="upper", tl.col = "black", tl.srt = 55)

corrplot(cor_matrix_Spearman,method = "square",type ="upper", tl.col = "black", tl.srt = 55)

var_combinations <- combn(names(data), 2)

for (i in 1:ncol(var_combinations)) {
  var1 <- var_combinations[1, i]
  var2 <- var_combinations[2, i]
  
  plot(data[[var1]], data[[var2]])
}

cor(data)
corrplot(cor_matrix_Spearman,type ="upper")
cor_matrix_Spearman

plot(data[[pop15]],data[[pop75]])
