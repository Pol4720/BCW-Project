rm(list=ls())

data("ALFA")

library(corrplot)
cor_matrix_Pearson <- cor(ALFA)
cor_matrix_Spearman <- cor(ALFA,method = "spearman")
corrplot(cor_matrix_Pearson,method = "circle",type ="upper", tl.col = "black", tl.srt = 55)

corrplot(cor_matrix_Spearman,method = "square",type ="upper", tl.col = "black", tl.srt = 55)

var_combinations <- combn(names(ALFA), 2)

for (i in 1:ncol(var_combinations)) {
  var1 <- var_combinations[1, i]
  var2 <- var_combinations[2, i]
  
  plot(ALFA[[var1]], ALFA[[var2]])
}

cor(ALFA)
corrplot(cor_matrix_Spearman,type ="upper")
cor_matrix_Spearman

plot(ALFA[[pop15]],ALFA[[pop75]])
