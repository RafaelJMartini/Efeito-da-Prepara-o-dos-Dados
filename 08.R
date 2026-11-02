################################################################################
# 08 - Nenhum pré-processamento aplicado
################################################################################

# Configuração do diretório de Trabalho (troque pelo caminho no seu computador)
setwd("./")

# Remoção de qualquer variável do ambiente
rm(list=ls())

contar_parametros <- function(lda.modelo, qda.modelo, nb.modelo) {
  K <- length(lda.modelo$finalModel$prior)      # número de classes
  p <- ncol(lda.modelo$finalModel$means)        # número de variáveis
  
  n_lda <- K * p + p * (p + 1) / 2 + (K - 1)
  n_qda <- K * p + K * p * (p + 1) / 2 + (K - 1)
  n_nb  <- sum(sapply(nb.modelo$finalModel$tables, length)) + (K - 1)
  
  c(LDA = n_lda, QDA = n_qda, NaiveBayes = n_nb)
}

################################################################################
# 1. LEITURA DOS DADOS                                                         #  
################################################################################
# Leitura do dataset
dados = read.csv("Rice_Osmancik_Cammeo_Dataset.csv", sep = ";", dec=",")

################################################################################
# 2. PREPARAÇÃO DOS DADOS                                                      #
################################################################################

################################################################################
# 2.1 PREPARAÇÃO DA ESTRUTURA                                                   

# Convertendo a variável alvo em factor
dados$CLASS  = factor(dados$CLASS)

################################################################################
# 2.2 DIVISÃO DO CONJUNTO DE DADOS EM TREINAMENTO E TESTE                                              

library(caret)

# Seleciona apenas as variáveis numéricas
num_vars = names(dados)[sapply(dados, is.numeric)]

# Garantindo repetibilidade
set.seed(1)

# Realizando seleção holdout estratificada
indicesTreinamento = createDataPartition(dados$CLASS, p = 2/3, list = FALSE)

# Separando os dados e respectivo rótulos em treinamento e teste 
dadosTreinamento   = dados[indicesTreinamento,num_vars]
rotulosTreinamento = dados$CLASS[indicesTreinamento]
dadosTeste         = dados[-indicesTreinamento,num_vars]
rotulosTeste       = dados$CLASS[-indicesTreinamento]

# Libera objetos que não serão mais usados
rm(num_vars, dados, indicesTreinamento)

cat("Variáveis (treino):", ncol(dadosTreinamento), "\n")
cat("Variáveis (teste): ", ncol(dadosTeste), "\n")

################################################################################
# 3. TREINAMENTO                                                               #
################################################################################

################################################################################
# 3.1. ESTIMAÇÃO DOS MODELOS

# method = "none": ajusta cada modelo uma vez, sem validação cruzada.
# classProbs = TRUE: necessário para obter probabilidades (curva ROC/AUC).
paramsTreinamento = trainControl(method = "none", 
                                 classProbs  = TRUE)

# QDA: gaussiana por classe com covariância PRÓPRIA (fronteira quadrática)
qda.modelo = train(dadosTreinamento,
                   rotulosTreinamento,
                   trControl = paramsTreinamento,
                   method = "qda")

# LDA: gaussiana por classe com covariância COMUM (fronteira linear)
lda.modelo = train(dadosTreinamento,
                   rotulosTreinamento,
                   trControl = paramsTreinamento,
                   method = "lda")

# Naive Bayes: assume variáveis independentes dada a classe.
#   usekernel = FALSE -> densidade gaussiana (TRUE usaria estimativa por kernel)
#   adjust = 1        -> largura de banda do kernel (ignorado se usekernel = FALSE)
#   fL = 0            -> sem suavização de Laplace (relevante só para categóricas)
nb.modelo = train(dadosTreinamento,
                  rotulosTreinamento,
                  trControl = paramsTreinamento,
                  tuneGrid  = data.frame(usekernel=FALSE, adjust=1, fL=0),
                  method = "nb")

# Removendo as variáveis desnecessárias
rm(paramsTreinamento)

################################################################################
# 3.2. AVALIAR DADOS DE TREINAMENTO

# Predição das classes
lda.predicao = predict(lda.modelo,dadosTreinamento)
qda.predicao = predict(qda.modelo,dadosTreinamento)
nb.predicao  = predict(nb.modelo,dadosTreinamento)

# Cria um data frame para armazenar as medidas de avaliação
resultado.treinamento = data.frame(Classificador=character(), Acuracia=double(),
                                   AcuraciaInferior=double(), AcuraciaSuperior=double(),
                                   stringsAsFactors = FALSE)

# Calcular as medidas de avaliação
res = confusionMatrix(lda.predicao,rotulosTreinamento)
resultado.treinamento[1,] = data.frame("LDA",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])
print("Treino:")
cat("LDA:", res$overall["Accuracy"], "\n")
res = confusionMatrix(qda.predicao,rotulosTreinamento)
resultado.treinamento[2,] = data.frame("QDA",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])
cat("QDA:", res$overall["Accuracy"], "\n")

res = confusionMatrix(nb.predicao,rotulosTreinamento)
resultado.treinamento[3,] = data.frame("Naive Bayes",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])
cat("NB:", res$overall["Accuracy"], "\n")

ggplot(resultado.treinamento, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.875, 0.95) +
  ggtitle("Desempenho do Treinamento") 

# Removendo as variáveis desnecessárias
rm(lda.predicao,qda.predicao,nb.predicao,res,resultado.treinamento,
   dadosTreinamento,rotulosTreinamento)


np <- contar_parametros(lda.modelo, qda.modelo, nb.modelo)
cat("Parâmetros -> LDA:", np["LDA"],
    "| QDA:", np["QDA"],
    "| NB:", np["NaiveBayes"], "\n")

################################################################################
# 4. TESTE                                                                     #
################################################################################

################################################################################
# 4.1. AVALIAÇÃO DE DESEMPENHO

# Aplicar o modelo aos dados de teste (retorna as classes preditas) 
lda.predicao = predict(lda.modelo,dadosTeste)
qda.predicao = predict(qda.modelo,dadosTeste)
nb.predicao  = predict(nb.modelo,dadosTeste)

print("Teste:")
# Cria um data frame para armazenar as medidas de avaliação
resultado.teste = data.frame(Classificador=character(), Acuracia=double(),
                             AcuraciaInferior=double(), AcuraciaSuperior=double(),
                             stringsAsFactors = FALSE)
# Calcular as medidas de avaliação
res = confusionMatrix(lda.predicao,rotulosTeste)
resultado.teste[1,] = data.frame("LDA",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])
cat(sprintf("LDA: %.4f | IC 95%%: [%.4f, %.4f]\n",
            res$overall["Accuracy"],
            res$overall["AccuracyLower"],
            res$overall["AccuracyUpper"]))

res = confusionMatrix(qda.predicao,rotulosTeste)
resultado.teste[2,] = data.frame("QDA",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])
cat(sprintf("QDA: %.4f | IC 95%%: [%.4f, %.4f]\n",
            res$overall["Accuracy"],
            res$overall["AccuracyLower"],
            res$overall["AccuracyUpper"]))

res = confusionMatrix(nb.predicao,rotulosTeste)
resultado.teste[3,] = data.frame("Naive Bayes",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])
cat(sprintf("NB: %.4f | IC 95%%: [%.4f, %.4f]\n",
            res$overall["Accuracy"],
            res$overall["AccuracyLower"],
            res$overall["AccuracyUpper"]))

ggplot(resultado.teste, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.875, 0.95) +
  ggtitle("Desempenho do Teste") 

rm(lda.predicao,qda.predicao,nb.predicao,res,resultado.teste)