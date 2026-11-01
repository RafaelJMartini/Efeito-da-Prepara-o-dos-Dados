################################################################################
# Laboratório 04 - Classificadores de Bayes (LDA, QDA e Naive Bayes)
#
# Objetivo : classificar grãos de arroz (Cammeo x Osmancik) a partir de
#            7 medidas morfológicas, comparando três classificadores
#            baseados no teorema de Bayes.
#
# Dataset  : Rice_Osmancik_Cammeo_Dataset.csv
################################################################################

# Configuração do diretório de Trabalho (troque pelo caminho no seu computador)
#setwd("")

# Remoção de qualquer variável do ambiente
rm(list=ls())

################################################################################
# 1. LEITURA DOS DADOS                                                         #  
################################################################################
# Leitura do dataset
dados = read.csv("Rice_Osmancik_Cammeo_Dataset.csv", sep = ";", dec=",")

# Estrutura do dataset
str(dados)

# Valores ausentes podem quebrar LDA/QDA/NB; é melhor saber antes de modelar
cat("Valores ausentes no dataset:", sum(is.na(dados)), "\n")

################################################################################
# 2. PREPARAÇÃO DOS DADOS                                                      #
################################################################################

################################################################################
# 2.1 PREPARAÇÃO DA ESTRUTURA                                                   

# Convertendo a variável alvo em factor
dados$CLASS  = factor(dados$CLASS)

# Verificação
str(dados) 

# Total de amostras por classe
table(dados$CLASS)

# Percentual de amostras por classe
table(dados$CLASS) / sum(table(dados$CLASS))

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

# Total de amostras por classe
table(rotulosTreinamento)

# Percentual de amostras por classe
table(rotulosTreinamento) / sum(table(rotulosTreinamento))

# Total de amostras por classe
table(rotulosTeste)

# Percentual de amostras por classe
table(rotulosTeste) / sum(table(rotulosTeste))

# Libera objetos que não serão mais usados
rm(num_vars, dados, indicesTreinamento)

################################################################################
# 2.3. REMOÇÃO DE OUTLIERS                                          

# Identifica linhas com pelo menos um outlier
outlier_linhas = apply(dadosTreinamento, 1, function(linha) { 
  any(sapply(1:length(linha), function(i) linha[i] %in% 
               boxplot.stats(dadosTreinamento[, i])$out))
})

# Remove as amostras com outliers
dadosTreinamento = dadosTreinamento[-outlier_linhas, ]
rotulosTreinamento = rotulosTreinamento[-outlier_linhas]

# Removendo as variáveis desnecessárias
rm(outlier_linhas)

################################################################################
# 2.4. NORMALIZAÇÃO DOS DADOS (Escore-Z)  

# Estima os parâmetros de normalização (média e desvio padrão)
normalizacaoParametros = preProcess(dadosTreinamento, 
                                    method = c("center", "scale"))

# Aplica a normalização Z-Score aos dados de treinamento e de teste
dadosTreinamento = predict(normalizacaoParametros, dadosTreinamento)
dadosTeste       = predict(normalizacaoParametros, dadosTeste)

# Exibe resumo estatístico dos dados normalizados
summary(dadosTreinamento)
summary(dadosTeste)

################################################################################
# 2.4. SELEÇÃO DAS CARACATERÍSTICAS
library(corrplot)

# Calcular a matriz de correlação apenas entre as variáveis numéricas
dadosCorrelacao = cor(dadosTreinamento)

# Visualizar a matriz de correlação de forma gráfica
corrplot.mixed(dadosCorrelacao, lower.col = "black")

# Identificar variáveis altamente correlacionadas
indicesCorrelacaoForte = findCorrelation(dadosCorrelacao, cutoff = 0.95, verbose = T)

# Exibir os nomes das variáveis correspondentes aos índices identificados
names(dadosTreinamento)[indicesCorrelacaoForte]

# Remover as variáveis fortemente correlacionadas
dadosTreinamento[,indicesCorrelacaoForte] = NULL
dadosTeste[,indicesCorrelacaoForte] = NULL

# Verificação
str(dadosTreinamento) 
str(dadosTeste) 

# Removendo as variáveis desnecessárias
rm(dadosCorrelacao)

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
# 3.2. CONHECENDO OS MODELOS

# ---- QDA ----
qda.modelo$finalModel                  # resumo do modelo
qda.modelo$finalModel$prior            # probabilidades a priori das classes
qda.modelo$finalModel$means            # média de cada variável por classe
qda.modelo$finalModel$scaing           # transformação utilizada para obter a inv(cov(x))

# ---- LDA ----
lda.modelo$finalModel                  # resumo do modelo
lda.modelo$finalModel$prior            # probabilidades a priori das classes
lda.modelo$finalModel$means            # média de cada variável por classe
lda.modelo$finalModel$scaling          # coeficientes do discriminante linear (LD1)

# Projeta os dados no discriminante linear (LD1)
ld1 = as.matrix(dadosTreinamento) %*% lda.modelo$finalModel$scaling
 
# Plota a distribuição de ld por classe
boxplot(ld1 ~ rotulosTreinamento,
        main = "Separação das Classes via LD1",
        col = c("lightblue", "lightgreen"))

# ---- Naive Bayes ----
nb.modelo$finalModel                   # estrutura geral
nb.modelo$finalModel$apriori           # probabilidades a priori das classes
nb.modelo$finalModel$tables            # média e desvio padrão por variável e classe

rm(ld1)

################################################################################
# 3.3. SALVAR MODELOS E PRÉ-PROCESSAMENTO

# Salva tudo o que é preciso para classificar dados novos: os modelos, os
# parâmetros de normalização e a lista de variáveis a eliminar
save(lda.modelo,qda.modelo,nb.modelo, 
     normalizacaoParametros, indicesCorrelacaoForte,
     file = "rice.modelos")

rm(normalizacaoParametros)

################################################################################
# 3.4. AVALIAR DADOS DE TREINAMENTO

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

res = confusionMatrix(qda.predicao,rotulosTreinamento)
resultado.treinamento[2,] = data.frame("QDA",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(nb.predicao,rotulosTreinamento)
resultado.treinamento[3,] = data.frame("Naive Bayes",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

ggplot(resultado.treinamento, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.875, 0.95) +
  ggtitle("Desempenho do Treinamento") 

# Removendo as variáveis desnecessárias
rm(lda.predicao,qda.predicao,nb.predicao,res,resultado.treinamento,
   dadosTreinamento,rotulosTreinamento)

################################################################################
# 4. TESTE                                                                     #
################################################################################

################################################################################
# 4.1. AVALIAÇÃO DE DESEMPENHO

# Aplicar o modelo aos dados de teste (retorna as classes preditas) 
lda.predicao = predict(lda.modelo,dadosTeste)
qda.predicao = predict(qda.modelo,dadosTeste)
nb.predicao  = predict(nb.modelo,dadosTeste)

# Cria um data frame para armazenar as medidas de avaliação
resultado.teste = data.frame(Classificador=character(), Acuracia=double(),
                             AcuraciaInferior=double(), AcuraciaSuperior=double(),
                             stringsAsFactors = FALSE)

# Calcular as medidas de avaliação
res = confusionMatrix(lda.predicao,rotulosTeste)
resultado.teste[1,] = data.frame("LDA",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(qda.predicao,rotulosTeste)
resultado.teste[2,] = data.frame("QDA",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(nb.predicao,rotulosTeste)
resultado.teste[3,] = data.frame("Naive Bayes",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

ggplot(resultado.teste, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.875, 0.95) +
  ggtitle("Desempenho do Teste") 

rm(lda.predicao,qda.predicao,nb.predicao,res,resultado.teste)
################################################################################
# 4.2. AVALIAÇÃO DE DESEMPENHO - ROC

library(ROCR)
# Estimar a probabilidade para cada amostra por classe
lda.probabilidades = predict(lda.modelo,dadosTeste,type="prob")

# Estima as métricas de desempenho com base nas probabilidade da classe positiva
# Classe positiva = Cammeo (segunda)
lda.predicao = prediction(lda.probabilidades[,1],rotulosTeste, 
                          label.ordering = c("Osmancik", "Cammeo"))

# Calcular as métricas de desempenho necessárias para a curva ROC
roc = performance(lda.predicao,"tpr","fpr")
# AUC: o valor fica no slot @y.values (uma lista com um elemento)
lda.auc = performance(lda.predicao, "auc")@y.values[[1]]

# Plotar a curva ROC
plot(roc,col='blue',main='ROC - Classe Postiva: Cammeo')

qda.probabilidades = predict(qda.modelo,dadosTeste,type="prob")
qda.predicao = prediction(qda.probabilidades[,1],rotulosTeste, 
                          label.ordering = c("Osmancik", "Cammeo"))
roc = performance(qda.predicao,"tpr","fpr")
qda.auc = performance(qda.predicao, "auc")@y.values[[1]]

plot(roc,col='red',add=TRUE)

nb.probabilidades = predict(nb.modelo,dadosTeste,type="prob")
nb.predicao = prediction(nb.probabilidades[,1],rotulosTeste, 
                         label.ordering = c("Osmancik", "Cammeo"))
roc = performance(nb.predicao,"tpr","fpr")
nb.auc  = performance(nb.predicao,  "auc")@y.values[[1]]
plot(roc,col='green',add=TRUE)

legend("center",
      legend = c(sprintf("LDA (AUC = %.3f)", lda.auc),
                 sprintf("QDA (AUC = %.3f)", qda.auc),
                 sprintf("Naive Bayes (AUC = %.3f)", nb.auc)),
       lty=1,col=c('blue','red','green'))

# Adicionar linha diagonal de referência (Aposta Aleatória)
abline(a = 0, b = 1, lty = 2, col = "gray")