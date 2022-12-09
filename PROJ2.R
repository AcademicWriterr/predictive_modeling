#install.packages("ggplot2")
#install.packages("tidyverse")
#install.packages("e1071")
#install.packages("caTools")
#install.packages("class")
#install.packages("party")
library(tidyverse)
library(GGally)
library(gridExtra)
library(caret)
library(MASS)
set.seed(9527)

load_data <- function(filename) {
  # a function load data to a dataframe from a text file
  df <- read.table(filename, 
                   col.names = c('x', 'y', 'label', 'NDAI', 'SD', 'CORR', 'DF', 
                  'CF', 'BF', 'AF', 'AN')) %>% select(c('x', 'y', 'NDAI', 'SD', 'CORR', 'DF', 'CF', 'BF', 'AF', 'AN', 'label'))
  return(df)
  }

label_ratio_check <- function(data) {
  data %>% group_by(label) %>% summarise(
    count=n(),
    label_ratio = count / nrow(data)
  )
}

get_summary <- function(filename){
  # a function to summarize the image data
  # Input: file path
  # Output: summary for each label count and ratio with a plot too
 df <- load_data(filename)
 
 # a temporary df for summarized count of each label and percentage of the label
 temp <- label_ratio_check(df)
 
 print('count of each label of its percentage to total number of labels')
 print(temp)
df$label <- as.character(df$label)
 df %>% ggplot(aes(x=x,y=y, color=label)) + geom_line() + 
   ggtitle('x, y corrdinates based on expert labels') + 
   xlab('x axis') + ylab('y axis')
}

get_summary('./image_data/imagem1.txt')
get_summary('./image_data/imagem2.txt')
get_summary('./image_data/imagem3.txt')

###!------------- Q1 Data Collection and Exploration ---------------!###
df <- load_data('./image_data/imagem1.txt')
df$label <- as.factor(df$label)
# pairwise plots
ggpairs(df)

plot_distribution <- function(feature, df){
  # function to show density plot and box plot of a feature
  feature <- sym(feature)  
  f1<- ggplot(data=df,aes(x=!!feature, fill=label) ) + geom_density(alpha=.4) + 
    ggtitle(paste('Density plot of', feature, 'according to label'))
  
  f2<- ggplot(data=df,aes(y=!!feature, fill=label) ) + geom_boxplot(alpha=.4) + 
    ggtitle(paste('Box plot of', feature, 'according to label'))
  grid.arrange(f1, f2, ncol=2)
  
}

plot_distribution('NDAI', df)
plot_distribution('SD', df)
plot_distribution('CORR', df)

###!------------- Q2 Preparation ---------------!###

# --- 2 a split data ---#
# merge the three images data
df <- load_data('./image_data/imagem1.txt')
df2 <- load_data('./image_data/imagem2.txt')
df3 <- load_data('./image_data/imagem3.txt')
df <- merge(df, df2, all = TRUE) %>% merge(df3, all = TRUE)
# check label ratio
df %>% label_ratio_check()

# remove unlabeled data and apply standard normalized on the predictors
df <- df %>% filter(label!=0 )
df[, -ncol(df)] <- df[, -ncol(df)] %>% scale
df$label <- as.factor(df$label)



#--- Split the data according to label distribution --- #

# shuffle the whole dataset
df <- df[sample(1:nrow(df)), ]
training_index <- createDataPartition(df$label, p = 0.8, list=FALSE, times=1)
training_set <- df[training_index, ]
temp <- df[-training_index, ]
valid_index <- sample(1:floor(nrow(temp)/2))
valid_set <- temp[valid_index, ]
testing_set <- temp[-valid_index,]

#--- Split the data with strata 
stratified <- df %>% group_by(label) %>% sample_n(size = 10000) %>% as.data.frame()
stratified <- stratified[sample(1:nrow(stratified)), ]
training_index <- sample(1:floor(nrow(stratified) * 0.8))
strata_training_set <- stratified[training_index, ]
temp <- stratified[-training_index, ]
valid_index <- sample(1:floor(nrow(temp)/2))
strata_valid_set <- temp[valid_index, ]
strata_testing_set <- temp[-valid_index,]

#--- 2 b trivial classifier --- #
library(e1071)
library(caTools)
library(class)

# apply knn model and check accuracy
predictions <- knn(training_set, testing_set, cl=training_set$label, k=5)
summary(predictions)
accuracy <- mean(testing_set$label == predictions)
print(paste('the accuracy of KNN classifier with k=5 is', accuracy))


#--- 2 c suggest three of the best features ---#
library(party)
cf1 <- cforest(label~ ., data = training_set, 
               control = cforest_unbiased(mtry=2, ntree = 5))
importance_df <- varimp(cf1)
importance_df <- tibble(feature=names(importance_df), 
                        importance=importance_df) %>% 
  arrange(desc(importance))
ggplot(data=importance_df, aes(reorder(feature, -importance), 
      weight=importance)) + geom_bar() + 
      labs(x='feature', y='importance', title='barplot of feature importance')



#--- CVmaster function ---#
cross_validation <- function(classifier, training_features, training_labels, 
                             kfolds, loss_function=NA){
  # a matrix to store result of ith fold as validation set, accuracy, loss
  # Output: the last model
  result <- matrix(0, nrow = kfolds, ncol = 3)
  folds <- createFolds(training_labels, k=kfolds)
  
  for (i in 1: kfolds) {
      valid_index = folds[[i]]
      valid_feature <- training_features[valid_index, ]
      train_feature <- training_features[-valid_index, ]
      valid_label <- training_labels[valid_index]
      train_label <- training_labels[-valid_index]
      df <- cbind(train_feature, train_label)
      model <- classifier(train_label ~., data=df)
      temp <- predict(model,valid_feature)
      temp <- loss_function(temp, valid_label)
      label <- temp[[1]]
      loss <- temp[[2]]
      accuracy <- mean(label == valid_label)
      # ***loss may change, it takes predicted labels and validation labels
      result[i, ] <- c(i, accuracy, loss)
      print(paste(i, accuracy, loss))
      print(paste('validation fold', i, 'accuracy:', accuracy, 'loss', loss))
    }
  
    accuracy2 <- mean(result[, 2])
    loss2 <- mean(result[, 3])
    print('---summary---')
    print(paste('The average accuracy over the', i, 'folds',  'is', accuracy2))
    print(paste('The average loss over the', i, 'folds', 'is', loss2))
  return(model)
}



###!------------- Q3 Modeling ---------------!###

##! A. try 4 classification methods
# logistic regression
logistic_loss <- function(predicted, original) {
  # function to produce predicted label and mean squared error
  # Input: response from predict function
  # Output: predicted label and mean squared error
  label <- ifelse(predicted > 0.5, 1, 2)
  answer <- list(label, sum(original-predicted) ^ 2 / length(original))
  return(answer)
}

# prepare data for split according to label ratio
training_set <- merge(training_set, valid_set, all = TRUE)
training_feature1 <- training_set[, -ncol(training_set)]
training_label1 <- as.numeric(training_set[, ncol(training_set)])

# prepare data for split with stratified method
training_set <- merge(strata_training_set, strata_valid_set, all = TRUE)
training_feature2 <- strata_training_set[, -ncol(strata_training_set)]
training_label2 <- as.numeric(strata_training_set[, ncol(strata_training_set)])

# merge validation set and testing set


# on the first split method
logistic_model <- cross_validation(glm, training_feature1, training_label1, 
                                   10, logistic_loss)
# on the second split method
logistic_model <- cross_validation(glm, training_feature2, training_label2, 
                                   10, logistic_loss)

# LDA 
lda_loss <- function(predicted, original) {
  # function to produce predicted label and mean squared error
  # Input: response from predict function
  # Output: predicted label and mean squared error
  label <- as.numeric(predicted$class)
  answer <- list(label, sum(original-label) ^ 2 / length(original))
  return(answer)
}
lda_model <- cross_validation(lda, training_feature, training_label, 10, lda_loss)
# on the first split method
lda_model <- cross_validation(lda, training_feature1, training_label1, 
                                   10, lda_loss)
# on the second split method
lda_model <- cross_validation(lda, training_feature2, training_label2, 
                                   10, lda_loss)

# QDA
qda_loss <- function(predicted, original) {
  # function to produce predicted label and mean squared error
  # Input: response from predict function
  # Output: predicted label and mean squared error
  label <- as.numeric(predicted$class)
  answer <- list(label, sum(original-label) ^ 2 / length(original))
  return(answer)
}
# on the first split method
qda_model <- cross_validation(qda, training_feature1, training_label1, 
                              10, qda_loss)
# on the second split method
qda_model <- cross_validation(qda, training_feature2, training_label2, 
                              10, qda_loss)
# naive Bayes
bay_loss <- function(predicted, original) {
  # function to produce predicted label and mean squared error
  # Input: response from predict function
  # Output: predicted label and mean squared error
  label <- as.numeric(predicted)
  answer <- list(label, sum(original-label) ^ 2 / length(original))
  return(answer)
}

# on the first split method
bay_model <- cross_validation(naiveBayes, training_feature1, training_label1, 
                              10, bay_loss)
# on the second split method
bay_model <- cross_validation(naiveBayes, training_feature2, training_label2, 
                              10, bay_loss)


library(pROC)
testing_feature <- strata_testing_set[, -ncol(strata_testing_set)]
testing_label <- as.numeric(strata_testing_set[, ncol(strata_testing_set)])

# logistic regression ROC curve
predictions <- ifelse(predict(logistic_model, testing_feature) > 0.5, 1, -1)
roc_score=roc(testing_label, predictions) #AUC score
plot(roc_score, print.thres=TRUE, main ="ROC curve -- Logistic Regression ")
roc_score

# LDA
predictions <- predict(lda_model, testing_feature)
roc_score=roc(testing_label, as.numeric(predictions$class)) #AUC score
plot(roc_score, print.thres=TRUE, main ="ROC curve -- LDA ")
roc_score

# QDA
predictions <- predict(qda_model, testing_feature)
roc_score=roc(testing_label, as.numeric(predictions$class)) #AUC score
plot(roc_score, print.thres=TRUE, main ="ROC curve -- QDA ")
roc_score

# Bays
predictions <- predict(bay_model, testing_feature)
roc_score=roc(testing_label, as.numeric(predictions)) #AUC score
plot(roc_score, print.thres=TRUE, main ="ROC curve -- Naive Bayes ")
roc_score

###!------------- Q4 Diagnostics---------------!###

## Analysis on the best model
predictions <- predict(lda_model, testing_feature)
# histogram
ldahist(data=predictions$x, g=testing_label) + title('Histogram of LDA predictions')

# confusion matrix and accuracy
confusionMatrix(as.factor(testing_label), predictions$class )
sensitivity(as.factor(testing_label), predictions$class)
specificity(as.factor(testing_label), predictions$class)

# Classifications analysis
misclassified_index <- predictions$class != testing_label
misclassified_feature <- testing_feature[misclassified_index, ]
misclassified_label<- testing_label[predictions$class != testing_label]
#misclassified_feature$predicted <- predictions$class[misclassified_index]
misclassified_feature$label <- as.factor(testing_label[misclassified_index])
temp <- melt(misclassified_feature, id ='label')
ggplot(temp, aes(x = variable, y = value, color=label)) + geom_boxplot()

misclassified_feature <- testing_feature[-misclassified_index, ]
misclassified_label<- testing_label[predictions$class == testing_label]
# misclassified_feature$predicted <- predictions$class[misclassified_index]
misclassified_feature$label <- as.factor(testing_label[-misclassified_index])
temp <- melt(misclassified_feature, id ='label')
ggplot(temp, aes(x = variable, y = value, color=label)) + geom_boxplot()

