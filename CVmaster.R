#--- CVmaster function ---#


cross_validation <- function(classifier, training_features, training_labels, 
                             kfolds, loss_function=NA){
  # a matrix to store result of ith fold as validation set, accuracy, loss
  # Output: the last model and model metrics will be printed
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