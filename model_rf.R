IS_RF <- TRUE
N_FOLDS <- 10
source('./code/data_processing.R')

rf_score <- rep(0,N_FOLDS)
rf_test_pred <- 0
rf_val_pred <- 0

cat("Random forest model for stacking\n")
for (i in 1:N_FOLDS){
  cat("Creating model ",i, " of ",N_FOLDS," models \n")
  
  st <- ceiling(nrow(train_data)/N_FOLDS)
  s <- (st*(i-1) + 1):min((st*i),nrow(train_data))
  train_train <- train_data[-s,]
  train_validation <- train_data[s,]
  train_train$target <- as.factor(train_train$target)
  rf_model <- ranger(target~., 
                     data=train_train,
                     num.trees=500,
                     write.forest = TRUE,
                     seed = 12,
                     mtry = 55,
                     probability = TRUE,
                     importance = "impurity",
                     replace = FALSE)
  
  cat("Create validation predictions\n")
  rf_predictions <- predict(rf_model,train_validation,type="response")
  rf_predictions <- rf_predictions$predictions
  rf_predictions <- rf_predictions[,2]
  rf_predictions <- pmax(pmin(rf_predictions,1-1e-15),1e-15)
  rf_score[i] <- logLoss(train_validation$target,rf_predictions)
  print(rf_score[i])
  
  if(i==1) {
    rf_val_pred <- rf_predictions
  }
  else {
    rf_val_pred <- c(rf_val_pred,rf_predictions)
  }
  
  cat("Create test predictions\n")
  rf_predictions <- predict(rf_model,test_data,type="response")
  rf_predictions <- rf_predictions$predictions
  rf_predictions <- rf_predictions[,2]
  rf_predictions <- pmax(pmin(rf_predictions,1-1e-15),1e-15)
  
  if(i==1) {
    rf_test_pred <- rf_predictions
  }
  else {
    rf_test_pred <- rf_test_pred+rf_predictions
  }
  gc(reset=TRUE)
}
rf_test_pred <- rf_test_pred/N_FOLDS

#65-0.4686909
#55-0.468484