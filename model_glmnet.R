IS_GLMNET <- TRUE
N_FOLDS <- 10
library(glmnet)
source('./code/data_processing.R')

glmnet_score <- rep(0,N_FOLDS)
glmnet_val_pred <- 0
glmnet_test_pred <- 0

for(i in 1:N_FOLDS){
  cat("Creating model ",i, " of ",N_FOLDS," models \n")
  
  st <- ceiling(nrow(train_data)/N_FOLDS)
  s <- (st*(i-1) + 1):min((st*i),nrow(train_data))
  train_train <- train_data[-s,]
  train_validation <- train_data[s,]
  train_train$target <- as.factor(train_train$target)
  
  x <- as.matrix(train_train[,-match(c("target"),colnames(train_train))])
  glmnet_model <- cv.glmnet(x,train_train$target,family="binomial")
  glmnet_model$lambda.min
  glmnet_pred <- predict(glmnet_model,
                         as.matrix(train_validation[,-match("target",colnames(train_validation))]),
                         s = "lambda.min",type="response")
  glmnet_pred <- pmax(pmin(glmnet_pred,1-1e-15),1e-15)
  glmnet_score[i] <- logLoss(train_validation$target,glmnet_pred)
  print(glmnet_score[i])
  
  if(i==1) {
    glmnet_val_pred <- glmnet_pred
  }
  else {
    glmnet_val_pred <- c(glmnet_val_pred,glmnet_pred)
  }
  
  glmnet_pred <- predict(glmnet_model, as.matrix(test_data), s = "lambda.min",type="response")
  glmnet_pred <- pmax(pmin(glmnet_pred,1-1e-15),1e-15)
  if(i==1) {
    glmnet_test_pred <- glmnet_pred
  }
  else {
    glmnet_test_pred <- glmnet_test_pred + glmnet_pred
  }
}
glmnet_test_pred <- glmnet_test_pred/N_FOLDS
write.csv(glmnet_val_pred,"glmnet_val_pred_v1.csv",row.names=FALSE)
write.csv(glmnet_test_pred,"glmnet_test_pred_v1.csv",row.names=FALSE)
#glmnet score - [1] 0.4856059 0.4908456 0.4798330 0.4766951 0.4808814 0.4759052 0.4957637 0.4796365 0.4774139 0.4899055