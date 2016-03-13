setwd("~/Documents/Kaggle/kaggle_bnp")
IS_RF <- FALSE
IS_GLMNET <- FALSE
N_FOLD <- 10
source('./code/data_processing.R')

CREATE_SUB <- TRUE
VISUALIZE <- FALSE

rf_val_pred <- read_csv('rf_val_pred_v2.csv')
rf_val_pred <- rf_val_pred$x
rf_test_pred <- read_csv('rf_test_pred_v2.csv')
rf_test_pred <- rf_test_pred$x

et_val_pred <- read_csv('./code/train_predictions_etc_v1.csv')
et_val_pred <- et_val_pred$PredictedProb
et_test_pred <- read_csv('./code/test_predictions_etc_v1.csv')
et_test_pred <- et_test_pred$PredictedProb

# glmnet_val_pred <- read_csv('glmnet_val_pred_v1.csv')
# glmnet_val_pred <- glmnet_val_pred$x
# glmnet_test_pred <- read_csv('glmnet_test_pred_v1.csv')
# glmnet_test_pred <- glmnet_test_pred[,1]

cat("Get RF values for stacking\n")
train_data$rf_pred <- rf_val_pred
test_data$rf_pred <- rf_test_pred
train_data$et_pred <- et_val_pred
test_data$et_pred <- et_test_pred

# cat("Get GLMNET values for stacking\n")
# train_data$glmnet_pred <- glmnet_val_pred
# test_data$glmnet_pred <- glmnet_test_pred

start_time <- Sys.time()
score <- rep(0,N_FOLD)
for (i in 1:N_FOLD) {
  cat("Creating model ",i, " of 10\n")
  cat("Sample Data\n")
  
  #Random
  set.seed(1234)
#   s <- sample(0.80*nrow(train_data))
#   train_train <- train_data[s,]
#   train_validation <- train_data[-s,]
#   rm(s)
  
  st <- ceiling(nrow(train_data)/N_FOLD)
  s <- (st*(i-1) + 1):min((st*i),nrow(train_data))
  train_train <- train_data[-s,]
  train_validation <- train_data[s,]
  
  cat("Create xgboost model\n")
  dtrain<-xgb.DMatrix(data=data.matrix(train_train[,-match(c("target"),colnames(train_train))]),
                      label=train_train$target)
  dval<-xgb.DMatrix(data=data.matrix(train_validation[,-match(c("target"),colnames(train_validation))]),
                    label=train_validation$target)
  watchlist<-list(val=dval,train=dtrain)
  param <- list(  objective           = "reg:logistic", 
                  booster             = "gbtree",
                  eta                 = 0.005, #Change to 0.001
                  max_depth           = 10, 
                  subsample           = 0.9,
                  colsample_bytree    = 0.7,
                  eval_metric         = "logloss"
  )
  
  xgb_model <- xgb.train(params=param,
                         data=dtrain,
                         nrounds=15000,
                         watchlist=watchlist,
                         early.stop.round=500,
                         maximize=FALSE)
  
  xgb_predictions <- predict(xgb_model,dval)
  xgb_predictions <- pmax(pmin(xgb_predictions,1-1e-15),1e-15)
  
  score[i] <- logLoss(train_validation$target,xgb_predictions)
  #print(logLoss(train_validation$target,xgb_predictions))
  print(score)
  
  if(VISUALIZE){
    #xgb_dump <- xgb.dump(xgb_model, with.stats = T)
    #head(xgb_dump,10)
    names <- names(train_train)[1:(ncol(train_train))]
    names <- names[-match("target",names)]
    imp_matrix <- xgb.importance(names,model=xgb_model)
    xgb.plot.importance(imp_matrix[1:50,])
  }
  
  if(CREATE_SUB){
    test_predictions <- predict(xgb_model,data.matrix(test_data))
    submission <- data.frame(ID=test_id,PredictedProb=test_predictions)
    filename <- paste("xgb_sub_v15_file_",i,".csv",sep="")
    write.csv(submission,filename,row.names=FALSE)
  }
}
print(Sys.time()-start_time)

#Raw 5 fold cv score 0.4638590 0.4576357 0.4546944 0.4626775 0.4621207
#Raw 5 fold cv score  (with all one hot encoding) - 0.4629084 0.4563442 0.4535088 0.4608090 0.4632652
#5 fold stacking - 0.4626662 0.4575043 0.4541675 0.4606216 0.4630127
#5 fold stacking (with n_na) - 0.4628117 0.4574310 0.4543138 0.4604529 0.4622149
#5 fold - xgb with n_na, RF without n_na-0.4629061 0.4576059 0.4538389 0.4604783 0.4629234
#5 fold - removing zero var and 109 as variable 0.4628673 0.4575760 0.4537044 0.4605410 0.4629253
#10 fold CV - [1] 0.4604237 0.4646530 0.4598208 0.4538147 0.4506693 0.4538520 0.4707032 0.4510785 0.4524065 0.4619989
# 0.4597459
#0.4516260 0.4558348 0.4497156 0.4473540 0.4422561 0.4467468 0.4627722 0.4432994 0.4434031 0.4535810