setwd("~/Documents/Kaggle/kaggle_bnp")
library(xgboost)
library(readr)
library(Metrics)
library(Ckmeans.1d.dp) #For xgb importance plot
library(caret)

start_time <- Sys.time()
impute_median <- function(val){
  m <- median(val,na.rm=TRUE)
  val <- ifelse(is.na(val),-1,val)
  return(val)
}

CREATE_SUB <- FALSE
VISUALIZE <- FALSE

cat("Read data\n")
train_data <- read_csv('train.csv')
test_data <- read_csv('test.csv')

#To get na in each column
# train_na <- colSums(apply(train_data,1,is.na))
# test_na <- colSums(apply(test_data,1,is.na))
# 
# train_data$is_na <- train_na==100
# test_data$is_na <- test_na==100

# cat("Remove columns with many many NAs\n")
# train_data <- train_data[,-match(na_col,colnames(train_data))]
# test_data <- test_data[,-match(na_col,colnames(test_data))]

# cat("Remove outliers\n")
# pos_outliers <- c("v1","v4","v6","v7","v15","v26","v27","v29","v32","v33","v34",
#                   "v35","v37","v41","v43","v49","v50","v55","v57","v60","v73",
#                   "v80","v83","v84","v85","v86","v88","v96","v99","v108","v111",
#                   "v116","v121","v127","v128","v130","v131")
# neg_outliers <- c("v11","v41","v48","v53","v61","v114")
# 
# for(i in 1:length(pos_outliers)){
#   train_data <- train_data[!(!is.na(train_data[,pos_outliers[i]]) & (train_data[,pos_outliers[i]] >16)),]
# }
# 
# for(i in 1:length(neg_outliers)){
#   train_data <- train_data[!(!is.na(train_data[,neg_outliers[i]]) & (train_data[,neg_outliers[i]] <= 1)),]
# }

train_target <- train_data$target
train_id <- train_data$ID
test_id <- test_data$ID

train_data <- train_data[,-match(c("target","ID"),colnames(train_data))]
test_data <- test_data[,-match(c("ID"),colnames(test_data))]

c_d <- rbind(train_data,test_data)

fv <- names(c_d)[unlist(lapply(c_d,class))=="character"]
nv <- names(c_d)[unlist(lapply(c_d,class))!="character"]
fv1 <- c("v22")
fv <- fv[-match(fv1,fv)]

cat("One hot encoding for factor variables\n")
for(i in 1:length(fv)){
  v <- as.data.frame(c_d[,match(fv[i],colnames(c_d))])
  names(v) <- c(paste(fv[i],"_",sep=""))
  df <- data.frame(model.matrix(~.-1,v))
  c_d <- cbind(c_d,df)
}
c_d <- c_d[,-match(fv,colnames(c_d))]

cat("Replacing factor variables with large values into numeric form\n")
for (f in fv1) {
    levels <- unique(c_d[[f]])
    c_d[[f]] <- as.integer(factor(c_d[[f]], levels=levels))
}

#cat("Remove v22\n")
#c_d <- c_d[-match(c("v22"),colnames(c_d)),]

# cat("Create variables to create NA 0/1 for each variable\n")
# for(i in 1:length(nv)){
#   is_na <- is.na(c_d[,nv[i]])
#   col_name <- paste("na_",nv[i],sep="")
#   c_d[,col_name] <- is_na
# }

cat("Impute NAs with median\n")
c_d <- as.data.frame(apply(c_d,2,impute_median))

train_data <- c_d[1:nrow(train_data),]
test_data <- c_d[(nrow(train_data)+1):nrow(c_d),]
train_data$target <- train_target

rm(c_d,df,v,f,i,levels)
gc(reset=TRUE)

#cat("Get zero variance variables\n")
#zero.var <- nearZeroVar(train_data[,-match(c("target"),colnames(train_data))],saveMetrics = TRUE)
#zero.var.variables <- rownames(zero.var[zero.var$zeroVar==TRUE,])

score <- c(0,0,0,0,0)
for (i in 2:5) {
  cat("Creating model ",i, " of 5\n")
  cat("Sample Data\n")
  set.seed(1234)
  #s <- sample(0.70*nrow(train_data))
  #train_train <- train_data[s,]
  #train_validation <- train_data[-s,]
  #rm(s)
  
  st <- ceiling(nrow(train_data)/5)
  s <- (st*(i-1) + 1):min((st*i),nrow(train_data))
  train_train <- train_data[-s,]
  train_validation <- train_data[s,]
  
#   if(length(zero.var.variables > 0)){
#     train_train <- train_train[,-match(zero.var.variables,colnames(train_train))]
#     train_validation<- train_validation[,-match(zero.var.variables,colnames(train_validation))]
#   }
  
  cat("Create xgboost model\n")
  dtrain<-xgb.DMatrix(data=data.matrix(train_train[,-match(c("target"),colnames(train_train))]),
                      label=train_train$target)
  dval<-xgb.DMatrix(data=data.matrix(train_validation[,-match(c("target"),colnames(train_validation))]),
                    label=train_validation$target)
  watchlist<-list(val=dval,train=dtrain)
  param <- list(  objective           = "reg:logistic", 
                  booster             = "gbtree",
                  eta                 = 0.005,
                  max_depth           = 10, 
                  subsample           = 0.9,
                  colsample_bytree    = 0.7,
                  eval_metric         = "logloss"
  )
  
  xgb_model <- xgb.train(params=param,
                         data=dtrain,
                         nrounds=12000,
                         watchlist=watchlist,
                         early.stop.round=500,
                         maximize=FALSE)
  
  xgb_predictions <- predict(xgb_model,dval)
  xgb_predictions <- pmax(pmin(xgb_predictions,1-1e-15),1e-15)
  
  score[i] <- logLoss(train_validation$target,xgb_predictions)
  print(logLoss(train_validation$target,xgb_predictions))
  
  if(VISUALIZE){
    xgb_dump <- xgb.dump(xgb_model, with.stats = T)
    head(xgb_dump,10)
    names <- names(train_train)[1:(ncol(train_train)-1)]
    imp_matrix <- xgb.importance(names,model=xgb_model)
    xgb.plot.importance(imp_matrix[1:50,])
  }
  
  if(CREATE_SUB){
    test_predictions <- predict(xgb_model,data.matrix(test_data))
    submission <- data.frame(ID=test_id,PredictedProb=test_predictions)
    filename <- paste("xgb_sub_v6_file_",i,".csv",sep="")
    write.csv(submission,filename,row.names=FALSE)
  }
}
print(Sys.time()-start_time)

#Raw 5 fold cv score 0.4638590 0.4576357 0.4546944 0.4626775 0.4621207
#Raw 5 fold cv score 0.4629084 0.4563442 0.4535088 0.4608090 0.4632652