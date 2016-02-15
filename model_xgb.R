setwd("~/Documents/Kaggle/kaggle_bnp")
library(xgboost)
library(readr)
library(Metrics)
library(Ckmeans.1d.dp) #For xgb importance plot
library(caret)

impute_median <- function(val){
  m <- median(val,na.rm=TRUE)
  val <- ifelse(is.na(val),m,val)
  return(val)
}

CREATE_SUB <- TRUE
VISUALIZE <- FALSE

cat("Read data\n")
train_data <- read_csv('train.csv')
test_data <- read_csv('test.csv')

train_target <- train_data$target
train_id <- train_data$ID
test_id <- test_data$ID

train_data <- train_data[,-match(c("target","ID"),colnames(train_data))]
test_data <- test_data[,-match(c("ID"),colnames(test_data))]

c_d <- rbind(train_data,test_data)

fv <- names(c_d)[unlist(lapply(c_d,class))=="character"]
fv1 <- c("v22","v56","v125")
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

cat("Impute NAs with median\n")
c_d <- as.data.frame(apply(c_d,2,impute_median))

train_data <- c_d[1:nrow(train_data),]
test_data <- c_d[(nrow(train_data)+1):nrow(c_d),]
train_data$target <- train_target

rm(c_d,df,v,f,i,levels)
gc(reset=TRUE)

cat("Get zero variance variables\n")
zero.var <- nearZeroVar(train_data[,-match(c("target"),colnames(train_data))],saveMetrics = TRUE)
zero.var.variables <- rownames(zero.var[zero.var$zeroVar==TRUE,])

cat("Sample Data\n")
set.seed(123)
s <- sample(0.70*nrow(train_data))
train_train <- train_data[s,]
train_validation <- train_data[-s,]
rm(s)

train_train <- train_train[,-match(zero.var.variables,colnames(train_train))]
train_validation<- train_validation[,-match(zero.var.variables,colnames(train_validation))]

cat("Create xgboost model\n")
dtrain<-xgb.DMatrix(data=data.matrix(train_train[,-match(c("target"),colnames(train_train))]),
                    label=train_train$target)
dval<-xgb.DMatrix(data=data.matrix(train_validation[,-match(c("target"),colnames(train_validation))]),
                  label=train_validation$target)
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:logistic", 
                booster             = "gbtree",
                eta                 = 0.01,
                max_depth           = 10, 
                subsample           = 0.9,
                colsample_bytree    = 0.7,
                eval_metric         = "logloss"
)

set.seed(123)
xgb_model <- xgb.train(params=param,
                       data=dtrain,
                       nrounds=2000,
                       watchlist=watchlist,
                       early.stop.round=200,
                       maximize=FALSE)

xgb_predictions <- predict(xgb_model,dval)
xgb_predictions <- pmax(pmin(xgb_predictions,1-1e-15),1e-15)

print(logLoss(train_validation$target,xgb_predictions))

if(VISUALIZE){
  xgb_dump <- xgb.dump(xgb_model, with.stats = T)
  head(xgb_dump,10)
  names <- names(train_train)[1:(ncol(train_train)-1)]
  imp_matrix <- xgb.importance(names,model=xgb_model)
  xgb.plot.importance(imp_matrix[1:20,])
}

if(CREATE_SUB){
  test_predictions <- predict(xgb_model,data.matrix(test_data))
  submission <- data.frame(ID=test_id,PredictedProb=test_predictions)
  write.csv(submission,"xgb_v3.csv",row.names=FALSE)
}

#0.4591959
#0.4600833
#0.4603292
#0.45933