setwd("~/Documents/Kaggle/kaggle_bnp")
library(xgboost)
library(readr)
library(Metrics)
library(Ckmeans.1d.dp) #For xgb importance plot
library(caret)
library(ranger)

impute_median <- function(val){
  m <- median(val,na.rm=TRUE)
  val <- ifelse(is.na(val),-1,val)
  return(val)
}

cat("Read data\n")
train_data <- read_csv('train.csv')
test_data <- read_csv('test.csv')

cat("Get columns with many NA\n")
na_col <- 0
if(IS_RF){
  #To get na in each column
  train_na <- colSums(apply(train_data,1,is.na))
  test_na <- colSums(apply(test_data,1,is.na))
  alot_na_train <- train_data[train_na==100,]
  alot_na_test <- test_data[test_na==100,]
  sum_isna <- function(x){
    return(sum(is.na(x)))
  }
  na_col <- names(alot_na_train)[apply(alot_na_train,2,sum_isna)==47438]
  
  train_data <- train_data[,-match(na_col,colnames(train_data))]
  test_data <- test_data[,-match(na_col,colnames(test_data))]
  
  rm(alot_na_train,alot_na_test)
}

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

cat("Impute NAs with median\n")
c_d <- as.data.frame(apply(c_d,2,impute_median))

train_data <- c_d[1:nrow(train_data),]
test_data <- c_d[(nrow(train_data)+1):nrow(c_d),]
train_data$target <- train_target

rm(c_d,df,v,f,i,levels)
gc(reset=TRUE)