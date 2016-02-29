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

cat("data_processing.R - Read data\n")
train_data <- read_csv('train.csv')
test_data <- read_csv('test.csv')

# cat("Remove redundant variables\n")
# red_var <- c("v91")
# train_data <- train_data[,-match(red_var,colnames(train_data))]
# test_data <- test_data[,-match(red_var,colnames(test_data))]

cat("data_processing.R - Adding column for number of NA in each row\n")
train_data$n_na <- colSums(apply(train_data,1,is.na))
test_data$n_na <- colSums(apply(test_data,1,is.na))

cat("data_processing.R - Get columns with many NA\n")
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

cat("data_processing.R - Saving target and ID variables\n")
train_target <- train_data$target
train_id <- train_data$ID
test_id <- test_data$ID

train_data <- train_data[,-match(c("target","ID"),colnames(train_data))]
test_data <- test_data[,-match(c("ID"),colnames(test_data))]

c_d <- rbind(train_data,test_data)

#cat("Create v22 frequency\n")
#val_freq <- as.data.frame(table(c_d$v22))
#names(val_freq) <- c("v22","v22_freq")
#c_d <- merge(c_d,val_freq,by.x="v22",by.y="v22",all.x=TRUE,all.y=FALSE)
# c_d <- c_d[,-match(c("v22"),colnames(c_d))]
# c_d$v22_freq <- as.character(c_d$v22_freq)

fv <- names(c_d)[unlist(lapply(c_d,class))=="character"]
nv <- names(c_d)[unlist(lapply(c_d,class))!="character"]
fv1 <- c("v22")
fv <- fv[-match(fv1,fv)]

cat("data_processing.R - One hot encoding for factor variables\n")
for(i in 1:length(fv)){
  v <- as.data.frame(c_d[,match(fv[i],colnames(c_d))])
  names(v) <- c(paste(fv[i],"_",sep=""))
  df <- data.frame(model.matrix(~.-1,v))
  #remove one redundant column, e.g We only need 3 for 4 factors.
  #df <- df[,-c(1)]
  c_d <- cbind(c_d,df)
}
c_d <- c_d[,-match(fv,colnames(c_d))]

cat("data_processing.R - Replacing factor variables with large values into numeric form\n")
for (f in fv1) {
  levels <- unique(c_d[[f]])
  c_d[[f]] <- as.integer(factor(c_d[[f]], levels=levels))
}

cat("data_processing.R - Impute NAs with median\n")
c_d <- as.data.frame(apply(c_d,2,impute_median))

train_data <- c_d[1:nrow(train_data),]
test_data <- c_d[(nrow(train_data)+1):nrow(c_d),]
train_data$target <- train_target

# cat("data_processing.R - Get zero variance variables\n")
# zero.var <- nearZeroVar(train_data[,-match(c("target"),colnames(train_data))],saveMetrics = TRUE)
# zero.var.variables <- rownames(zero.var[zero.var$zeroVar==TRUE,])
# train_data <- train_data[,-match(zero.var.variables,colnames(train_data))]
# test_data <- test_data[,-match(zero.var.variables,colnames(test_data))]

rm(c_d,df,v,f,i,levels)
gc(reset=TRUE)
