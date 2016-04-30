# kaggle-bnp
## 146th place code
### https://www.kaggle.com/c/bnp-paribas-cardif-claims-management

The code illustrates implementation of stacking as illustrated on this post. https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19134/blended-ensemble-why-it-never-works-for-me/109125#post109125
ExtraTrees, GLMNET and Random Forest are used to do 10 fold predictions on test and train set. The predictions are used with train data in 10 fold CV XGBOOST to produce final predictions. The 10 predictions generated from XGBOOST are averaged to produce submission file.
