library(MASS)
library(lightgbm)
library(survival)
library(dplyr)
library(magrittr)
library(survcomp)
library(data.table)
source('loss_func.R')
#using xgboost with loss function of cox paritial likelihood or cindex

lgb.sur<-function(datax,datay,method=c('defaut','pl','C'),nfolds=5,nround=NULL,
                  lambda=NULL,alpha=NULL,eta=NULL,early_stopping_rounds=NULL
)
{
  x_train=datax
  y=datay
  method=match.arg(method)
  if(is.null(lambda))
    lambda=.01
  if(is.null(alpha))
    alpha=.01
  if(is.null(eta))
    eta=.01
  if(is.null(nround))
    nround=1000
  if(is.null(early_stopping_rounds))
    early_stopping_rounds=20

  tt<-length(x_train[,1])
  surv_time <- Surv(y$time,y$status)
  y_train_boost <-  2 * y[,2] * (y[,1] - .5) #make fisrt col status and second col time
  #y_train<-surv_time
  LDtrain <- lgb.Dataset(x_train, label = y_train_boost)
  if(method=='C')
    model<-lgb.cv(list(objective = cidx_lgb_obj,
                       eta = eta, lambda = lambda, alpha = alpha, subsample = .5,
                       colsample_bytree = .5), LDtrain, nround = nround,eval=cidx_lgb_func,
                  nfold = nfolds, verbose = 0, early_stopping_rounds = early_stopping_rounds)
  else    model<-lgb.cv(list(objective = Cox_lgb_obj,
                             eta = eta, lambda = lambda, alpha = alpha, subsample = .5,
                             colsample_bytree = .5), LDtrain, nround = nround,eval=cidx_lgb_func,
                        nfold = nfolds, verbose = 0, early_stopping_rounds = early_stopping_rounds)
  model
}
