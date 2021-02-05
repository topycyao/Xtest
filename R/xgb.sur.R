library(MASS)
library(xgboost)
library(survival)
library(dplyr)
library(magrittr)
library(survcomp)
library(data.table)
#using xgboost with loss function of cox paritial likelihood or cindex

xgb.sur<-function(datax,datay,method=c('defaut','pl','C'),nfolds=5,nround=NULL,
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
  XDtrain <- xgb.DMatrix(x_train, label = y_train_boost)
  if(method=='C')
    model<-xgb.cv(list(objective = cidx_xgb_obj, eval_metric = cidx_xgb_func,
                       tree_method = 'hist', grow_policy = 'lossguide',
                       eta = eta, lambda = lambda, alpha = alpha, subsample = .5,
                       colsample_bytree = .5), XDtrain, nround = nround,
                  nfold = nfolds, verbose = F, early_stopping_rounds = early_stopping_rounds, maximize = T,
                  callbacks = list(cb.cv.predict(T)))
  else    model<-xgb.cv(list(objective = 'survival:cox', eval_metric = cidx_xgb_func,
                             tree_method = 'hist', grow_policy = 'lossguide',
                             eta = eta, lambda = lambda, alpha = alpha, subsample = .5,
                             colsample_bytree = .5), XDtrain, nround = nround,
                        nfold = nfolds, verbose = F, early_stopping_rounds = early_stopping_rounds, maximize = T,
                        callbacks = list(cb.cv.predict(T)))
  model
  }
