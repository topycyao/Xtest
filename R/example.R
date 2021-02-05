#example
library(MASS)
library(xgboost)
library(survival)
library(dplyr)
library(magrittr)
library(lightgbm)
library(survcomp)
library(data.table)
library(ggplot2)
library(Ckmeans.1d.dp)

n <- 1000
p <- 100
rho <- 0.5
tt <- round(.8*n)
V <- diag(p)

X <- mvrnorm(n = n, mu = rep(0, p), Sigma = V)
xname<-rep(0,100)
for (i in 1:100) {
  tx<-paste("X",i,sep='')
  xname[i]<-tx

}
colnames(X)<-xname
### nonlinear transformation
mu <- exp(2*pnorm((X[, 10] > 0.5) + X[, 20] ^ 2 - 1) +
            2*pnorm(0.5*X[, 30]+X[, 40]^2 - 1) +
            2*pnorm(0.5 * X[, 50]+X[, 60] ^ 2 - 1) +
            2*pnorm(sin(X[, 70]) + X[, 80] ^ 2 - 1) +
            2*pnorm(cos(X[, 90])+X[, 100] ^ 2 - 1))
### survival time simulation

obs_time <- -(log(runif(n)))/(mu)

a <- 2*rbinom(n = n, size = 1, prob = 1/3)
b <- runif(n = n, min = 0, max = 2)
a[a == 0] <- b[a == 0]
C <- a
obs_time <- pmin(obs_time, C)
status <- as.numeric(obs_time <= C)
surv_time <- Surv(obs_time, status)
y<-cbind(status,obs_time)
colnames(y)<-c('status','time')
y<-as.data.frame(y)
surv_time_boost <-  2 * obs_time * (status - .5)
x_train <- X[seq_len(tt), ]
y_train <- y[seq_len(tt),]

x_test <- X[(tt + 1) : n, ]
y_test <- surv_time[(tt + 1) : n]
y_test_boost <- surv_time_boost[(tt + 1) : n]


xgb_cox_m<-xgb.sur(x_train,y_train)
xgb_cix_m<-xgb.sur(x_train,y_train,method = 'C')

lgb_cox_m<-lgb.sur(x_train,y_train)
lgb_cix_m<-lgb.sur(x_train,y_train,method = 'C')



### Convert test data to XGB/LGB dataset

XDtest <- xgb.DMatrix(x_test, label = y_test_boost)


LDtest <- lgb.Dataset(x_test, label = y_test_boost)
#prediction
#xgb model
y_xgcox_predict <- -rowMeans(sapply(xgb_cox_m$models, predict, XDtest))
y_xgcidx_predict <- -rowMeans(sapply(xgb_cix_m$models, predict, XDtest))

#lgb model
y_lgcox_pred <- y_lgcidx_pred <- matrix(0, n - tt, 5)
for (i in seq_len(5)) {
  y_lgcox_pred[, i] <- predict(lgb_cox_m$boosters[[i]]$booster, x_test)
  y_lgcidx_pred[, i] <- predict(lgb_cix_m$boosters[[i]]$booster, x_test)
}

y_lgcox_predict <- -rowMeans(y_lgcox_pred)
y_lgcidx_predict <- -rowMeans(y_lgcidx_pred)

#validation cindex value
cidx_result <- c('XGB_Cox' = concordance(y_test ~ y_xgcox_predict)$con,
                 'XGB_Cidx' = concordance(y_test ~ y_xgcidx_predict)$con,
                 'LGB_Cox' = concordance(y_test ~ y_lgcox_predict)$con,
                 'LGB_Cidx' = concordance(y_test ~ y_lgcidx_predict)$con)
print(cidx_result)

#importance plot
#xgb model
imp_xgb=xgb.importance(colnames(x_train), model = xgb_cox_m$models[[1]])
imp_xgb_cix=xgb.importance(colnames(x_train), model = xgb_cix_m$models[[1]])

xgb.plot.importance(imp_xgb,top_n = 10)
xgb.plot.importance(imp_xgb_cix,top_n = 10)
#ggplot
(gg <- xgb.ggplot.importance(imp_xgb, top_n = 10,measure = "Frequency", rel_to_first = TRUE))
gg + ggplot2::ylab("Frequency")

#lgb model
imp_lgb=lgb.importance(lgb_cox_m$boosters[[1]]$booster,percentage = TRUE)
imp_lgb_cix=lgb.importance(lgb_cix_m$boosters[[1]]$booster,percentage = TRUE)


lgb.plot.importance(imp_lgb,top_n = 10)
lgb.plot.importance(imp_lgb_cix,top_n = 10)

