This repository includes supplementary material to the manuscript 
Ghasempour, Moosavi and de Luna (2023, Convolutional neural networks for valid and efficient causal inference; [arXiv version](https://arxiv.org/abs/2301.11732)).

Simulations
----------

Below are the packages that are neccesary for running the simulation study.

``` r
library(DNNCausal)       # available at: github.com/stat4reg/DNNCausal
library(tmle)
library(xtable)
library(glmnet)
library(MASS)
library(hdm)
```
Performing the post-lasso methods requires the following functions from the package 'hdim.ui', which is accessible from the GitHub page: 'github.com/stat4reg/hdim.ui'. 
``` r
source('lasso_ATT.R')    # available at: github.com/stat4reg/hdim.ui
source('sandImpACT.R')   # available at: github.com/stat4reg/hdim.ui
```

DGPs
----

Two different Data generating processes were considered in the
simulation study. 

### First DGP

``` r
DGP_1 <- function(N) {
  
  # generate the covariates
  x1 = rnorm(N, 100, 20)
  x2 = rnorm(N, 102, 15)
  x3 = rnorm(N, 105, 13)
  x4 = rnorm(N, 107, 11)
  x5 = rnorm(N, 109, 8)
  x6 = rnorm(N, 110, 20)
  x7 = rnorm(N, 112, 15)
  x8 = rnorm(N, 115, 13)
  x9 = rnorm(N, 117, 11)
  x10 = rnorm(N, 119, 8)
  
  # define the outcome models
  f0 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    return(1+0.001*((x2-x1)^2+(x4-x3)^3+ (x6-x5)^2+(x8-x7)^3+(x10-x9)^2))
  }
  f1 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    return(2-0.001*((x2-x1)^2+(x4-x3)^3+ (x6-x5)^2+(x8-x7)^3+(x10-x9)^2))
  }
  
  # call the outcome models on generated covariates
  m0 = f0(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
  m1 = f1(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) 
  
  # generate outcome errors from random variables with a normal distribution
  e0 = rnorm(N, mean = 0, sd = 13)
  e1 = rnorm(N, mean = 0, sd = 13)
  
  # simulate the potential outcomes
  y0 = m0 + e0
  y1 = m1 + e1
  
  # generate treatment probabilities
  prob = 1/(1+exp(0.000005*(  (x2-x1)^2+(x4-x3)^3+(x6-x5)^2+(x8-x7)^3+(x10-x9)^2  )))
  
  # simulate the exposure level based on the probabilities 
  Tr = rbinom(N, 1, prob)
  
  # form the observed outcome
  Y = Tr * y1 + (1 - Tr) * y0
  # return the list of covariates, observed outcome, and treatment level
  return(list(x=cbind(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6,x7=x7,x8=x8,x9=x9,x10=x10),y0=y0,y1=y1, T=Tr, p = prob,y=Y))
}
```

### Second DGP

``` r
DGP_2 <- function(N) {
  
  # generate the covariates
  x1 = rnorm(N, 100, 20)
  x2 = rnorm(N, 102, 15)
  x3 = rnorm(N, 105, 13)
  x4 = rnorm(N, 107, 11)
  x5 = rnorm(N, 109, 8)
  x6 = rnorm(N, 110, 20)
  x7 = rnorm(N, 112, 15)
  x8 = rnorm(N, 115, 13)
  x9 = rnorm(N, 117, 11)
  x10 = rnorm(N, 119, 8)
  
  # define the non-linear step functions that will be used in the outcome models
  I1 = function(x,y,z,w){
    return(10*(((y-x)/x) > .15 & ((z-y)/y) > .15 & ((w-z)/z) > .15 ))
  }
  I2 = function(x,y,z,w){
    return(5*(((y-x)/x) < .05 & ((z-y)/y) < .05 & ((w-z)/z) < .05))
  }
  I3 = function(x,y,z,w){
    return(3*(sign(y - 1.1*x)*sign(z - 1.1*y) ==-1 & sign(y - 1.1*x)*sign(z - 1.1*y) ==-1) )
  }
  
  # define the outcome models
  f0 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    return(1 + 1*(I1(x1,x2,x3,x4) + I2(x4,x5,x6,x7) + I3(x6,x7,x8,x9) ))
  }
  f1 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    return(2 - 1*(I1(x1,x2,x3,x4) + I2(x4,x5,x6,x7) +I3(x6,x7,x8,x9) ))
  }
  
  # call the outcome models on generated covariates
  m0 = f0(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
  m1 = f1(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) 
  
  # generate outcome errors from random variables with a normal distribution
  e0 = rnorm(N, mean = 0, sd = 1)
  e1 = rnorm(N, mean = 0, sd = 1)
  
  # simulate the potential outcomes
  y0 = m0 + e0
  y1 = m1 + e1
  
  # generate treatment probabilities
  prob = 1/(1+1*exp(.1*(0.05*x5 - I1(x1,x2,x3,x4) + I2(x4,x5,x6,x7) +  I3(x6,x7,x8,x9)   )  ))
  # simulate the exposure level based on the probabilities 
  Tr = rbinom(N, 1, prob)
  
  # form the observed outcome
  Y = Tr * y1 + (1 - Tr) * y0
  
  # return the list of covariates, observed outcome, and treatment level
  return(list(x=cbind(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6,x7=x7,x8=x8,x9=x9,x10=x10),y0=y0,y1=y1, T=Tr, p = prob,y=Y))
}
```
<!---
### Third DGP

``` r
DGP_4 <- function(N) {
  
  x1 = rnorm(N, 100, 20)
  x2 = rnorm(N, 102, 15)
  x3 = rnorm(N, 105, 13)
  x4 = rnorm(N, 107, 11)
  x5 = rnorm(N, 109, 8)
  x6 = rnorm(N, 110, 20)
  x7 = rnorm(N, 112, 15)
  x8 = rnorm(N, 115, 13)
  x9 = rnorm(N, 117, 11)
  x10 = rnorm(N, 119, 8)
  
  I1 = function(x,y,z,w){
    return(1*(((y-x)/x) > .15 & ((z-y)/y) > .15 & ((w-z)/z) > .15 ))
    
  }
  I2 = function(x,y,z,w){
    return(1*(((y-x)/x) < .05 & ((z-y)/y) < .05 & ((w-z)/z) < .05))
    
  }
  I3 = function(x,y,z,w){
    return(1*((sign(y - 1.1*x)*sign(z - 1.1*y) ==-1 & sign(y - 1.1*x)*sign(z - 1.1*y) ==-1) ))
  }
  
  
  f0 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    
    return(1+
             10*(0.0001*(x2-x1)^3+0.0001*(x3-x2)^3+0.0001*(x4-x3)^3+0.0001*(x5-x4)^3+
                   I1(x1,x2,x3,x4) + I1(x2,x3,x4,x5) + I1(x3,x4,x5,x6)  +
                   I2(x1,x2,x3,x4) + I2(x2,x3,x4,x5) +I2(x3,x4,x5,x6) +
                   I3(x1,x2,x3,x4) +I3(x2,x3,x4,x5) + I3(x3,x4,x5,x6)) +
             10*floor(5*sin(5/(x3-x2))*(x4-x3)))
  }
  f1 = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10){
    
    return(2-10*(    0.0001*(x2-x1)^3+0.0001*(x3-x2)^3+0.0001*(x4-x3)^3+0.0001*(x5-x4)^3+
                       I1(x1,x2,x3,x4) + I1(x2,x3,x4,x5) + I1(x3,x4,x5,x6)  +
                       I2(x1,x2,x3,x4) + I2(x2,x3,x4,x5) + I2(x3,x4,x5,x6) +
                       I3(x1,x2,x3,x4) + I3(x2,x3,x4,x5) +I3(x3,x4,x5,x6)) -
             10*floor(5*sin(5/(x3-x2))*(x4-x3)))
  }
  
  m0 = f0(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
  m1 = f1(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) 
  
  e0 = rnorm(N, mean = 0, sd = 1)
  e1 = rnorm(N, mean = 0, sd = 1)
  
  y0 = m0 + e0
  y1 = m1 + e1
  
  prob =           1/(1+exp(0.04*(0.0001*(x2-x1)^3+0.0001*(x3-x2)^3+0.0001*(x4-x3)^3+0.0001*(x5-x4)^3+
                                    I1(x1,x2,x3,x4) + I1(x2,x3,x4,x5) + I1(x3,x4,x5,x6)  +
                                    I2(x1,x2,x3,x4) + I2(x2,x3,x4,x5) + I2(x3,x4,x5,x6) +
                                    I3(x1,x2,x3,x4) + I3(x2,x3,x4,x5) + I3(x3,x4,x5,x6)) -
                              floor(0.1*sin(0.1/(x3-x2))*(x4-x3))
  ))
  
  
  

  Tr = rbinom(N, 1, prob)

  
  Y = Tr * y1 + (1 - Tr) * y0

  mean(y1)-mean(y0)
  mean(Y[Tr==1])- mean(Y[Tr==0])
  
  return(list(x=cbind(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6,x7=x7,x8=x8,x9=x9,x10=x10),y0=y0,y1=y1, T=Tr, p = prob,y=Y))
}
```
--->

The R function to run the simulations
-----------------------

``` r

simulation=function(rep=1000, n = 10000, k= 10,seed = 1169876, FUN){
  set.seed(seed)
  # defining lists to store the results of replications
  post_lasso=list()
  att_list=c()
  aipwCNN = list()
  aipwfNN = list()
  
  
  for (i in 1:rep) {
    # generate data with the DGP function 
    data = FUN(n)
    
    # remove objects that have been defined in the previous iteration to make sure that Keras use them in the current iteration.
    optimizer_m = NULL
    optimizer_p = NULL
    optimizer = NULL
    optimizer1_m = NULL
    optimizer1_p = NULL
    optimizer1 = NULL
    model_m = NULL
    model_p = NULL
    model1_m = NULL
    model1_p = NULL
    
    # call the garbage collector to avoid overstack flow
    gc()
    
    # define optimizer functions
    optimizer_m = keras::optimizer_adam(lr = 0.003)
    optimizer_p = keras::optimizer_adam(lr = 0.003)
    optimizer = c(optimizer_m, optimizer_p)
    optimizer1_m = keras::optimizer_adam(lr = 0.003)
    optimizer1_p = keras::optimizer_adam(lr = 0.003)
    optimizer1 = c(optimizer1_m, optimizer1_p)
    
    
    # define the models using keras sequential model
    model_m = keras::keras_model_sequential()
    model_p = keras::keras_model_sequential()
    model_m =  keras::layer_conv_1d(model_m,128,4,padding = 'valid',activation = "relu",input_shape = c(k,1))
    model_m =  keras::layer_conv_1d(model_m,16,3,padding = 'same',activation = "relu")
    model_m =  keras::layer_flatten(model_m)
    
    model_p =  keras::layer_conv_1d(model_p,32,4,padding = 'valid',activation = "relu",input_shape = c(k,1 ))
    model_p =  keras::layer_conv_1d(model_p,8,3,padding = 'same',activation = "relu")
    model_p =  keras::layer_flatten(model_p)
    
    model1_m = keras::keras_model_sequential()
    model1_p = keras::keras_model_sequential()
    model1_m =  keras::layer_dense(model1_m,128,activation = "relu",input_shape = k)
    model1_m =  keras::layer_dense(model1_m,80,activation = "relu")
    
    model1_p =  keras::layer_dense(model1_p,32,activation = "relu",input_shape = k )
    model1_p =  keras::layer_dense(model1_p,8,activation = "relu")
    
    # perform post lasso estimations
    apr_data= list('X'=as.matrix(data$x) ,'Y'=data$y ,'T'= as.logical(data$T))
    post_la=simulation_postlasso(apr_data)
    post_lasso[[i]]=((results_post_lasso(post_la))) 
    
    # store the real ATT
    att_list[i]= mean(data$y1[data$T==1]-data$y0[data$T==1])
    
    # perform the CNN estimator
    aipwCNN[[i]] = aipw.att(data$y,as.logical(data$T), X_t = list(data$x) ,model =c(model_m,model_p),
                            optimizer = optimizer,epochs = c(100,50),batch_size = 1000,verbose = FALSE,
                            use_scalers = FALSE, debugging = FALSE,rescale_outcome = TRUE,
                            do_standardize ="Column" )#,X_r"Column"
    
    # perform the FNN estimator
    aipwfNN[[i]] = aipw.att(data$y,as.logical(data$T), X_t = data$x ,model =c(model1_m,model1_p),
                            optimizer = optimizer1,epochs = c(100,50),batch_size = 1000,verbose = FALSE,
                            use_scalers = FALSE, debugging = FALSE,rescale_outcome = TRUE,
                            do_standardize = "Column")#,X_r"Column"
    
  }
  # return the list of results
  return(list('post_lasso' = post_lasso, 'real_ATT' = att_list, 'CNN_ATT' = aipwCNN, 'FNN_ATT' = aipwfNN))
}
```

Now everything is ready and just calling the simulation function is
needed.

``` r
result_DGP1 = simulation(rep = 100, FUN =DGP_1)
result_DGP2 = simulation(rep = 100, FUN =DGP_2)
```
