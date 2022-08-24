This file contains supplementary material on the design of the study of the effects of early retirement on health
reported in Ghasempour, Moosavi and de Luna (2022, Convolutional neural networks for valid and efficient causal inference; soon to be published on arXiv).
The study is an observational study using Swedish register data from Statistics Sweden and the Swedish Nationa Board of Health and Welfare, which were linked at the individual level at the Umeå SIMSAM Lab (https://www.umu.se/forskning/infrastruktur/umea-simsam-lab/).

To study the effects of early retirement we consider those who were still alive at age 62, and either retire at age 62 ($T=1$, treatment) or retire later ($T=0$, control group). 

### Retirement indicator

An individual alive at age 62 is considered as taking early retirement at that age if hers/his pension transfers become larger than income from work at that age for the first time (i.e., they were never so earlier).

Two income from work definition were used and both gave similar results (the second one below corresponds to the results reported in the paper):

First definition:

> I1 = LoneInk+ArbLos+AmPol+Fortid+FInk+SjukSum\_belop,

where the variables from the LISA register at Statistics Sweden:

-   LoneInk: Cash gross salary
-   ArbLos: Total income caused by unemployment
-   AmPol: Total income caused by labor market policy measures
-   Fortid: Total income caused by early retirement/sickness benefit
-   Fink: Surplus income from active business activities
-   SjukSum\_belop: Sums the amount paid for types of compensation of
    illnesses during the year that has been transferred to sickness
    benefit, preventive sickness benefit, occupational injury benefit,
    and/or rehabilitation benefit.

Second definition: 

In the second definition, the value **CSFVI** is considered, which corresponds to
all recorded taxable incomes and is obtained from the Income and Taxation Register (IoT)
register at Statistics Sweden. 

Transfers from old-age pensions is obtained from the variable Aldpens (from LISA). 
Whereas the main variable in the second definition (CSFVI)
comes from the Income and Taxation Register (IoT) dataset. The measures
are defined by: **P/ (I1 +1)** and **P/ (I2 +1)**, where P is the
**Aldpens** and I2 is the Max (0, CSFVI - P).

Therefore, the criteria for retirement year are when those measures
increased from a value less than one to a value above. Of course, we
make sure that the measures do not fall below one so long as data has
been recorded. Otherwise, the last rise is considered to determine the
retirement year. In this analysis, the retirement age is considered a
treatment in the way that who retired exactly at 62 are defined as treated
individuals, and who retired after 62 are defined as
untreated. This research aims to control for available covariates to
estimate the effect of the described treatment on average health
conditions. The covariates we have controlled for them are: Gender,
Marriage status, municipality, education level, and the number of
biological children. Besides these scalar variables, we can control for
time series data. Therefore, multiple time series are also controlled:

-   Log of LoneInk: (10 years before treatment)
-   AldPens (10 years before treatment)
-   Unemployment = ArbLos+AmPol (10 years before treatment)
-   Fortid (10 years before treatment)
-   SjukSum\_Belopp (10 years before treatment)
-   Par\_SV (number of Inpatient cares per year, 10 years)
-   Par\_OV (number of Specialized outpatient cares per year, 10 years)
-   Spouse retirement (a time series that shows the state of the
    spouse’s retirement, 10 years)

The developed method based on CNN is implemented by using the package
CausalDNN to perform the analysis on the population cohort who were born
in 1946 and 1947 and survived till sixty one.

### CNN Architecture

Here is the code for running Cnn based aipw estimator for the average
treatment effect on the treated. One can define any arbitrary deep
neural network architecture and pass it to the function. Models can be created using the Keras package. The outcome model and propensity score are estimated by two different models. The outcome model is a convolutional network with two layers with 128 and 16 filters in the layers respectively. For the propensity score, just 32 and 8 filters were used in two layers.
Inputted time series can be centralized either columnwise or rowwise by the package. In this case, we have used columnwise centrilization. For both cases, the mean and standard deviation of the vectors are kept as a new variable and are passed to the part of the architecture that handles the scalers as inputs.

``` r
model_m = keras::keras_model_sequential()
model_m = keras::layer_conv_1d(model_m, 128,4, padding = 'valid' , activation = 'relu', input_shape = c(10,7))
model_m = keras::layer_conv_1d(model_m, 16,3, padding = 'same', activation = 'relu')
model_m = keras::layer_flatten(model_m)
model_p = keras::keras_model_sequential()
model_p = keras::layer_conv_1d(model_p, 32,4, padding = 'valid' , activation = 'relu', input_shape = c(10,7))
model_p = keras::layer_conv_1d(model_p, 8,3, padding = 'same', activation = 'relu')
model_p = keras::layer_flatten(model_p)


ATT = DNNCausal::aipw.att(Y=Observed_outcomes, T=Treatment, X_t=Timeseries_covariates,X = scalar_covariates, model = c(model_m, model_p), do_standardize = 'Column', verbose=FALSE, epochs = c(64,32), batch_size = 500)
```

For running this model the following hyperparameters have been chosen.
batch size is 500 and the number of epochs is 64 for the fitting outcome
model. the number of epochs for training propensity score in this study
is chosen to be 32.

For each group men and women, and for each outcome years after treatment
one to five, a similar code has been utilized to estimate the parameter
of interest in that case.
