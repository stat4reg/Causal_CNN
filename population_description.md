### Retirement indicator

Two different criteria have been created to classify individuals as
still working or retired. In both cases, the main intention is to
compare the different amounts of income according to if they are sourced
by pensions or not. Due to the retirement process that might end in up
to five years to be completed, a threshold is needed for dividing people
into two classes retired and non-retired. The threshold is used to
determine what percentage of someone’s income must be pensions for
counting them as retired. The distinction between the proposed criteria
comes from the difference in calculating the total income excluding the
pensions, and the first criteria is defined by:

> I1 = LoneInk+ArbLos+AmPol+Fortid+FInk+SjukSum\_belop,

where the definition of these recorded statistics is:

-   LoneInk: Cash gross salary
-   ArbLos: Total income caused by unemployment
-   AmPol: Total income caused by labor market policy measures
-   Fortid: Total income caused by early retirement/sickness benefit
-   Fink: Surplus income from active business activities
-   SjukSum\_belop: Sums the amount paid for types of compensation of
    illnesses during the year that has been transferred to sickness
    benefit, preventive sickness benefit, occupational injury benefit,
    and/or rehabilitation benefit.

In the second definition, the value **CSFVI** is considered, which is
recorded as all taxable incomes. The first set of variables is collected
from the Integrated database for labor market research, Statistics
Sweden. In addition to Aldpens that is the Total income from old-age
pensions and is used in both criteria to compare with other kinds of
incomes. Whereas the main variable in the second definition (CSFVI)
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
neural network architecture and pass it to the function. though here we
just used the default architecture that is convolutional with two layers
with 128 and 64 filters in the layers respectively.

``` r
ATT = DNNCausal::aipw.att(Y=Observed_outcomes, T=Treatment, X_t=Timeseries_covariates,X = scalar_covariates, verbose=FALSE, epochs = c(64,32), batch_size = 500)
```

For running this model the following hyperparameters have been chosen.
batch size is 500 and the number of epochs is 64 for the fitting outcome
model. the number of epochs for training propensity score in this study
is chosen to be 32.

For each group en and women, and for each outcome years after treatment
one to five, a similar code has been utilized to estimate the parameter
of interest in that case.
