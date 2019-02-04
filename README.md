# Linear-Multivariate-Regression-Analysis

## TASK 1

- The mean and variance of all 4 independent variables, X1, X2, X3, X4, and, X5, and of the dependent variable Y are calculated. 
- Their histograms are also plotted to check if they follow normal distribution. 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/mean_variance_variables.PNG) |
|:--:|
|*Mean and Variance of Variables*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X1_hist.png) 
|:--:|
|*Histogram of X1*| 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X2_hist.png) |
|:--:|
|*Histogram of X2*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X3_hist.png) |
|:--:|
|*Histogram of X3*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X4_hist.png) |
|:--:|
|*Histogram of X4*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X5_hist.png) |
|:--:|
|*Histogram of X5*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/Y_hist.png) |
|:--:|
|*Histogram of Y*|

- The inter-quartile range scores (IQR scores) of all variables are used to obtain outliers. 
- Post this, outliers are removed from the main dataframe. 
- After the outlier removal, the dimensions of the dataframe are modified from 300x6 to 275x6. 

# TASK 2

## Simple Linear Regression 

### Statistical Analysis

- A simple linear regression is carried out to estimate the parameters of the model:
Y = a0 + a1X1 + ε.

- The estimates of coefficients were calculated using statsmodels package. The estimated values of coefficients are shown in the figure below.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/ols-simple.JPG) |
|:--:|
|*OLS Regression Results*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/p-value-simple.JPG) |
|:--:|
|*p-value and variance estimate*|

As observed, 

- p-value is almost zero.
- R^2 value is 0.562.
- F-statistic is 382.6.

Considering all these, it is safe to say that the null hypothesis is rejected, and, regression coefficients are significant. 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/Linear%20Regression%20Simple.png) |
|:--:|
|*Regression Line*|

Since the data is not linear, a linear fit isn’t ideal to use for this dataset.

### Residual Analysis

- Q-Q Plot
![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/qqplot.png) |
|:--:|
|*Q-Q Plot*|

From the Q-Q plot it is observed that,
  -(Lower quantiles on x-axis) < (Lower quantiles on y-axis). Hence, left tail of this distribution is shorter as compared to the            standard normal distribution.
  -(Higher quantiles on x-axis) < (Higher quantiles on y-axis). Hence, right tail of this distribution is shorter as compared to            the normal distribution.

- Histogram of Residual 

From the figure it is observed that the residual’s distributions’ left tail is shorter and right tail is also shorter than the standard normal distribution.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/error_hist.png) |
|:--:|
|*Histogram of Residuals*|

- Scatter Plot

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/scatter%20plot.png) |
|:--:|
|*Scatter Plot*|

It is observed that, the variance of data is increasing significantly along the Y-axis. This plot pattern is non-random (U-shaped), suggesting a better fit for a non-linear model.

## Higher Order Regression

### Statistical Analysis

- A higher linear regression is carried out to estimate the parameters of the model.

- The estimates of coefficients were calculated using statsmodels package. The estimated values of coefficients are shown in the figure below.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/ols-high.JPG) |
|:--:|
|*OLS Regression Results*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/p-value-high.JPG) |
|:--:|
|*p-value and variance estimate*|

As observed, 

- The p-values are approximately zero for intercept variable and predictor variable 𝑎2. This means that these variables are significant.
p-value of 𝑎1 is greater than 𝛼=0.05 (𝑎𝑠𝑠𝑢𝑚𝑝𝑡𝑖𝑜𝑛), hence it does not have a significant value.
- R^2 value is 0.953.
  - Theoretically, a good value for R^2 is when it is close to 1. 
  - Most times, the higher this value, better is the fit of the model. 
  - From the observed value, it can be said that, the fit of the model is a better fit than the simple regression or multivariate           regression model.
- F-statistic is 382.6.
  - F-statistic is large.
  - The F-test indicates that:
    Ho: The fit of the intercept only model and the user model are equal.
    Ha: The fit of the user model is significantly better than the intercept only model.
   - F-test can also be used for a different test:
      Ho: 𝑎0=𝑎1=𝑎2=0
      Ha: Atleast one predictor variable is ≠ 0
   - Here, there are two predictor variables, i.e. 𝑎0 𝑎𝑛𝑑,𝑎2 whose values is not equal to zero (as observed from p-value analysis).
   - It can be said that the fit of the user model is better than the intercept only model.

Hence, null hypothesis is rejected, and, regression coefficients are significant. .

- Regression Plot

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/higher order_regression%20fit.png) |
|:--:|
|*Regression Line*|

This model perfectly fits the original data. It signifies that original data is non-linear and has a parabolic curve. Hence, the best fit for this is a non-linear model.

### Residual Analysis

- Q-Q Plot
![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/higher%20order_qqplot.png) |
|:--:|
|*Q-Q Plot*|

- Histogram of Residual 

From the figure it is observed that the residual’s distributions’ left tail is shorter and right tail is also shorter than the standard normal distribution.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/error_hist_highorder.png) |
|:--:|
|*Histogram of Residuals*|

It's observed from both Q-Q plot and histogram of residuals, that the original distribution closely follows the standard normal distribution.

- Scatter Plot

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/scatter%20plot_higher%20order.png) |
|:--:|
|*Scatter Plot*|

From the scatter plot of residuals, as seen in Fig. 33, it is observed,
- Residuals Have No Trends
- Residuals Are Normally Distributed
- Residuals Have Constant Variance

# TASK 3

## Multivariable Linear Regression

### Statistical Analysis

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/ols-multiple.JPG) |
|:--:|
|*OLS Regression Results*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/p-value-multi.JPG) |
|:--:|
|*p-value and variance estimate*|

- As observed, the values of predictor variables, 𝑎1 𝑎𝑛𝑑 𝑎4, are significant as their p-values are approximately zero. 
- The values of predictor variables, 𝑎0,𝑎2,𝑎3, and 𝑎5, are not significant as their p-values are greater than the common alpha level of 0.001. 
- Therefore, in the model above, we should consider removing variables X2, X3, X5 and the constant.

After removing the variables, 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/ols-removal.JPG) |
|:--:|
|*OLS Regression Results*|

- R^2 = 0.961
- Theoretically, a good value for 𝑹𝟐 is when it is close to 1. The higher this value, better is the fit of the model. From the observed value, it can be said that, the fit of the model is a better fit than the model with all variables included.
- F-statistic is large. Therefore, it can be said that the fit of the user model is better than the intercept only model.

Hence, null hypothesis is rejected.

### Residual Analysis

- Q-Q Plot
![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/qqplot_multiple.png) |
|:--:|
|*Q-Q Plot*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/qqplot_multiple_mod.png) |
|:--:|
|*Q-Q Plot after removal of variables*|

- Histogram of Residual 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/error_hist_multiple.png) |
|:--:|
|*Histogram of Residuals*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/mod_hist_task3.png) |
|:--:|
|*Histogram of Residuals after removal of variables*|

It is observed that after removal of variables, the original distribution follows the standard normal distribution more closely.

- Scatter Plot

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/scatter%20plot%20multi.png) |
|:--:|
|*Scatter Plot*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/scatter%20plot%20mod.png) |
|:--:|
|*Scatter Plot after removal of variables*|

As seen from the scatter plots, after removal of certain independent variables, the plot has a better shape as compared to the Multivariable Polynomial Regression containing all independent variables. This plot pattern is non-random (U-shaped), suggesting a better fit for a non-linear model.

# Conclusion

Higher Order Regression model is the best fit for the given dataset. 
