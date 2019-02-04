# Linear-Multivariate-Regression-Analysis

## TASK 1

- The mean and variance of all 4 independent variables, X1, X2, X3, X4, and, X5, and of the dependent variable Y are calculated. 
- Their histograms are also plotted to check if they follow normal distribution. 

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/mean_variance_variables.PNG) |
|:--:|
|*Mean and Variance of Variables*|

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/X1_hist.png) |
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

- A higher linear regression is carried out to estimate the parameters of the model.

- The estimates of coefficients were calculated using statsmodels package. The estimated values of coefficients are shown in the figure below.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/ols-high.JPG) |
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

# TASK 3



