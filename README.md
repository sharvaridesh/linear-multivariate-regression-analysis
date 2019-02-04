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

- A simple linear regression is carried out to estimate the parameters of the model:
Y = a0 + a1X1 + ε.

- The estimates of coefficients were calculated using statsmodels package. The estimated values of coefficients are shown in the figure below.

![](https://github.com/sharvaridesh/Linear-Multivariate-Regression-Analysis/blob/master/results/Y_hist.png) |
|:--:|
|*Histogram of Y*|





