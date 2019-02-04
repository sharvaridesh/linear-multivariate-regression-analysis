#!/usr/bin/env python
# File: regression.py
# Author: Sharvari Deshpande <shdeshpa@ncsu.edu>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import pylab

#------------------------------------------Reading CSV File------------------------------------------#
df = pd.read_csv('shdeshpa.csv',header=None)
print(df)

#----------------------------------------------TASK 1-------------------------------------------------#

#Mean Calculation
x1mean = df[0].mean()
print('X1_mean:', x1mean)
x2mean = df[1].mean()
print('X2_mean:', x2mean)
x3mean = df[2].mean()
print('X3_mean:', x3mean)
x4mean = df[3].mean()
print('X4_mean:', x4mean)
x5mean = df[4].mean()
print('X5_mean:', x5mean)

#Variance Calculation
x1var = df[0].var()
print('X1_variance:', x1var)
x2var = df[1].var()
print('X2_variance:', x2var)
x3var = df[2].var()
print('X3_variance:', x3var)
x4var = df[3].var()
print('X4_variance:', x4var)
x5var = df[4].var()
print('X5_variance:', x5var)

#Histogram Plotting
plt.hist(df[0], color=['green'])
plt.xlabel("X1 values")
plt.ylabel("Frequency")
plt.title('Histogram of X1 variable')
plt.show()

plt.hist(df[1], color=['green'])
plt.xlabel("X2 values")
plt.ylabel("Frequency")
plt.title('Histogram of X2 variable')
plt.show()

plt.hist(df[2], color=['green'])
plt.xlabel("X3 values")
plt.ylabel("Frequency")
plt.title('Histogram of X3 variable')
plt.show()

plt.hist(df[3], color=['green'])
plt.xlabel("X4 values")
plt.ylabel("Frequency")
plt.title('Histogram of X4 variable')
plt.show()

plt.hist(df[4], color=['green'])
plt.xlabel("X5 values")
plt.ylabel("Frequency")
plt.title('Histogram of X5 variable')
plt.show()

plt.hist(df[5], color=['green'])
plt.xlabel("Y values")
plt.ylabel("Frequency")
plt.title('Histogram of Y variable')
plt.show()

#Correlation Matrix
print(df.corr())

#Boxplot
plt.boxplot(df[0])
plt.ylabel("X1 values")
plt.title('X1 Boxplot')
plt.show()

plt.boxplot(df[1])
plt.ylabel("X2 values")
plt.title('X2 Boxplot')
plt.show()

plt.boxplot(df[2])
plt.ylabel("X33 Values")
plt.title('X3 Boxplot')
plt.show()

plt.boxplot(df[3])
plt.ylabel("X5 values")
plt.title('X4 Boxplot')
plt.show()

plt.boxplot(df[4])
plt.ylabel("X5 values")
plt.title('X5 Boxplot')
plt.show()

plt.boxplot(df[5])
plt.ylabel("Y values")
plt.title('Y Boxplot')
plt.show()

#IQR Calculation
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

#Removal of Outliers
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out.shape
print(df_out)

#--------------------------------------TASK 2------------------------------------------#
X1 = df[[0]]
Y = df[5]

# with statsmodels
X = sm.add_constant(X1)  # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
# print(predictions)
print_model = model.summary()
print(print_model)
error_estimate = Y - predictions
# print(error_estimate)
res = model.resid

#Histogram of Residuals
plt.hist(error_estimate)
plt.ylabel('Frequency')
plt.xlabel('Residual Values')
plt.title('Histogram of Residual')
plt.show()

#p-value
print("P-Value:", model.pvalues)

#Variance Estimate Calculation
df_error = pd.DataFrame(error_estimate)
var = df_error.var()
print('Variance Estimate:', var)

#Regression Line Plot
plt.scatter(X1, Y)
plt.plot(X1, predictions)
plt.title('Regression Line Plot')
plt.show()

#Q-Q Plot
fig = sm.qqplot(error_estimate, stats.distributions.norm)
stats.probplot(error_estimate, dist="norm",plot=pylab)
pylab.show()

#Chi-squared Test
normal_distribution = stats.norm.pdf(res, np.mean(res), np.std(res))
res = stats.norm.rvs(size=100)
print('Chi Squared Test:', stats.normaltest(res))

#Scatter Plot Residuals
plt.scatter(predictions, error_estimate)
plt.xlabel('Prediction Values')
plt.ylabel('Residual Values')
plt.title('Scatter Plot of Residuals')
plt.show()

#----------------------TASK 2: 2.7-higher order polynomial regression-------------------------------#

X1_square = X1**2
X1_new = sm.add_constant(X1)
X1_new[1] = X1_square
# print(X1_new.shape, X1_square.shape)
model = sm.OLS(Y,X1_new).fit()
print(model.summary())

predictions_high = model.predict(X1_new)
# print(predictions_high)
print_model_high = model.summary()
print(print_model_high)
error_estimate_high = Y - predictions_high
res_high = model.resid

#Histogram of Residuals
plt.hist(error_estimate_high)
plt.ylabel('Frequency')
plt.xlabel('Residual Values')
plt.title('Histogram of Residual')
plt.show()

#p-value calculation
print("P-Value of higher order:", model.pvalues)

#Variance Estimate Calculation
df_error_high = pd.DataFrame(error_estimate_high)
var_high = df_error_high.var()
print('Variance Estimate for a Higher Order Polynomial Regression:', var_high)

#Q-Q plot
fig = sm.qqplot(error_estimate_high, stats.distributions.norm)
stats.probplot(error_estimate_high, dist="norm",plot=pylab)
pylab.show()

#Chi Squared Test
normal_distribution = stats.norm.pdf(res_high, np.mean(res_high), np.std(res_high))
res_high = stats.norm.rvs(size=100)
print('Chi Squared Test:', stats.normaltest(res_high))

#Regression Line Plot
plt.plot(X1, Y, '+',label='original')
plt.plot(X1, predictions_high, '<',label='prediction')
plt.legend(loc='upper left')
plt.title('Higher Order Polynomial Regression')
plt.show()

#Scatter Plot of Residuals
plt.scatter(predictions_high, error_estimate_high)
plt.ylabel('Residual Values')
plt.xlabel('Prediction Values')
plt.title('Scatter Plot of Residuals')
plt.show()

#-----------------------------------------------TASK 3----------------------------------------#

X_multiple = df[[0, 1, 2, 3, 4]]
Y = df[5]

X_constant = sm.add_constant(X_multiple)  # adding a constant
model = sm.OLS(Y, X_constant).fit()
predictions_1 = model.predict(X_constant)
print('Predictions:',predictions_1)
print_model_1 = model.summary()
print(print_model_1)
error_estimate_1 = Y - predictions_1
print('Residual:', error_estimate_1)
res_1 = model.resid

#Variance Estimate Calculation
df_error_1 = pd.DataFrame(error_estimate_1)
# print(df_error)
var_1 = df_error_1.var()
print('Variance Estimate for multivariate regression:', var_1)

#Histogram of Residuals
plt.hist(error_estimate_1)
plt.ylabel('Frequency')
plt.xlabel('Residual Values')
plt.title('Histogram of Residuals')
plt.show()

#p-values calculation
print("P-Value:", model.pvalues)

#Q-Q plot
fig = sm.qqplot(error_estimate_1,stats.distributions.norm)
stats.probplot(error_estimate_1,dist="norm",plot=pylab)
pylab.show()

#Chi Squared Test
normal_distribution = stats.norm.pdf(res_1, np.mean(res_1), np.std(res_1))
res_1 = stats.norm.rvs(size=100)
print('Chi Squared Test:', stats.normaltest(res_1))

#Scatter Plot of Residuals
plt.scatter(predictions_1,error_estimate_1)
plt.xlabel('Prediction Values')
plt.ylabel('Residual Values')
plt.title('Scatter Plot of Residuals')
plt.show()

#--------Modified Task 3 (After Independent Variable Removal)------#

X_mod = df[[0, 3]]
Y = df[5]

model = sm.OLS(Y, X_mod).fit()
predictions_mod = model.predict(X_mod)
print_model_mod = model.summary()
print(print_model_mod)
error_estimate_mod = Y - predictions_mod
res_mod = model.resid

#Histogram of Residuals
plt.hist(error_estimate_mod)
plt.ylabel('Frequency')
plt.xlabel('Residual Values')
plt.title('Histogram of Residuals')
plt.show()

#p-value calculation
print("P-Value:", model.pvalues)

#Variance Estimate Calculation
df_error_mod = pd.DataFrame(error_estimate_mod)
var_mod = df_error_mod.var()
print('Variance Estimate for multivariate regression:', var_mod)

#Q-Q plot
fig = sm.qqplot(error_estimate_mod,stats.distributions.norm)
stats.probplot(error_estimate_mod,dist="norm",plot=pylab)
pylab.show()

#Chi Squared Test
normal_distribution = stats.norm.pdf(res_mod, np.mean(res_mod), np.std(res_mod))
res_mod = stats.norm.rvs(size=100)
print('Chi Squared Test:', stats.normaltest(res_mod))

#Scatter Plot of Residuals
plt.scatter(predictions, error_estimate)
plt.xlabel('Prediction Values')
plt.ylabel('Residual Values')
plt.title('Scatter Plot of Residuals')
plt.show()

