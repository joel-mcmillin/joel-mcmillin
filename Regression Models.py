#!/usr/bin/env python
# coding: utf-8

# ### Working with Regression Models

# In[1]:


#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.datasets import make_classification
import seaborn as sns #I found this which was great for making a heat map!
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math 


# In[2]:


# Load the automobile dataset
df = pd.read_csv('auto-mpg.csv')
df.head() 


# In[3]:


# To make the variables easier to work with later, will rename columns with spaces between words
df = df.rename(columns={'model year': 'model_year', 'car name': 'car_name'})


# In[4]:


# Data preparation
# Removing the 'car name' column to avoid modeling issues
# Also attempted dummying the variable but there are too many makes/models to 
# be useful in modeling. Dropping the 'car_name' variable

#pd.get_dummies(df['car_name'])

df.drop('car_name', axis=1, inplace=True)


# In[5]:


# The horsepower column values may have imported as string data type. 
# First, confirming what the data types are for the df
# Confirmed horspower is 'object' type, shown as string type
df.dtypes


# In[6]:


# Checking unique values in the 'horsepower' column helps recognize which values
# needing removal or replacement
print(df['horsepower'].unique())


# In[7]:


# Unique value search shows '?' as a non-numeric value in the 'horsepower' column
# To address this, I replace the '?' values with column mean
df['horsepower'] = df['horsepower'].replace('?', 
    np.mean(pd.to_numeric (df['horsepower'], errors='coerce')))

df.head()


# In[8]:


pd.get_dummies(df['origin'])


# In[9]:


# Creating a correlation coefficient matrix and visualization. 
# Checking for correlation
df.corr()


# In[10]:


# Visualizing the data with a heatmap using Seaborn
# Early observations show negative correlations between MPG and cylinders, displacement and weight
# While not extremely strong, it also appears model year and vehicle origin are correlated
# Cylinders, weight and displacement are highly correlated with each other
sns.heatmap(df.corr(), annot=True, fmt='.2g', cmap='coolwarm')


# In[11]:


# Statistical information about our data
display(df.describe())


# In[12]:


# Plotting 'MPG' vs 'weight'. The Scatter Plot shows that higher weight vehicles will 
# usually have worse fuel economy as measured by MPG
plt.scatter(df.mpg, df.weight)
plt.show()


# In[13]:


# 'Mpg' for different vehicles shows a strong right skew with most vehicles
# having 'mpg' on the lower end
df.hist(column='mpg')


# In[14]:


# Plotting 'MPG' vs 'acceleration'. The Scatter Plot shows that acceleration and MPG 
# appear to have some correlation at lower values, becoming increasingly more random
plt.scatter(df.mpg, df.acceleration)
plt.show()


# In[15]:


# 'Acceleration' for different vehicles has pretty even distribution with most car
# being centered around 15.0
df.hist(column='acceleration')


# #### Linear Regression Model

# In[16]:


# Randomly split the data into 80% training data and 20% test data with MPG as the target variable
y = df.mpg  
X = df.drop('mpg', axis=1)


# In[17]:


# Setting up the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)


# In[18]:


# Confirming training data is as expected
X_train


# In[19]:


# MPG information (the target variable) split between training and testing sets
y_train
y_test


# In[20]:


# Training an ordinary linear regression model on the data
regression = LinearRegression()


# In[21]:


#Fitting the linear regression...
model = regression.fit(X_train, y_train)


# In[22]:


# Model intercept - the mean value of the response variable at point where all predictors equal 0
model.intercept_


# In[23]:


# Coefficient - amount the mean of the dependent variable changes with a shift of 1 unit in a particular 
# independent variable with others at a constant value
model.coef_


# In[25]:


linear_pred = model.predict(X_test)


# In[27]:


# Calculate R2, RMSE, and MAE on both the training and test sets 
# The results appear to show test and training scores are close in predictive ability.
# For ex. the R-Squared score at 84% indicates predictive ability but still
# not completely accurate

print('Train Score: {}\n'.format (model.score(X_train, y_train)))
# score the model on the test set
print('Test Score: {}\n'.format (model.score(X_test, y_test)))
# calculate the overall accuracy of the model
print('R^2/Overall Model Accuracy: {}\n'.format (r2_score (y_test, linear_pred)))
# compute the mean squared error of the model
print('Mean Squared Error: {}\n'.format (mean_squared_error (y_test, linear_pred)))
# compute rmse
print('Root Mean Squared Error: {}'.format(math.sqrt(mean_squared_error(y_test, linear_pred))))


# #### Lasso Regression Model

# In[28]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print ("Lasso model:", (lasso.coef_))


# In[29]:


# The Lasso model gives similar results to the Linear Regression model, with the
# R-Squared score also roughly 84% - Both models here have similar predictive reliability
pred_train_lasso = lasso.predict(X_train)
print(np.sqrt(mean_squared_error(y_train, pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso = lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))


# In[ ]:




