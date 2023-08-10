#!/usr/bin/env python
# coding: utf-8

# ### Using a Pipeline to determine best model for loan data

# In[1]:


# Import libraries

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Import the data and review the loan data

df = pd.read_csv('Loan_Train.csv')
df.head()


# In[3]:


# Data preparation prior to modeling by performing the following steps:
# I will drop the column “Load_ID”, columns with any rows with missing data, and I will convert 
# the categorical features into dummy variables

df.drop('Loan_ID', axis=1, inplace=True)
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.head()


# In[4]:


# Spliting the data into a training and test set, where the “Loan_Status” column is the target variable
# Test size will be 20% and Training size 80%

X = df.drop(columns=['Loan_Status_Y'])
y = df['Loan_Status_Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Creating a pipeline with a min-max scaler and a KNN classifier

pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())])
pipe.fit(X_train, y_train)


# In[6]:


# Fitting a default KNN classifier to the data with this pipeline and displaying model accuracy on the test set - 

pipe.score(X_test, y_test)


# In[7]:


# Creating a search space for the KNN classifier where the “knn__n_neighbors” parameter varies from 1 to 10 - 

pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())])
param_grid = {'knn__n_neighbors': range(1, 11)}


# In[8]:


# Next I am fitting a grid search with the pipeline, search space, and a 5-fold cross-validation 
# to find the best value for the “knn__n_neighbors” parameter - Shows us that 3 folds is the best fit

classifier = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
classifier.best_params_


# In[9]:


# Now, to find the accuracy of the grid search best model on the test set. 
# Note: It is possible that this will not be an improvement over the default model, but likely it will be.

classifier.best_score_

# (As noted, this is not an improvement over the default model - it's actually 4% lower (drop from 78& to 74%))


# In[10]:


# Then, I repeat steps 6 and 7 with the same pipeline, but expand my search space to include logistic 
# regression and random forest models with hyperparameter values 

pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', KNeighborsClassifier())])

search_space = [{
    "classifier": [LogisticRegression()],
    "classifier__penalty": ['l2'],
    "classifier__C": np.logspace(0, 4, 10)
}, {
    "classifier": [RandomForestClassifier()],
    "classifier__n_estimators": [10, 100, 1000],
    "classifier__max_features": [1, 2, 3]
}, {
    "classifier": [KNeighborsClassifier()],
    "classifier__n_neighbors": range(1, 11),
    "classifier__weights": ['uniform', 'distance']
}]
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
gridsearch.fit(X_train, y_train)


# In[13]:


# Here we will find the best model and hyperparameters found from the grid search.
# Also, checking the accuracy of this model on the test set! 

best_model = gridsearch.fit(X_test, y_test)
best_model.best_params_

# This output indicates that the RandomForestClassifier is the best model, and the best
# hyperparameters are max features of 3 with 1000 as the estimator


# In[14]:


best_model.best_score_

#Accuracy of this is 81%, which is better than the previous scores found


# ### Results Summary:
# 
# #### The results showed that the original model had a 78% accuracy score, while the accuracy fo the grid model was less, at 74%. Finally, the results of the last model, where we were actively looking for best fit  using parameters and hyperparameters showed that by using RandomForestClassifier model with the hyperparameters of 3 max features with 1000 as estimator we get an accuracy of 81% which is better than either of the other models tested.

# In[ ]:




