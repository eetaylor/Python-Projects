
# coding: utf-8

# In[1]:


#Pull in Packages
import matplotlib.pyplot as plt
import numpy as np
import os as os
from sklearn.metrics import mean_squared_error
import requests
import pandas as pd

path="/Users/emmydoo19/Desktop/BIA6303/"
os.chdir(path)



# In[2]:


#Pull in Prostate Data 
Prostate=pd.read_csv("prostate.csv")
Prostate.head(10)


# In[3]:


# designate target variable name
targetName = 'lpsa'
targetSeries = Prostate[targetName]
#remove target from current location and insert in column 0
del Prostate[targetName]
Prostate.insert(0, targetName, targetSeries)
#reprint dataframe and see target is in position 0
Prostate.head(10)


# In[4]:


#Filling an null variables with mean
Prostate2 = Prostate.fillna((Prostate.mean()))


# In[5]:


#Checking data types 
Prostate2.head(10)
Prostate2.dtypes


# In[6]:


#Splitting data into target variable and features 
Prostate2.target=Prostate2['lpsa'] 
Prostate2.features=Prostate2.drop(['lpsa'], axis=1)                         
print(Prostate2.target.shape)
print(Prostate2.features.shape)


# In[7]:


#Splitting data and standardizing, may need to use for grid search optimizaiton 
ProstateZ = pd.DataFrame((Prostate2-Prostate2.mean())/Prostate2.std())


ProstateZ.target=ProstateZ['lpsa'] 
ProstateZ.features=ProstateZ.drop(['lpsa'], axis=1)                         
print(ProstateZ.target.shape)
print(ProstateZ.features.shape)
ProstateZ.head(10)


# In[8]:


# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# fit a linear regression model to the data
model_LR = LinearRegression(normalize=True)
model_LR.fit(Prostate2.features, Prostate2.target)
print(model_LR)
# make predictions
expected_LR = Prostate2.target
predicted_LR = model_LR.predict(Prostate2.features)
# summarize the fit of the model
print("Coef", model_LR.intercept_, model_LR.coef_)
print("MSE", mean_squared_error(expected_LR, predicted_LR))
print("R2 Score", r2_score(expected_LR, predicted_LR))
print ("Explained Variance", explained_variance_score(expected_LR, predicted_LR))


# In[9]:


#Plotting the line to see some potential outliers 
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[10]:


# Ridge Regression
from sklearn.linear_model import Ridge

# fit a ridge regression model to the data
model_RG = Ridge(alpha=20)
model_RG.fit(Prostate2.features, Prostate2.target)
print(model_RG)
# make predictions
expected_RG = Prostate2.target
predicted_RG= model_RG.predict(Prostate2.features)
# summarize the fit of the model
print("Coef", model_RG.intercept_, model_RG.coef_)
print("MSE", mean_squared_error(expected_RG, predicted_RG))
print("R2 Score", r2_score(expected_RG, predicted_RG))
print ("Explained Variance", explained_variance_score(expected_RG, predicted_RG))


# In[11]:


#Plotting the Ridge vs the Linear
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
plt.plot(model_RG.coef_, label='Ridge')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[12]:


# Lasso Regression
from sklearn.linear_model import Lasso

# fit a LASSO model to the data
model_LAS = Lasso(alpha=1)
model_LAS.fit(Prostate2.features, Prostate2.target)
print(model_LAS)
# make predictions
expected_LAS = Prostate2.target
predicted_LAS = model_LAS.predict(Prostate2.features)
# summarize the fit of the model
print("Coef", model_LAS.intercept_,model_LAS.coef_)
print("MSE", mean_squared_error(expected_LAS, predicted_LAS))
print("R2 Score", r2_score(expected_LAS, predicted_LAS))
print ("Explained Variance", explained_variance_score(expected_LAS, predicted_LAS))


# In[13]:


#Plotting the Ridge, Linear, and Lasso 
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
plt.plot(model_RG.coef_, label='Ridge')
plt.plot(model_LAS.coef_, label='Lasso')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[14]:


# ElasticNet Regression
from sklearn.linear_model import ElasticNet

# fit a model to the data
model_EN = ElasticNet(alpha=2)
model_EN.fit(Prostate2.features, Prostate2.target)
print(model_EN)
# make predictions
expected_EN = Prostate2.target
predicted_EN = model_EN.predict(Prostate2.features)
# summarize the fit of the model
print("Coef", model_EN.intercept_, model_EN.coef_)
print("MSE", mean_squared_error(expected_EN, predicted_EN))
print("R2 Score", r2_score(expected_EN, predicted_EN))
print ("Explained Variance", explained_variance_score(expected_EN, predicted_EN))


# In[15]:


#Grid searches on Lasso 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}

# run grid search
grid_search = GridSearchCV(model_LAS, param_grid=param_grid)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[16]:


#Grid Searches on Elastic Net 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}


# run grid search
grid_search = GridSearchCV(model_EN, param_grid=param_grid)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[17]:


#Grid Search on Ridge
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[18]:


#Closing in on best Alpha 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [300, 400, 500, 600, 700]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[19]:


#Closing in on best alpha 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [380, 390, 400, 410, 420]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[20]:


#ADD Cross validation to Grid Search 

# use a full grid over several parameters and cross validate 5 times
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [380, 390, 400, 410, 420]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid,cv=5)
grid_search.fit(ProstateZ.features, ProstateZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)

