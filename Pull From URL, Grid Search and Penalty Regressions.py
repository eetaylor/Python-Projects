
# coding: utf-8

# In[24]:


#Import Packages
import matplotlib.pyplot as plt
import numpy as np
import os as os
from sklearn.metrics import mean_squared_error
import requests
import pandas as pd

#Pull in URL Data
r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data")


# In[25]:


#Splitting the data for every variable 
lines = r.text.split("\r\n")


# In[26]:


#Making a list for each item, and replacing the ? with Nulls 
parsedLines = []
for line in lines:
    parsedLines.append(line.replace('?', 'NaN').split(","))
    


# In[27]:


#Pulling in the Column Headers and combining them in a DataFrame with the Data
r2 = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names")
arffText = r2.text[r2.text.find("@attribute"):r2.text.rfind("@data")].strip().replace('@attribute ', '')
colHeaders = arffText.split("\n")
print(colHeaders)

df = pd.DataFrame(parsedLines, columns=colHeaders, dtype=float)
df.head(10)
df.dtypes


# In[28]:


# designate target variable name
targetName = 'ViolentCrimesPerPop numeric'
targetSeries = df[targetName]
#remove target from current location and insert in column 0
del df[targetName]
df.insert(0, targetName, targetSeries)
#reprint dataframe and see target is in position 0
df.head(10)


# In[29]:


#Replacing the Null Values with averages. 
df2 = df

df2.fillna((df2.mean()), inplace=True)


# In[30]:


df2.head(10)


# In[31]:


#Dropping columns not needed for predictions 
df3 = df2.drop('communityname string', 1)
df4 = df3.drop('state numeric', 1)


# In[32]:


#Splitting out the test and train data 
df4.target=df4['ViolentCrimesPerPop numeric'] 
df4.features=df4.drop(['ViolentCrimesPerPop numeric'], axis=1)                         
print(df4.target.shape)
print(df4.features.shape)


# In[33]:


#Creating a seperate data set thats normalized,use later to optimize grid searches 
dfZ = pd.DataFrame((df4-df4.mean())/df4.std())


dfZ.target=dfZ['ViolentCrimesPerPop numeric'] 
dfZ.features=dfZ.drop(['ViolentCrimesPerPop numeric'], axis=1)                         
print(dfZ.target.shape)
print(dfZ.features.shape)
dfZ.head(10)


# In[34]:


# Linear Regression Model, Added R2 and Variance Score to find the best model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# fit a linear regression model to the data
model_LR = LinearRegression(normalize=True)
model_LR.fit(df4.features, df4.target)
print(model_LR)
# make predictions
expected_LR = df4.target
predicted_LR = model_LR.predict(df4.features)
# summarize the fit of the model
print("Coef", model_LR.intercept_, model_LR.coef_)
print("MSE", mean_squared_error(expected_LR, predicted_LR))
print("R2 Score", r2_score(expected_LR, predicted_LR))
print ("Explained Variance", explained_variance_score(expected_LR, predicted_LR))


# In[35]:


#Plotting the Model, can see two obvious outliers 
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[36]:


# Ridge Regression Model
from sklearn.linear_model import Ridge

# fit a ridge regression model to the data
model_RG = Ridge(alpha=20)
model_RG.fit(df4.features, df4.target)
print(model_RG)
# make predictions
expected_RG = df4.target
predicted_RG= model_RG.predict(df4.features)
# summarize the fit of the model
print("Coef", model_RG.intercept_, model_RG.coef_)
print("MSE", mean_squared_error(expected_RG, predicted_RG))
print("R2 Score", r2_score(expected_RG, predicted_RG))
print ("Explained Variance", explained_variance_score(expected_RG, predicted_RG))


# In[37]:


#Plot Ridge Regression vs Linear 
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
plt.plot(model_RG.coef_, label='Ridge')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[38]:


# Lasso Regression
from sklearn.linear_model import Lasso

# fit a LASSO model to the data
model_LAS = Lasso(alpha=1)
model_LAS.fit(df4.features, df4.target)
print(model_LAS)
# make predictions
expected_LAS = df4.target
predicted_LAS = model_LAS.predict(df4.features)
# summarize the fit of the model
print("Coef", model_LAS.intercept_,model_LAS.coef_)
print("MSE", mean_squared_error(expected_LAS, predicted_LAS))
print("R2 Score", r2_score(expected_LAS, predicted_LAS))
print ("Explained Variance", explained_variance_score(expected_LAS, predicted_LAS))


# In[39]:


#Plotting Linear, Ridge, and Lasso 
fig, ax = plt.subplots()
plt.plot(model_LR.coef_, label='LR')
plt.plot(model_RG.coef_, label='Ridge')
plt.plot(model_LAS.coef_, label='Lasso')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()


# In[40]:


# ElasticNet Regression
from sklearn.linear_model import ElasticNet

# fit a model to the data
model_EN = ElasticNet(alpha=2)
model_EN.fit(df4.features, df4.target)
print(model_EN)
# make predictions
expected_EN = df4.target
predicted_EN = model_EN.predict(df4.features)
# summarize the fit of the model
print("Coef", model_EN.intercept_, model_EN.coef_)
print("MSE", mean_squared_error(expected_EN, predicted_EN))
print("R2 Score", r2_score(expected_EN, predicted_EN))
print ("Explained Variance", explained_variance_score(expected_EN, predicted_EN))


# In[41]:


#What is the grid search doing? Why would we do this?
#Grid search is a method to perform hyper-parameter optimisation, it is a method to find the best combination of hyper-parameters (an example of an hyper-parameter is the learning rate of the optimiser), for a given model and test dataset. 
#You have several models, each with a different combination of hyper-parameters. 
#Each of these combinations of parameters, which correspond to a single model, can be said to lie on a point of a "grid". 
#The goal is then to train each of these models and evaluate them. You then select the one that performed best.



# In[42]:



from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}

# run grid search
grid_search = GridSearchCV(model_LAS, param_grid=param_grid)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[43]:



from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}


# run grid search
grid_search = GridSearchCV(model_EN, param_grid=param_grid)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[44]:


#Closing in on better alpha 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [2,10, 50, 100, 500]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[45]:


#Closing in on better alpha 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [0, 50, 100, 150, 200]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[46]:


#Closing in on better alpha 
from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [130, 140, 150, 160, 170]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[47]:


#What is Cross Validaiton doing? 
#Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
#The procedure has a single item called k that refers to the number of groups that a given data sample is to be split into. 
#When a specific value for k is chosen, such as k=10 becoming 10-fold cross-validation.


# In[48]:


#ADD Cross Validation to Grid Score

from sklearn.model_selection import GridSearchCV
param_grid = {"alpha": [130, 140, 150, 160, 170]}

# run grid search
grid_search = GridSearchCV(model_RG, param_grid=param_grid,cv=5)
grid_search.fit(dfZ.features, dfZ.target)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)    
print(grid_search.best_score_)


# In[ ]:


#The model with the best parameters ended up being the simple linear regression with an MSE of 0.016 and 70% explained variance. 
#The ridge model came in second with an MSE of 0.017 and 67% of variance explained. 
#Both the Lasso and Elastic net were awful which could be due to very high colinearity.  
#The ridge penalty will prefer equal weighting of colinear variables while lasso penalty will not be able to choose
#The cross validation and grid searches didn't help the penalty models, it slightly performed worse. 
#I think this is a case where Occam's Razor applies, the simplest model performed the best. 

