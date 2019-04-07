
# coding: utf-8

# In[1]:


#Pull in Packages
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#From Scikit Learn
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


# In[2]:


#Change Directory
import os
path="/Users/emmydoo19/Desktop/BIA6303/"
os.chdir(path)


# In[3]:


#Pulling in Churn Data 
Churn=pd.read_csv("Churn_Calls.csv")
Churn.head(10)


# In[4]:


# designate target variable name
targetName = 'churn'
targetSeries = Churn[targetName]
#remove target from current location and insert in column 0
del Churn[targetName]
Churn.insert(0, targetName, targetSeries)
#reprint dataframe and see target is in position 0
Churn.head(10)
Churn.shape


# In[5]:


#EDA 
Churn.info()

#Checking to see the data types, colums, and shape of the data. 


# In[6]:


Churn.describe()
#Checking the Count, Mean, St Dev, Min, Max, and Quartiles. 
#Good quick info to grab and decide if I want to normalize or not. 


# In[7]:


Churn.dtypes

#Need to be coginizant of the fact I'm working with Int, Float, and Objects. 
#Will have to do some transformation on the ojects to run through any models.


# In[8]:


groupby = Churn.groupby(targetName)
targetEDA=groupby[targetName].aggregate(len)
plt.figure()
targetEDA.plot(kind='bar', grid=False)
plt.axhline(0, color='k')

#Did a quick plot to visualize the data, can see its an unbalanced data set and only 15% of the time does churn happen.
#If the model doesnt do better than 0.85% accuracy the we are better off guessing. 


# In[9]:


#Dummy Variables and replacing Null Values 

from sklearn import preprocessing
le_dep = preprocessing.LabelEncoder()
Churn['churn'] = le_dep.fit_transform(Churn['churn'])

for col in Churn.columns[1:]:
	attName = col
	dType = Churn[col].dtype
	missing = pd.isnull(Churn[col]).any()
	# discretize (create dummies)
	if dType == object:
		Churn = pd.concat([Churn, pd.get_dummies(Churn[col], prefix=col)], axis=1)
		del Churn[attName]

        
Median=Churn.median()
Churn = Churn.fillna(Median)

Churn.shape
Churn.head(10)


# In[10]:


#Train Test Split

features_train, features_test, target_train, target_test = train_test_split(
    Churn.iloc[:,1:].values, Churn.iloc[:,0].values, test_size=0.30, random_state=0)


# In[11]:


#check shape of the new data split out 

print(features_test.shape)
print(features_train.shape)
print(target_test.shape)
print(target_train.shape)


# In[12]:


#Decision Tree train model. 

from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier()

#Fit clf to the training data
clf_dt = clf_dt.fit(features_train, target_train)
#Predict clf DT model again test data
target_predicted_dt = clf_dt.predict(features_test)


# In[13]:


#Confusion Matrix, Accuracy Score, and other metrics 
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))
print(classification_report(target_test, target_predicted_dt))
print(confusion_matrix(target_test, target_predicted_dt))


# In[14]:


#Change Gini to Entropy in Desicion Tree 

from sklearn import tree 
clf_dt2 = tree.DecisionTreeClassifier(criterion='entropy')
clf_dt2 = clf_dt2.fit(features_train, target_train)
target_predicted_dt2 = clf_dt2.predict(features_test)


# In[15]:


#Confusion Matrix, Accuracy Score, and other metrics 
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt2))
print(classification_report(target_test, target_predicted_dt2))
print(confusion_matrix(target_test, target_predicted_dt2))


# In[16]:


#Change Splitter from best to random best 

from sklearn import tree 
clf_dt3 = tree.DecisionTreeClassifier(splitter='random')
clf_dt3 = clf_dt3.fit(features_train, target_train)
target_predicted_dt3 = clf_dt3.predict(features_test)


# In[17]:


#Confusion Matrix, Accuracy Score, and other metrics 
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt3))
print(classification_report(target_test, target_predicted_dt3))
print(confusion_matrix(target_test, target_predicted_dt3))


# In[18]:


#Cross Validation for original desicion tree 
scores = cross_val_score(clf_dt, features_train, target_train, cv=10)
print("Cross Validation Score for each K",scores)
scores.mean()  


# In[19]:


#Knn Model 

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(features_train, target_train)
predicted_model = model.predict(features_test)


# In[20]:


print("KNN Accuracy Score", accuracy_score(target_test, predicted_model))
print(classification_report(target_test, predicted_model))
print(confusion_matrix(target_test, predicted_model))


# In[26]:


#Cross Validation KNN
KNNscores = cross_val_score(model, features_train, target_train, cv=10)
print("Cross Validation Score for each K",scores)
KNNscores.mean()  

#KNN Scores stayed about the same as the orginal model. 
#Probably isnt the best model to run for this data set if both run pretty poorly with mariginal changes and CV


# In[29]:


#RF Model 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 500)
rf.fit(features_train, target_train)
# test random forest model
target_predicted_rf = rf.predict(features_test)
#print accuracy_score(target_test, target_predicted_rf)
target_names = [" No", "Yes"]
print(classification_report(target_test, target_predicted_rf, target_names=target_names))
print("RF Accuracy", accuracy_score(target_test, target_predicted_rf))


# In[31]:


#Cross Validation for RF Forest Model 
#Cross Validation RF
RFscores = cross_val_score(rf, features_train, target_train, cv=10)
print("Cross Validation Score for RF",scores)
RFscores.mean() 

