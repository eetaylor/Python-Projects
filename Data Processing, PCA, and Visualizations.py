
# coding: utf-8

# In[2]:


#Load Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


path="/Users/emmydoo19/Desktop/BIA6303/"
os.chdir(path)


# In[4]:


#Read In Auto CSV

Auto = pd.read_csv("AUTO.csv", encoding='latin-1', sep=",", header=0)
Auto.head()
Auto.shape

Auto=Auto.rename(columns = {'Weight (lbs)':'Weight'})
Auto=Auto.rename(columns = {'Mileage (mpg)':'MPG'})
Auto['Weight'] = Auto['Weight'].str.replace(',', '')
Auto["Weight"] = Auto.Weight.astype(float)


# In[5]:


#Replace Nulls with Median
Median=Auto.median()
Auto = Auto.fillna(Median)
Auto.dtypes


# In[6]:


#Characteristics of Data
Auto.mean()


# In[7]:


Auto.std()


# In[8]:


Auto.groupby('Name').size()


# In[9]:


Auto.groupby('Drive Type').size()


# In[10]:


Auto["Fuel Type"] = Auto["Fuel Type"].str.strip()
Auto.groupby('Fuel Type').size()


# In[11]:


# Z Score on Weight

from scipy.stats import zscore

ZWeight = pd.DataFrame((Auto['Weight']-Auto['Weight'].mean())/Auto['Weight'].std())
ZWeight.rename(columns={'Weight': 'ZWeight'}, inplace=True)
ZWeight.head(10)


AutoZ = pd.concat([Auto, ZWeight], axis=1, join_axes=[Auto.index])
AutoZ.head(10)
AutoZ[['Weight','ZWeight']]



# In[12]:


#Dummy Out Categorical Variables
# perform data transformation. Creates dummies of any categorical feature
for col in AutoZ.columns[1:]:
	attName = col
	dType = AutoZ[col].dtype
	missing = pd.isnull(AutoZ[col]).any()
	uniqueCount = len(AutoZ[attName].value_counts(normalize=False))
	# discretize (create dummies)
	if dType == object:
		AutoZ = pd.concat([AutoZ, pd.get_dummies(AutoZ[col], prefix=col)], axis=1)
		del AutoZ[attName]
  
        
AutoZ.head(10)


# In[13]:


#Correlation

AutoZ.corr()



# In[14]:


#Correlation Plot
plt.matshow(AutoZ.corr())


# In[18]:


#PCA 
from sklearn.decomposition import PCA
AutoPCA= AutoZ.drop(['Name'], axis=1)
AutoPCA.head(10)
pca = PCA(n_components=3)
pca.fit(AutoPCA)

PCA = pd.DataFrame(pca.components_,columns=AutoPCA.columns,index = ['PC-1','PC-2','PC-3'])
PCA
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())
#3 Pricipal components capture 99.980% of the variance 


# In[19]:


#Scatter Plot
plt.scatter(AutoZ.Weight,AutoZ.MPG)
plt.title('Weight to MPG')
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.xlim(0, 10000)
plt.show()


# In[20]:


#Histogram for Luggage

AutoZ=AutoZ.rename(columns = {'Luggage (cu. ft.)':'Luggage'})
n, bins, patches = plt.hist(AutoZ.Luggage, 6, facecolor='green', alpha=0.75)
plt.ylim(0,200)
plt.title('Luggage using 6 bins')
plt.xlabel("Luggage")
plt.ylabel("Frequency")
plt.show()


# In[21]:


#Histogram for MPG 
n, bins, patches = plt.hist(AutoZ.MPG, 7, facecolor='red', alpha=0.75)
plt.ylim(0,200)
plt.title('MPG using 6 bins')
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.show()


# In[22]:


#Cross Tabs
Auto=Auto.rename(columns = {'Fuel Type':'FuelType'})
Auto=Auto.rename(columns = {'Drive Type':'DriveType'})

Auto_cross=pd.crosstab(Auto.FuelType, Auto.DriveType,colnames=['Drive Type'],rownames=['Fuel Type'])

print(Auto_cross)


# In[23]:


#Stacked Bar Chart for Drive and Fuel Type
Auto_cross.plot(kind='bar', stacked=True)
plt.title('Drive Type by Fuel Type')


# In[24]:


#Sub Query for Regular fuel type that get over 21 MPG
AutoZ=AutoZ.rename(columns = {'Fuel Type_Regular':'FuelType_Regular'})
Auto_sub=AutoZ.query('MPG > 21' and 'FuelType_Regular == 1') 
Auto_sub

