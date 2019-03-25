
# coding: utf-8

# In[109]:


#T1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import os
path = "/Users/emmydoo19/Desktop/BIA 6304/" #where to get/put files
os.chdir=(path)
pd.set_option('display.max_colwidth', 15000) #important for getting all the text

#Pulling in Spam Data Frame, needed updated encoding 
Spamdf = pd.read_csv(path + "spam2.csv", encoding='latin-1') 

Spamdf.head()

#Removed Uneeded Columns 
Spamdf = Spamdf.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
Spamdf = Spamdf.rename(columns={"SMS Class":"label", "Text":"text"})

Spamdf.tail()

Spamdf.label.value_counts()

# convert label to a dummy variable
Spamdf['label_dummy'] = Spamdf.label.map({'ham':0, 'spam':1})

type(Spamdf)



# In[110]:


#T1
#Split Test Train Sets


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(Spamdf["text"],Spamdf["label"], test_size = 0.2, random_state = 10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Creating Bag of words to run through models

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
vect.fit(X_test)


print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)


print(X_train_df.shape)


 


# In[111]:


#T1
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# fit a Decision Tree model to the data
model = DecisionTreeClassifier(random_state = 42)
print(model)
model.fit(X_train_df, y_train)

# make predictions
clf1_expected = y_test
clf1_predicted = model.predict(X_test_df)

print(model.score(X_test_df, y_test))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf1_expected, clf1_predicted)))
print(metrics.classification_report(clf1_expected, clf1_predicted))


# In[112]:


#T1
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# fit a Naive Bayes model to the data
model = MultinomialNB()
print(model)
model.fit(X_train_df, y_train)

# make predictions
clf2_expected = y_test
clf2_predicted = model.predict(X_test_df)

print(model.score(X_test_df, y_test))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf2_expected, clf2_predicted)))
print(metrics.classification_report(clf2_expected, clf2_predicted))


# In[113]:


#T1
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# fit a logistic regression model to the data
model = LogisticRegression(random_state = 42)
print(model)
model.fit(X_train_df, y_train)

# make predictions
clf3_expected = y_test
clf3_predicted = model.predict(X_test_df)

print(model.score(X_test_df, y_test))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf3_expected, clf3_predicted)))
print(metrics.classification_report(clf3_expected, clf3_predicted))



# In[114]:


#T2 Stemming Text
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() 

def stem_text(row):
    text = str(row).split() #splits the text apart before stemming
    stemtext = [ps.stem(word) for word in text] #tells it which stemmer to apply and how
    stem2text = ' '.join(stemtext) #puts everything back together again
    return stem2text

#Checking to make sure it stemmed
Spamdf['stemmed'] = Spamdf["text"].apply(lambda x: stem_text(x)) #apply the above function to our text
print( Spamdf.text[0:1])
print("~~~~~~~~~~~~~~~~~~~")
print(Spamdf.stemmed[0:1])


#Train test splits
from sklearn.model_selection import train_test_split
X_train2,X_test2,y_train2,y_test2 = train_test_split(Spamdf["stemmed"],Spamdf["label"], test_size = 0.2, random_state = 10)
print(X_train2.shape)
print(X_test2.shape)
print(y_train2.shape)
print(y_test2.shape)

#Creating Bag of words to run through models

from sklearn.feature_extraction.text import CountVectorizer
vect2 = CountVectorizer()
vect2.fit(X_train2)
vect2.fit(X_test2)


print(vect2.get_feature_names()[0:20])
print(vect2.get_feature_names()[-20:])

X_train_df2 = vect2.transform(X_train2)
X_test_df2 = vect2.transform(X_test2)


print(X_train_df2.shape)


# In[115]:


#T2 Stemmed Desicion Tree

# fit a Decision Tree model to the data
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(random_state = 42)

model2.fit(X_train_df2, y_train2)

# make predictions
clf2_expected = y_test2
clf2_predicted = model2.predict(X_test_df2)

print(model2.score(X_test_df2, y_test2))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf2_expected, clf2_predicted)))
print(metrics.classification_report(clf2_expected, clf2_predicted))


# In[116]:


#T2 Stemmed Naive bayes
from sklearn.naive_bayes import MultinomialNB

# fit a Naive Bayes model to the data
model2 = MultinomialNB()
print(model2)
model2.fit(X_train_df2, y_train2)

# make predictions
clf2_expected = y_test2
clf2_predicted = model2.predict(X_test_df2)

print(model2.score(X_test_df2, y_test2))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf2_expected, clf2_predicted)))
print(metrics.classification_report(clf2_expected, clf2_predicted))


# In[117]:


#T2 Stemmed Logistic Regression
from sklearn.linear_model import LogisticRegression

# fit a logistic regression model to the data
model = LogisticRegression(random_state = 42)
print(model)
model.fit(X_train_df2, y_train2)

# make predictions
clf3_expected = y_test2
clf3_predicted = model.predict(X_test_df2)

print(model.score(X_test_df2, y_test2))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf3_expected, clf3_predicted)))
print(metrics.classification_report(clf3_expected, clf3_predicted))


# In[118]:


#T2 Apply lemmetazation and TDIF to text 
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

#Checked to make sure it lemmatized
Spamdf['lemmatize']=[wnl.lemmatize(word) for word in Spamdf["text"]]
print( Spamdf.text[0:1])
print("~~~~~~~~~~~~~~~~~~~")
print(Spamdf.lemmatize[0:1])


from sklearn.model_selection import train_test_split
X_train3,X_test3,y_train3,y_test3 = train_test_split(Spamdf["lemmatize"],Spamdf["label"], test_size = 0.2, random_state = 10)
print(X_train3.shape)
print(X_test3.shape)
print(y_train3.shape)
print(y_test3.shape)

#Creating Bag of words to run through models using weights

from sklearn.feature_extraction.text import TfidfVectorizer
vect3 = TfidfVectorizer()
vect3.fit(X_train3)
vect3.fit(X_test3)


print(vect3.get_feature_names()[0:20])
print(vect3.get_feature_names()[-20:])

X_train_df3 = vect3.transform(X_train3)
X_test_df3 = vect3.transform(X_test3)


print(X_train_df3.shape)


# In[119]:


#T2 Lemmatized and Weights Desicion Tree

# fit a Decision Tree model to the data
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 42)

model.fit(X_train_df3, y_train3)

# make predictions
clf2_expected = y_test3
clf2_predicted = model.predict(X_test_df3)

print(model.score(X_test_df3, y_test3))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf2_expected, clf2_predicted)))
print(metrics.classification_report(clf2_expected, clf2_predicted))


# In[120]:


#T2 Lemmatized and Weights Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# fit a Naive Bayes model to the data
model = MultinomialNB()
print(model)
model.fit(X_train_df3, y_train3)

# make predictions
clf2_expected = y_test3
clf2_predicted = model.predict(X_test_df3)

print(model.score(X_test_df3, y_test3))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf2_expected, clf2_predicted)))
print(metrics.classification_report(clf2_expected, clf2_predicted))


# In[121]:


#T2 Lemmatized and Weights Logistic Regression
from sklearn.linear_model import LogisticRegression

# fit a logistic regression model to the data
model2 = LogisticRegression(random_state = 42)
print(model)
model.fit(X_train_df3, y_train3)

# make predictions
clf3_expected = y_test3
clf3_predicted = model.predict(X_test_df3)

print(model3.score(X_test_df3, y_test3))

# summarize the fit of the model
print("accuracy: " + str(metrics.accuracy_score(clf3_expected, clf3_predicted)))
print(metrics.classification_report(clf3_expected, clf3_predicted))


# In[122]:


#T3


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# instantiate vectorizer
tfidf1 = TfidfVectorizer(lowercase=True, 
                        max_df=0.95, 
                        min_df=0.05)
                        


# fit and transform text
tfidf_dm = tfidf1.fit_transform(Spamdf['text'])


#Make 3 clusters
spam_dm = tfidf_dm.toarray()
My_k = 3
km = KMeans(n_clusters=My_k, init='k-means++', max_iter=100, random_state = 42)
spam_k = km.fit(spam_dm)
clusters = km.labels_.tolist()
Spamdf['clusters'] = clusters

print(Spamdf['clusters'].value_counts())
Spamdf.head()



# In[123]:


#T3
#Trying to find best K
k_range = range(1,20)
k_means_set = [KMeans(n_clusters=k,init='k-means++', max_iter=100, random_state = 42).fit(spam_dm) for k in k_range]
centroids_list = [km_result.cluster_centers_ for km_result in k_means_set]

# calc euclidean dist from each point to each cluster center
from scipy.spatial.distance import cdist, pdist

k_euclid = [cdist(spam_dm, thing, 'euclidean') for thing in centroids_list]
distance_set = [np.min(k_euc, axis=1) for k_euc in k_euclid]

# total within-cluster sum of squares
wcss = [np.sum(distance**2) for distance in distance_set]

# total sum of squares
tss  = np.sum(pdist(spam_dm)**2) / spam_dm.shape[0]

# between cluster sum of squares
bss = tss - wcss

# plot elbow chart


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, '^-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('K n_clusters')
plt.ylabel('% Variance Explained')
plt.title('% Var Explained vs K')


# In[124]:


#T3
# create clusters with new K


spam_dm = tfidf_dm.toarray()
My_k = 15
km = KMeans(n_clusters=My_k, init='k-means++', max_iter=100, random_state = 42)
spam_k = km.fit(spam_dm)
clusters = km.labels_.tolist()
Spamdf['clusters'] = clusters

print(Spamdf['clusters'].value_counts())
Spamdf


# In[125]:


#T3
from sklearn.feature_extraction.text import CountVectorizer

#Finding most common words in clusters 
print(cv8_Spam.shape)

num_clusters = 15

for i in range(num_clusters):
    cv8 = CountVectorizer(binary=False, min_df = .005, stop_words = "english") #define the transformation
    cv8_Spam = cv8.fit_transform(Spamdf.loc[Spamdf['clusters'] == i, 'text']) #apply the transformation
    
    names = cv8.get_feature_names()   #create list of feature names
    count = np.sum(cv8_Spam.toarray(), axis = 0) # add up feature counts 
    count2 = count.tolist()  # convert numpy array to list
    count_df = pd.DataFrame(count2, index = names, columns = ['count']) # create a dataframe from the list
    sorted_count_df = count_df.sort_values(['count'], ascending = False)[0:3]

    print('Cluster: %i,' % i)
    print(sorted_count_df)
    print()


