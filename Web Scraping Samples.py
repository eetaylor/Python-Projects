
# coding: utf-8

# In[1]:


#T1 

#Imported Libraries 
import pandas as pd   
from bs4 import BeautifulSoup
import requests
pd.set_option('display.max_colwidth', 150) 

#Get page from Rotten tomatoes 
page = requests.get('https://www.rottentomatoes.com/')
 
#Create soup object from the page
soup = BeautifulSoup(page.text, "html5lib")


#New list called movies from top box office 
movies = soup.select("#Top-Box-Office .sidebarInTheaterOpening")

print(movies[0])
print()
print("We scraped", len(movies), "movies")


# In[2]:


#Pulling in the time and creating a dictionary 

import time
top_box_office_movies = {}


#Making for loop to go through each movie
for movie in movies:
    
    #Selecting individual parts of movie info 
    movie_details = movie.select('td > a')
    
    #Printed for visualization 
    print('0: ' + movie_details[0].get_text(strip=True) + ' 1: ' + movie_details[1].get_text(strip=True) + ' 2: ' + movie_details[2].get_text(strip=True))
    
    #Setting key to Movie Title and then append movie details to list    
    key = movie_details[1].get_text(strip=True)
    top_box_office_movies.setdefault(key, [])
    top_box_office_movies[key].append(movie_details[0].get_text(strip=True))
    top_box_office_movies[key].append(movie_details[2].get_text(strip=True))
    top_box_office_movies[key].append(time.strftime("%m/%d/%Y"))

#Print to see results
print()
print(top_box_office_movies)
    


# In[3]:


#Making new data frame from dictionary 
moviesdf = pd.DataFrame.from_dict(top_box_office_movies,orient="index")
moviesdf.reset_index(level=[0], inplace=True)
print(moviesdf.shape)

#Naming column tiles and showing the head of the data 
moviesdf.columns = ['Title', 'Rating', 'Revenue', 'Date']
moviesdf.head()


# In[4]:


#T2 

from sklearn.feature_extraction.text import CountVectorizer
import math
import pandas as pd   
from bs4 import BeautifulSoup
import requests
pd.set_option('display.max_colwidth', 150) 


# In[5]:



from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request


#Get  new page from Bleacher Report since Movie Titles didn't have the same words  
page = requests.get('https://bleacherreport.com/articles/2791799-helmet-rule-will-change-the-nfl-but-only-if-the-league-can-figure-it-out?utm_source=cnn.com&utm_medium=referral&utm_campaign=editorial')
 
#Create soup object from the page
soup = BeautifulSoup(page.text, "html5lib")


#Pulling only the article in the text in, we excluded text from ads and such 
article_html = soup.select(".organism.contentStream > div > p")

article_text = []

#Turned text into a list we can use 
for tag in article_html:
    article_text.append(tag.get_text(" ", strip=True))

# Don't have to run print, more of a check 
print(article_text)


# In[6]:


#Made our first vector
cv1 = CountVectorizer(binary=False) 

#Applied the transformation
cv1_chat = cv1.fit_transform(article_text) 

#Don't have to run print, more of a check 
print(type(cv1_chat))
print(cv1_chat.shape)

#Bag of Words 
print(cv1.get_feature_names())

#Printed in Data Frame format 
pd.DataFrame(cv1_chat.toarray(),columns = cv1.get_feature_names())


# In[7]:


#Applied stop words to new vector 
cv2 = CountVectorizer(binary=False, stop_words='english') 
cv2_chat = cv2.fit_transform(article_text) 

print(type(cv2_chat))
print(cv2_chat.shape)

#Included head only, but might be more helpful if all rows included
pd.DataFrame(cv2_chat.toarray(),columns = cv2.get_feature_names()).head(10)


# In[8]:


# Adding ngram to new vector 
pd.set_option('display.max_columns', None)
cv3 = CountVectorizer(binary=False, stop_words = 'english', ngram_range = (1,2)) 
cv3_chat = cv3.fit_transform(article_text) 

print(type(cv3_chat))
print(cv3_chat.shape)

#Included head, but might be more helpful if all rows included
pd.DataFrame(cv3_chat.toarray(),columns = cv3.get_feature_names()).head(10)


# In[9]:


#Adding min and max which could be helful in reducing the number of words 

cv4 = CountVectorizer(binary=False, min_df= .1,  stop_words = 'english') 
# only asking it to make changes based on document frequency


cv4_chat = cv4.fit_transform(article_text) 

print(type(cv4_chat))
print(cv4_chat.shape)

#Included head, but might be more helpful if all rows included
pd.DataFrame(cv4_chat.toarray(),columns = cv4.get_feature_names()).head(10)

#New parameters reduced size


# In[10]:


#Checked Weights of each word

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf1 = TfidfVectorizer(use_idf=True, norm=None) 
tf1_chat = tfidf1.fit_transform(article_text) 

pd.DataFrame(tf1_chat.toarray(),columns = tfidf1.get_feature_names()).head(10)

