
# coding: utf-8

# In[2]:


#T1

#Imported Libraries 
import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup
import requests
pd.set_option('display.max_colwidth', 1500) 

#Import Frederick Douglas Speech 
page = requests.get('http://www.historyplace.com/speeches/douglass.htm')
 


# In[3]:


soup = BeautifulSoup(page.text, "html5lib")


# In[4]:


#Pulling only the speech in the text in, we excluded text from ads and such 
speech_html = soup.select("blockquote > blockquote > p > b")

speech_text = []

#Turned text into a list we can use 
for tag in speech_html:
    speech_text.append(tag.get_text(" ", strip=True))


#Making the list into a data frame
speech_df = pd.DataFrame({'paragraph':speech_text})

# cleaning paragraphs of punctuation and useless spaces
speech_df['paragraph'] = speech_df.paragraph.apply(lambda x: [word.rstrip('?:!.,;').lower() for word in x.split() if True])

print(speech_df)

type(speech_df)


# In[5]:


#T1
import math
import requests
pd.set_option('display.max_colwidth', 150) 

#Imported the stop words
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words("english")

#made a new column with the stop words removed
speech_df['clean_paragraph'] = speech_df['paragraph'].map(lambda x: [word for word in x if word not in nltk_stopwords])
print(speech_df)

#Printed the pargraph section and clean paragraph section to spot check and to get the lenghs of each text
paragraph_len = sum(speech_df['paragraph'].map(lambda x: len(x)))
clean_paragraph_len = sum(speech_df['clean_paragraph'].map(lambda x: len(x)))
print(speech_df['paragraph'].map(lambda x: len(x)))
print('Paragraph Length: {}'.format(paragraph_len))
print(speech_df['clean_paragraph'].map(lambda x: len(x)))
print('Clean Paragraph Length: {}'.format(clean_paragraph_len))


# In[105]:


#T2

import os 
path="/Users/emmydoo19/Desktop/BIA 6304/"
os.chdir(path)

#Pulling in our sentiment dictionary and giving instructions on how to determine if a paragraph is positive or negative
afinn = {}
for line in open(path+"AFINN-111.txt"):
    tt = line.split('\t')
    afinn.update({tt[0]:int(tt[1])})

    
HLpos = [line.strip() for line in  open(path+'HLpos.txt','r')]
HLneg = [line.strip() for line in  open(path +'HLneg.txt','r',encoding = 'latin-1')]
print("HL pos  size: " + str(len(HLpos)))
print(HLpos[0:10])
print("HL neg  size: " + str(len(HLneg)))
print(HLneg[0:10])
    

def afinn_sent(word_list):
    
    sentcount =0
    for word in word_list:  
        if word in afinn:
            sentcount = sentcount + afinn[word]
            
    
    if (sentcount < 0):
        sentiment = 'Negative'
    elif (sentcount >0):
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    
    return sentiment
    #return sentcount

#Applying the dictionary to the original paragraph text
speech_df['afinn'] = speech_df.paragraph.apply(lambda x: afinn_sent(x))
print(speech_df['afinn'])

#Applying the dictionary to the clean paragraph text with no stop words
speech_df['clean_afinn'] = speech_df.clean_paragraph.apply(lambda x: afinn_sent(x))
print(speech_df['clean_afinn'])


# In[106]:


#T3

#Surprised to see paragraph positve paragraphs given the topic of the speech
#Amplified and Negated words in the text

negate = ["aint", "arent","cant", "couldnt" , "didnt" , "doesnt" ,"dont" ,"hasnt" , "isnt" ,"mightnt" , "mustnt" ,"neither" ,"never", "no" ,"nobody" , "nor", "not" , "shant", "shouldnt", "wasnt" , "werent" ,"wont", "wouldnt"]
amplify = ["acute" ,"acutely", "certain", "certainly" ,"colossal", "colossally","deep" , "deeply" , "definite","definitely" ,"enormous","enormously" , "extreme", "extremely" ,"great","greatly" ,"heavily", "heavy", "high","highly" ,"huge","hugely" , "immense", "immensely" ,"incalculable" ,"incalculably","massive", "massively", "more","particular" ,"particularly","purpose", "purposely", "quite" ,"real" ,"really","serious", "seriously", "severe","severely" ,"significant" ,"significantly","sure","surely" , "true" ,"truly" ,"vast" , "vastly" , "very"]

#Creating a new sentiment dictionary with instructions of how to apply our Amplified and Negated words 
def afinn_sent2(word_list):
    
    sentcount =0
    i=0
    

    for word in word_list:
        prev = word_list.pop(i-1)

        if word in afinn:
            if (prev == 'no'):
                sentcount = sentcount - afinn[word] - afinn[prev]
            elif (prev == 'not'):
                sentcount = sentcount - afinn[word]
            else:
                sentcount = sentcount + afinn[word]
            i+=1
    
    if (sentcount < 0):
        sentiment = 'Negative'
    elif (sentcount >0):
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    
    
    return sentiment

#Applied new sentiment dictionary to original speech paragraphs
speech_df['afinn2'] = speech_df.paragraph.apply(lambda x: afinn_sent2(x))
print(speech_df['afinn2'])

#Applied new sentiment dictionary to clean speech paragraphs without stop words
speech_df['clean_afinn2'] = speech_df.clean_paragraph.apply(lambda x: afinn_sent2(x))
print(speech_df['clean_afinn2'])


# In[107]:


#Bonus

from collections import Counter 

from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech

#defining the POS tagger to use 
def get_pos( word ):
    w_synsets = wordnet.synsets(word)

    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0] # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )

#Running the lemmitizer with POS tagged words
words = ["running","lying","cars","dropped"]
wnl = WordNetLemmatizer()
for word in words:
    print(wnl.lemmatize( word, get_pos(word) )) #printing without newline character

