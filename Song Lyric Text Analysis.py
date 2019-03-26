
# coding: utf-8

# In[111]:


#import and preview the data
import pandas as pd
import os 
pathname = "/Users/emmydoo19/Desktop/BIA 6304/"
os.chdir(pathname)

#import and preview the data

df = pd.read_csv(pathname+"lyrics.csv",low_memory=False)
df.head()


# In[112]:


#replace carriage returns
df = df.replace({'\n': ' '}, regex=True)
df.head()


# In[113]:


#count the words in each song
df['word_count'] = df['lyrics'].str.split().str.len()
df.head()


# In[114]:


#let's see what the songs with 1 word look like
df1 = df.loc[df['word_count'] == 1]
df1.head()


# In[115]:


#elimintate the 1-word songs and review the data again
df = df[df['word_count'] != 1]
df['word_count'].groupby(df['genre']).describe()


# In[117]:


#There are still some outliers on the low end. Reviewing songs with less than 100 words.
df100 = df.loc[df['word_count'] <= 100]
df100.head()


# In[118]:


#let's check on the high end
df1000 = df.loc[df['word_count'] >= 1000]
df1000.head()


# In[119]:


#let's get rid of the outliers on the low and high end 
del df1, df100, df1000 

print(df.shape)

df_clean1 = df[df['word_count'] >= 100]
print(df_clean.shape)

df_clean2 = df_clean1[df_clean1['word_count'] <= 1000]
print(df_clean.shape)

df_clean2['word_count'].groupby(df_clean2['genre']).describe()
print(df_clean2.shape)

genre = df_clean2.groupby(['genre'],as_index=False).count()
genre2 = genre[['genre','song']]
genre2



# In[120]:


liquor = pd.DataFrame(df_clean2.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('liquor')].count()))
liquor.reset_index(inplace=True)
liquor.columns = ['genre', 'liquor_lyrics']
liquor


# In[121]:


beer = pd.DataFrame(df_clean2.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('beer')].count()))
beer.reset_index(inplace=True)
beer.columns = ['genre', 'beer_lyrics']
beer


# In[122]:


wine = pd.DataFrame(df_clean2.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('wine')].count()))
wine.reset_index(inplace=True)
wine.columns = ['genre', 'wine_lyrics']
wine


# In[123]:


pills = pd.DataFrame(df_clean2.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('pills')].count()))
pills.reset_index(inplace=True)
pills.columns = ['genre', 'pills_lyrics']
pills


# In[124]:


weed = pd.DataFrame(df_clean2.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('weed')].count()))
weed.reset_index(inplace=True)
weed.columns = ['genre', 'weed_lyrics']
weed


# In[125]:


import functools
dfs = [genre2,beer,wine,liquor,pills,weed]
genre3 = functools.reduce(lambda left,right: pd.merge(left,right,on='genre', how='outer'), dfs)
genre3


# In[126]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt

genreCount = df_clean2['genre'].value_counts()
yearCount  = df_clean2['year'].value_counts().head( 12 )

fig, axarr = plt.subplots(2, 2)
fig.tight_layout()


genreCount.plot.pie( figsize=(12, 12), fontsize=16, ax=axarr[0][0] , autopct='%.1f' )
genreCount.plot.bar( figsize=(22, 12), fontsize=16, ax=axarr[0][1] )

yearCount.plot.pie( figsize=(12, 12), fontsize=16, ax=axarr[1][0], autopct='%.1f' )
yearCount.plot.bar( figsize=(22, 12), fontsize=16, ax=axarr[1][1] )


# In[127]:


import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import seaborn as sns

customStopWords = ["'s", "n't", "'m", "'re", "'ll","'ve","...", "ä±", "''", '``',                  '--', "'d", 'el', 'la']
stopWords = stopwords.words('english') + customStopWords

words = ""
for song in df_clean2.iterrows():
    words += " " + song[1]['lyrics']

words = nltk.word_tokenize( words.lower() )
words = [ word for word in words if len(word) > 1                             and not word.isnumeric()                             and word not in stopWords ]
    
word_dist = FreqDist( words  )
print("The 10 most common words in the dataset are :")
for word, frequency in word_dist.most_common(10):
    print( u'{} : {}'.format( word, frequency ) )

plt.figure(figsize=(15, 10))
nlp_words = word_dist.plot( 20 )


# In[181]:


def clean(string):
    char_to_rem = ["\n", "'", ",", "]", "[", ")", "("]

    for c in char_to_rem:
        string = string.replace(c, "")

    final_string = []

    for word in string.split(' '):
        word = word.lower()

        if word == "fag" or word == "ho" or word == "hoe" or word == "ass":
            final_string.append(word)
            continue

        if len(word) > 3 and word not in stopWords:
            final_string.append(word)

    return final_string


def update(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key] = dic1[key] + dic2[key]
        else:
            dic1[key] = dic2[key]

# Starting with evaluating the top 5 words for every artist along with their frequencies

import time

start_time = time.time()

grouped_by_artist = df_clean2.groupby('artist')


# saving the total words in this dict
# total number of songs
ar_di = {}
tot_words = {}
tot_words_list = []

artist_strings = {}

for artist, songs in grouped_by_artist:
    num_total_words = 0
    num_songs = 0
    artist_string = []
    
    words = {}

    for index, rows in songs.iterrows():
        num_songs += 1
        clean_text_list = clean(rows["lyrics"])
        num_total_words += len(clean_text_list)
            
    

        tot_words_list += clean_text_list
        artist_string += clean_text_list
        
        for word in clean_text_list:
            if word in words:
                words[word] = words[word] + 1
            else:
                words[word] = 1

        update(tot_words, words)
        artist_strings[artist] = list(artist_string)
        
    
    print ("Artist: ", artist)
    print ("Total words in all songs", num_total_words)

    for key, val in sorted(words.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:5]:
            print ("\t", key, "used", val, "times")

    print ("\n\n")

end_time = time.time()
print('Time: {} sec'.format(end_time - start_time))


# In[182]:


cuss_words = ["f**k", "f*g", "d**k", "t**s", "p***y", "h**", "a**", "n-word", "s**t", "c**k", "b***h", "c**t"]

# preprocessing to update some counts
tot_words["n-word"] = tot_words["nigger"] + tot_words["nigga"]
del tot_words["nigger"]
del tot_words["nigga"]

tot_words["h**"] = tot_words["ho"] + tot_words["hoe"]
del tot_words["hoe"]

tot_words["f**k"] = tot_words["fuck"] 
del tot_words["fuck"]

tot_words["f*g"] = tot_words["fag"] 
del tot_words["fag"]

tot_words["d**k"] = tot_words["dick"] 
del tot_words["dick"]

tot_words["t**s"] = tot_words["tits"] 
del tot_words["tits"]

tot_words["p***y"] = tot_words["pussy"] 
del tot_words["pussy"]

tot_words["a**"] = tot_words["ass"] 
del tot_words["ass"]

tot_words["s**t"] = tot_words["shit"] 
del tot_words["shit"]

tot_words["c**k"] = tot_words["cock"] 
del tot_words["cock"]

tot_words["b***h"] = tot_words["bitch"] 
del tot_words["bitch"]

tot_words["c**t"] = tot_words["cunt"] 
del tot_words["cunt"]

counts_cuss_words = [tot_words[x] for x in cuss_words]


# In[183]:


fig = plt.figure(figsize=(9, 6))

cuss_series = pd.Series.from_array(counts_cuss_words)

# plt.bar(np.arange(len(cuss_words)), counts_cuss_words, color="grey")
ax = cuss_series.plot(kind='bar')
# ax.set_title("Amount Frequency")
# ax.set_xlabel("Amount ($)")
# ax.set_ylabel("Frequency")
ax.set_xticklabels(cuss_words)
# ax.xaxis.set_visible(False)

# plt.xticks(cuss_words)
plt.show()


plt.savefig("bar_cuss_words.png")


# In[184]:


artist_cuss = []

for artist in artist_strings.keys():
    counter = 0

    for sting in artist_strings[artist]:
        if sting in cuss_words:
            counter += 1
     
    artist_cuss.append((counter, artist))

sorted(artist_cuss, reverse=True)

