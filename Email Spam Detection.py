#!/usr/bin/env python
# coding: utf-8

# In[32]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[33]:


import chardet 
with open("D:\Projects\email spam detection\spam.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# In[34]:


data = pd.read_csv("D:\Projects\email spam detection\spam.csv", encoding = 'Windows-1252')
data.head()


# In[35]:


data.describe()


# In[36]:


data.isnull().sum()


# In[37]:


{c:data[c].dropna().values for c in data.columns}


# In[38]:


data = data.reset_index(drop=True)
data.head()


# In[39]:


data.info()


# In[40]:


data.rename(columns = {'v1':'col1', 'v2':'col2', 'Unnamed: 2':'col3','Unnamed: 3':'col4','Unnamed: 4':'col5'}, inplace = True)
data


# In[41]:


data.info()


# In[42]:


data.drop(['col3','col4','col5'], inplace=True, axis=1)


# In[43]:


data.isnull().sum()


# In[44]:


import nltk
nltk.download('stopwords')


# In[45]:


pip install string


# In[46]:


import nltk
from nltk.corpus import stopwords
import string


# In[47]:


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_word = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_word


# In[48]:


data['col2'].head().apply(process_text)


# In[49]:


message4 = 'hello world hello hello world play'
message5 = 'test test test test one hello'
print(message4)
print()


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer 
bow4 = CountVectorizer(analyzer = process_text).fit_transform([[message4],[message5]])
print(bow4)
print()
print(bow4.shape)


# In[51]:


from sklearn.feature_extraction.text import CountVectorizer
message_bow = CountVectorizer(analyzer=process_text).fit_transform(data['col2'])


# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(message_bow,data['col1'],test_size=0.2, random_state=0)


# In[53]:


message_bow.shape


# In[54]:


pip install MultinomialNB


# In[55]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_train, y_train)
print(classifier.predict(x_train))
print(y_train.values)


# In[56]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
pred = classifier.predict(x_train)
print(classification_report(y_train,pred))
print()
print('confusion_matrix :\n',confusion_matrix(y_train,pred))
print()
print('accuracy_score:\n',accuracy_score(y_train,pred))


# In[57]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_test, y_test)
print(classifier.predict(x_test))
print(y_test.values)


# In[58]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
pred = classifier.predict(x_test)
print(classification_report(y_test,pred))
print()
print('confusion_matrix :\n',confusion_matrix(y_test,pred))
print()
print('accuracy_score:\n',accuracy_score(y_test,pred))

