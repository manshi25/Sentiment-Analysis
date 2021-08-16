import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
stopword_list=nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

# remove html tag
def html_tag(text):
  soup=BeautifulSoup(text,"html.parser")
  new_text=soup.get_text()
  return new_text

# !pip install contractions
import contractions
def con(text):
  expand=contractions.fix(text)
  return expand

# removal of special character
import re
def remove_sp(text):
  pattern=r'[^A-Za-z0-9\s]'
  text=re.sub(pattern,'',text)
  return text


# remove stop words
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()

def remove_stop_words(text):
  tokens=tokenizer.tokenize(text)
  tokens=[token.strip() for token in tokens]
  filtered_tokens=[token for token in tokens if token not in stopword_list]
  filtered_text=' '.join(filtered_tokens)
  return filtered_text

st.title("Tweet's Sentiment Analysis")

df = pd.read_csv('https://raw.githubusercontent.com/bhukyavamshirathod/Sentimental-Analysis/main/Train.csv')

df.tweet=df.tweet.apply(lambda x:x.lower())
df.tweet=df.tweet.apply(html_tag)
df.tweet=df.tweet.apply(con)
df.tweet=df.tweet.apply(remove_sp)
df.tweet=df.tweet.apply(remove_stop_words)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs=SentimentIntensityAnalyzer()

df['compound']=df['tweet'].apply(lambda x: vs.polarity_scores(x)['compound'])

conditions = [
    (df['compound'] >= 0.05),
    (df['compound'] > -0.05) & (df['compound'] < 0.05),
    (df['compound'] <= -0.05)
    ]

# create a list of the values we want to assign for each condition
values = ['Positive', 'Neutral', 'Negative']

# create a new column and use np.select to assign values to it using our lists as arguments
df['tier'] = np.select(conditions, values)

x = df.iloc[:,2].values
y = df.iloc[:,-1].values

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 
text_model.fit(x,y)
select = st.text_input('Enter your message')

if(st.markdown(
    '<span class="badge badge-pill badge-success"> Badge </span>',
    unsafe_allow_html=True
)):
  op = text_model.predict([select])
  ans=op[0]

  if ans == 'Positive':
    st.success("Positive 🙂")
  if ans == 'Negative':
    st.error("Negative 😠")
  if ans== 'Neutral':
    st.warning("Neutral 😐") 
