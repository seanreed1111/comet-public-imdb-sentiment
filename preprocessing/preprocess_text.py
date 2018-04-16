# file to preprocess the IMDB sentiment X_test.tsv and X_train.tsv files
# X_test is (id  review), X_train is (id  sentiment review)

# import os
# import re
# import logging
# import pandas as pd
# import numpy as np
# from gensim.models.tfidfmodel import TfidfModel
# from gensim.models.phrases import Phraser, Phrases
# from gensim.models.doc2vec import Doc2Vec
# from gensim.models.word2vec import Word2Vec

from gensim.corpora.dictionary import Dictionary
import spacy #`conda install -c conda-forge spacy=2.0.11`
# import en_core_web_lg

from bs4 import BeautifulSoup

def clean_y(y_raw):
  """
  Strips out truth values from dicts in y_raw, converts to np array
  Return np array of dtype=int
  """
  y_clean = np.array([item["POSITIVE"] for item in y_raw], dtype=int)
  return y_clean

def clean_review(raw_review):
  """
  Input:
    raw_review: raw text of a movie review. Includes HTML

  Output:
    Cleaned string converted to lower case
  """
  # use BS to remove HTML markup
  text = BeautifulSoup(raw_review,'lxml').get_text().lower()
  # Return a cleaned string
  return text

def tokenize_review(review):
  pass

def remove_stopwords(word_list, stopwords):
  """
  Input:  [words]
  Output: [words]
  """
  stops = set(stopwords)
  words = [w for w in word_list if w not in stops]
  return words

def make_ngram(review):
  """
  Uses gensim to convert unigrams to bigrams, bigrams to trigrams, etc.
  example
     ice cream => ice_cream
     natural language_processing => natural_language_processing


  """
  pass


# test = ["The quick brown fox jumped over the lazy dog.".split(),
# "The quick brown fox is not that bright.".split()]

# # print(clean_review(test))
# #MUST RUN THE TOKENIZER AND SPLIT SENTENCES INTO ARRAY BEFORE Dictionary is instantiated.
# dic = Dictionary(test)
# print(dic.doc2bow("quick brown fox".split())) #[(1, 1), (3, 1), (7, 1)]
# print(dic) #Dictionary(13 unique tokens: ['The', 'brown', 'dog.', 'fox', 'jumped']...)
# print(dic.token2id)#{'The': 0, 'brown': 1, 'dog.': 2, 'fox': 3, 'jumped': 4, 'lazy': 5, 'over': 6, 'quick': 7, 'the': 8, 'bright.': 9, 'is': 10, 'not': 11, 'that': 12}

# spacy test

test = "The quick brown fox jumped over the lazy dog. The quick brown fox is not that bright. Don't at me."

#`python -m spacy download en`
nlp = spacy.load('en')
doc = nlp(test)

print("Type of doc is \n", type(doc))
for token in doc:
  #print("type of token.text is {}",type(token.text))
  print("text is",token.text)

print(list(doc.sents)) #doesn't work because I need the dependency parse
