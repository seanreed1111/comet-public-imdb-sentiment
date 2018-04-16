# file to preprocess the IMDB sentiment X_test.tsv and X_train.tsv files
# X_test is (id  review), X_train is (id  sentiment review)

import os
import re
import logging
import pandas as pd
import numpy as np
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.phrases import Phraser, Phrases
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.corpora.dictionary import Dictionary

from bs4 import BeautifulSoup
ps = PorterStemmer()
# lem = WordNetLemmatizer.lemmatize()

def clean_y(y_raw):
  """
  Strips out True and False values from the dicts in y_raw. Makes and returns an np array of dtype=int from the values
  """
  y_clean = np.array([item["POSITIVE"] for item in y_raw], dtype=int)
  return y_clean


def clean_string(raw_string):
  """
  Input:
    raw_review: raw text of a movie review. Includes HTML

  Output:
    Cleaned string converted to lower case
  """
  # use BS to remove HTML markup
  new_string = BeautifulSoup(raw_review,'lxml').get_text().lower()
  # Return a cleaned string
  return new_string

def make_ngram(review):
  """
  Uses gensim to convert unigrams to bigrams, bigrams to trigrams, etc.
  example
     ice cream => ice_cream
     natural language_processing => natural_language_processing
  """
  pass

# def remove_stopwords(word_list, stopwords):
#   """
#   Input:  [words]
#   Output: [words]
#   """
#   stops = set(stopwords)
#   words = [w for w in word_list if w not in stops]
#   return words



# def clean_review(X_raw):
#   """
#   Transforms X_raw from thinc.datasets into X_clean,ready for modeling

#   Runs:
#     clean_string()
#     tokenize_text()
#     remove_stopwords() #make sure do not remove negative words
#     make_ngram()
#     make_dictionary()
#   """
#   pass

# test = ["The quick brown fox jumped over the lazy dog.".split(),
# "The quick brown fox is not that bright.".split()]


# #MUST RUN THE TOKENIZER AND SPLIT SENTENCES INTO ARRAY BEFORE Dictionary is instantiated.
# dic = Dictionary(test)
# print(dic.doc2bow("quick brown fox".split())) #[(1, 1), (3, 1), (7, 1)]
# print(dic) #Dictionary(13 unique tokens: ['The', 'brown', 'dog.', 'fox', 'jumped']...)
# print(dic.token2id)#{'The': 0, 'brown': 1, 'dog.': 2, 'fox': 3, 'jumped': 4, 'lazy': 5, 'over': 6, 'quick': 7, 'the': 8, 'bright.': 9, 'is': 10, 'not': 11, 'that': 12}

# spacy

def clean_string(raw_review):
  """
  Input:
    raw_review: raw text of a movie review. Includes HTML

  Output:
    Cleaned string converted to lower case
  """
  # use BS to remove HTML markup
  new_string = BeautifulSoup(raw_review,'lxml').get_text().lower()
  # Return a cleaned string
  return new_string

def clean_and_tokenize(X_test_raw):
 return [sent_tokenize(clean_string(review)) for review in X_test_raw]

def stem(word):
  return ps.stem(word)

def stem_and_remove_stops(review):
  """
  Input: list of sentences in the review
  Output: list of sentences where stop words have been removed
    and words have been stemmed with Porter Stemmer
  """
  new_review = []
  for sent in review:
    #run word tokenizer
    #remove stopwords
    #stem
    new_sent = word_tokenize(sent)

    for i,word in enumerate(new_sent[:]):
      if word not in stops:
        new_sent[i] = WordNetLemmatizer.lemmatize(word)
      else:
        new_sent[i] = ""
    new_review.append(" ".join(new_sent))
  return new_review


from download_imdb import maybe_download_imdb

if __name__ == '__main__':

  # test ="""The quick brown fox jumped over the lazy dog. The quick brown fox is not that bright. Don't at me. Just another lazy Sunday. HUMAN BEHAVIOR IS PROGRAMMABLE.Learn the basics, best practices, science, and secrets behind Behavioral Design from the hacker neuroscientist founders of Boundless.ai in our first book, Digital Behavioral Design.
  # """
  filename = 'imdb_thinc_data.pickle'
  filepath = os.path.join('..','data','thinc',filename)
  X_train_raw, _, X_test_raw, _ =maybe_download_imdb(filepath)

  stops = set(stopwords.words('English')) - set(["no", "not"])

  #first tokenize by sentences. Then remove and stem stopwords



  X_transform = clean_and_tokenize(X_test_raw[11:15]) #

  X_transform = [stem_and_remove_stops(review) for review in X_transform]

  print(X_test_raw[11])
  print(X_transform[11])

