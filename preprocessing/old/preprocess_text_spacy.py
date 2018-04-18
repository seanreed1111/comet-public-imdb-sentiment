# file to preprocess the IMDB sentiment X_test.tsv and X_train.tsv files
# X_test is (id  review), X_train is (id  sentiment review)

import os
import re
import logging
import pandas as pd
import numpy as np
# from gensim.models.tfidfmodel import TfidfModel
# from gensim.models.phrases import Phraser, Phrases
# from gensim.models.doc2vec import Doc2Vec
# from gensim.models.word2vec import Word2Vec

from gensim.corpora.dictionary import Dictionary
#`conda install -c conda-forge spacy=2.0.11`
#`python -m spacy download en`

import spacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import PhraseMatcher
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES


from bs4 import BeautifulSoup

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

def remove_stopwords(word_list, stopwords):
  """
  Input:  [words]
  Output: [words]
  """
  stops = set(stopwords)
  words = [w for w in word_list if w not in stops]
  return words



def clean_review(X_raw):
  """
  Transforms X_raw from thinc.datasets into X_clean,ready for modeling

  Runs:
    clean_string()
    tokenize_text()
    remove_stopwords() #make sure do not remove negative words
    make_ngram()
    make_dictionary()
  """
  pass

# test = ["The quick brown fox jumped over the lazy dog.".split(),
# "The quick brown fox is not that bright.".split()]


# #MUST RUN THE TOKENIZER AND SPLIT SENTENCES INTO ARRAY BEFORE Dictionary is instantiated.
# dic = Dictionary(test)
# print(dic.doc2bow("quick brown fox".split())) #[(1, 1), (3, 1), (7, 1)]
# print(dic) #Dictionary(13 unique tokens: ['The', 'brown', 'dog.', 'fox', 'jumped']...)
# print(dic.token2id)#{'The': 0, 'brown': 1, 'dog.': 2, 'fox': 3, 'jumped': 4, 'lazy': 5, 'over': 6, 'quick': 7, 'the': 8, 'bright.': 9, 'is': 10, 'not': 11, 'that': 12}

# spacy test

from download_imdb import maybe_download_imdb
if __name__ == '__main__':

  # test ="""The quick brown fox jumped over the lazy dog. The quick brown fox is not that bright. Don't at me. Just another lazy Sunday. HUMAN BEHAVIOR IS PROGRAMMABLE.Learn the basics, best practices, science, and secrets behind Behavioral Design from the hacker neuroscientist founders of Boundless.ai in our first book, Digital Behavioral Design.
  # """
  filename = 'imdb_thinc_data.pickle'
  filepath = os.path.join('..','data','thinc',filename)
  X_train_raw, _, _, _ =maybe_download_imdb(filepath)

  nlp = spacy.load('en')
  # matcher = PhraseMatcher(nlp.vocab, max_length=3) #trigrams
  # lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


  doc = nlp(" ".join(X_train_raw))

  print("len(matcher) is", len(matcher) )

  # print("Type of doc is \n", type(doc))
  # # for token in doc:
  # #   #print("type of token.text is {}",type(token.text))
  # #   print("text is",token.text)

  # print(list(doc.sents))
  # for item in doc.noun_chunks:
  #   print(item)
