# file to preprocess the IMDB sentiment X_test.tsv and X_train.tsv files
# X_test is (id  review), X_train is (id  sentiment review)

import os
import re
import logging
import pandas as pd
import numpy as np
import gensim

def clean_review(raw_review):
    """
    Input:
            raw_review: raw text of a movie review. Includes HTML
            remove_stopwords: a boolean variable to indicate whether to remove stop words
            output_format: if "string", return a cleaned string
                           if "list", a list of words extracted from cleaned string.
    Output:
            Cleaned string or list.
    """

    # Remove HTML markup
    text = BeautifulSoup(raw_review)

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Split words make lower, and store to list
    words = text.lower().split()


    # Return a cleaned string
    return " ".join(words)

def tokenize_review(review):
  pass

def remove_stopwords(review):
  pass

def make_ngram(review):
  """
  Uses gensim to convert unigrams to bigrams, bigrams to trigrams, etc.
  example
     ice cream => ice_cream
     natural_language processing => natural_language_processing


  """
  pass
