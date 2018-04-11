# -*- coding: utf-8 -*-
import numpy as np
import os
import tarfile
import pickle
from comet_ml import Experiment
#from keras.datasets import imdb



# #create an experiment with your api key
# exp = Experiment(api_key = os.environ.get("COMET_API_KEY"),
#   project_name ='imdb-sentiment',
#   auto_param_logging=True)

# skip_top=30
# maxlen=300
# random_seed = 42

# params = {
#   "skip_top":skip_top,
#   "maxlen":maxlen,
#   "random_seed":random_seed
# }

#extract tarfile
dest_path = "../data/extract"
src_path = "../data/tarfiles/aclImdb_v1.tar.gz"

def extract_all(src_path, dest_path):
  try:
    with tarfile.open(src_path,'r') as f:
      f.extractall(dest_path)

  except Exception as e:
    print(e)




# def download_imdb_data(filepath):
#   try:
#     with tarfile.open(filepath,'r') as f:
#       f.extractall("../data/extract")

#   except FileNotFoundError as e:
#     print("Error. Downloading from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")


# def load_keras_data(filepath="data/imdb_keras_data.pickle"):
#   try:
#     with open(filepath,'rb') as f:
#       data = pickle.load(f)
#     X_train,y_train,X_test,y_test =\
#     data["X_train"],data["y_train"], data["X_test"], data["y_test"]

#   except FileNotFoundError:
#     print("Error Loading from file. Loading from Keras now...")
#     (X_train, y_train), (X_test, y_test) = imdb.load_data(
#       skip_top=skip_top,
#       maxlen=maxlen,
#       seed=random_seed)

#   return X_train,y_train,X_test,y_test

# lets load the actual words
#http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.


# X_train,y_train,X_test,y_test = load_keras_data()

# set up experiment logs
# exp.log_multiple_params(params)
# exp.log_dataset_hash(X_train)

# print("output classes in training set", np.unique(y_train))
# print("Bincount of training set ", np.bincount(y_train))
# print("\n\noutput classes in test set", np.unique(y_test))
# print("Bincount of test set ", np.bincount(y_test))

# print(X_train[1])

