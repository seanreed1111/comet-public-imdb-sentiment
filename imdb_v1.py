# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
from comet_ml import Experiment
from keras.datasets import imdb

#create an experiment with your api key
exp = Experiment(api_key = os.environ.get("COMET_API_KEY"),
  project_name ='imdb-sentiment',
  auto_param_logging=True)

skip_top=30
maxlen=300
random_seed = 42

params = {
  "skip_top":skip_top,
  "maxlen":maxlen,
  "random_seed":random_seed
}

def load_data(filepath="data/imdb_data.pickle"):
  try:
    with open(filepath,'rb') as f:
      data = pickle.load(f)
    X_train,y_train,X_test,y_test =\
    data["X_train"],data["y_train"], data["X_test"], data["y_test"]

  except FileNotFoundError:
    print("Error Loading from file. Loading from keras now...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
      skip_top=skip_top,
      maxlen=maxlen,
      seed=random_seed)

  return X_train,y_train,X_test,y_test


X_train,y_train,X_test,y_test = load_data()

# set up experiment logs
exp.log_multiple_params(params)
exp.log_dataset_hash(X_train)

print("output classes in training set", np.unique(y_train))
print("Bincount of training set ", np.bincount(y_train))
print("\n\noutput classes in test set", np.unique(y_test))
print("Bincount of test set ", np.bincount(y_test))