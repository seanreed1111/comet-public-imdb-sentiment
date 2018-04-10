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

# with open("data/imdb_data.pickle",'rb') as f:
#   X_train,X_test,y_train,y_test = pickle.load(f)

def load_data(filepath="data/imdb_data_old.pickle"):
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
X_train.shape
X_test.shape