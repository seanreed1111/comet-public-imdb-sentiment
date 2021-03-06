import thinc.extra.datasets
import random, os, pickle
import numpy as np


def download_imdb(limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    #based on https://github.com/explosion/spacy/blob/master/examples/training/train_textcat.py

    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data) #TTD: set random seed
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def maybe_download_imdb(filepath):
  try:
    with open(filepath,'rb') as f:
      data = pickle.load(f)
    X_train_raw,y_train_raw,X_test_raw,y_test_raw =\
    data["X_train"],data["y_train"], data["X_test"], data["y_test"]

  except FileNotFoundError:
    print("Error Loading from file. Loading from thinc.extra.datasets now...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) =download_imdb()

    data = {'X_train':X_train_raw,
        'y_train':y_train_raw,
        'X_test':X_test_raw,
        'y_test':y_test_raw
    }
    with open(filepath, 'wb') as f:
      pickle.dump(data, f)

  return(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

# def clean_y(y_raw):
#   """
#   Strips out truth values from dicts in y_raw, converts to np array
#   Return np array of dtype=int
#   """
#   y_clean = np.array([item["POSITIVE"] for item in y_raw], dtype=int)
#   return y_clean

import preprocess_text



if __name__ == '__main__':
  filename = 'imdb_thinc_data.pickle'
  filepath = os.path.join('..','data','thinc',filename)
  X_train_raw, y_train_raw, X_test_raw, y_test_raw =maybe_download_imdb(filepath)




  # y_train = preprocess_text.clean_y(y_train_raw)
  # y_test = preprocess_text.clean_y(y_test_raw)
  # X_train = preprocess_text.clean_review(X_train_raw)
  # X_test = preprocess_text.clean_review(X_test_raw)