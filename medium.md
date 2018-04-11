Description of IDMB Sentiment Task
The IMDB dataset is a collection of positive and negative reviews of [X] different movies from the IMDB database. The individual reviews are contained in separate text files that are annotated with "0" for a negative review and "1" for a positive review.
The data is split into train and test sets.
We then run several different classes of models on the training data, all the while tracking our experiments via Comet.ml
Finally we will predict our test set and review several error metrics appropriate to the for the classification task.

Note: The preprocessing steps needed for different models could be different. 
    Check MAX_LEN and skip_top for each configuration
    Need to standardize them or mark where and when they vary.

We will run the following list of models on this dataset, using unigrams, bigrams, and trigrams, and Bag of Words with TF-IDF Vectorizer unless otherwise noted:

    Bernoulli Naive Bayes
    Binary Multinominal Naive Bayes
    Logistic Regression: Bag of Words with Count Vectorizer   
    Logistic Regression: Bag of Words with TF-IDF Vectorizer
    Linear SVM: Bag of Words with TF-IDF Vectorizer
    Linear SVM: Bag of Words with Count Vectorizer
    Kernel SVM: Bag of Words with TF-IDF Vectorizer
    Kernel SVM: Bag of Words with Count Vectorizer
    Random Forest Classifier
    Gradient Boosted Classifier

    Word Vectors: 
        pretrained word2vec (gensim)
        pretrained doc2vec (gensim)
        wikipeida pretrained fastText (spacy)
    spaCy v2.0 has an internal text classifier on IMDB that loads the dataset automatically 
    LSTM(128) with recurrent dropout
    Bidirectional LSTM(8) with dropout
    CNN-LSTM with Conv1D 64 5x5 Filters feeding into a LSTM(70)


sklearn Pipeline: Train Models via gridsearchCV. Validate Parameters. Test!
Keras: Train Models, Grid Search validation params, test!
