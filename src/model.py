from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import pandas as pd
import numpy as np

def BoG_model(X,y, clf = 'Logistic', vocab = None, combining = False):
    """
    This function take in training data as X, y. 
    Then the data will be processed with BagOfWord, then feed into the Logistic / SGD Classifier.

    Return:
        pipe: The fitted Logistic Classifier.
    """
    if vocab is None:
        if clf == 'Logistic':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer()),
               ('clasiffier', LogisticRegression())
               ])
        if clf == 'SVM':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer()),
               ('clasiffier', SGDClassifier())
               ])
    else:
        if combining == True:
            count_vect = CountVectorizer().fit(X)   
            vocab_lst = np.unique(list(count_vect.vocabulary_) + list(vocab))
        if combining == False:
            vocab_lst = vocab
            
        if clf == 'Logistic':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
               ('clasiffier', LogisticRegression())
               ])
        if clf == 'SVM':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
               ('clasiffier', SGDClassifier())
               ])
    pipe.fit(X, y)
    return pipe

def Tfidf_model(X,y, clf = 'Logistic', vocab = None, combining = False):

    """
    This function take in training data as X, y. 
    Then the data will be processed with BagOfWord and Tf-Idf, then feed into the Logistic / SGD Classifier.

    Return:
        pipe: The fitted Logistic Classifier.
    """
    if vocab is None:
        if clf == 'Logistic':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer()),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', LogisticRegression())
                ])
        if clf == 'SVM':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer()),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', SGDClassifier())
                ])
    else:
        if combining == True:
            count_vect = CountVectorizer().fit(X)   
            vocab_lst = np.unique(list(count_vect.vocabulary_) + list(vocab))
        if combining == False:
            
            vocab_lst = vocab
            
        if clf == 'Logistic':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', LogisticRegression())
                ])
        if clf == 'SVM':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', SGDClassifier())
                ])
    pipe.fit(X, y)
    return pipe
