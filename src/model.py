from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import pandas as pd
import os
import numpy as np
import pickle


def BoG_model(X,y, clf = 'Logistic', vocab = None):
    """
    This function take in training data as X, y. 
    Then the data will be processed with BagOfWord, then feed into the Logistic / SGD Classifier.

    Return:
        pipe: The fitted Logistic Classifier.
    """
    if vocab == None:
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
        if clf == 'Logistic':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab)),
               ('clasiffier', LogisticRegression())
               ])
        if clf == 'SVM':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab)),
               ('clasiffier', SGDClassifier())
               ])
    pipe.fit(X, y)
    return pipe

def Tfidf_model(X,y, clf = 'Logistic', vocab = None):

    """
    This function take in training data as X, y. 
    Then the data will be processed with BagOfWord and Tf-Idf, then feed into the Logistic Classifier.

    Return:
        pipe: The fitted Logistic Classifier.
    """
    if vocab == None:
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
        if clf == 'Logistic':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', LogisticRegression())
                ])
        if clf == 'SVM':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', SGDClassifier())
                ])
    pipe.fit(X, y)
    return pipe



def build_model(path,save_path, model ='BoG', clf = 'Logistic', train_size = 0.6):
    """
    This function will take in the data set, and split the train/validation/test with 60%/20%/20%.
    Also build the model based on the choice of model.
    Print the accuracy of the model on different section of model.
    
    Return:
        clf: The fitted Logistic Classifier.
    """

    if model not in ['BoG', 'Tfidf']:
        print("Please select the right model: BoG or Tfidf")
        return
    
    df = pd.read_csv(path)
    X = df.summary
    Y = df.type_code
    
    # train/validation/test with 60%/20%/20%.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = train_size)
    X_val,X_test, y_val,y_test = train_test_split(X_test, y_test, test_size = 0.5)
    
    # Training and saving the BoG model
    if model =='BoG':
        clf = BoG_model(X_train,y_train, clf)
        model_name = "BoG_model.pkl"
        with open(save_path + model_name, 'wb') as file:
            pickle.dump(model, file)
    
    # Training and saving the Tf-Idf model
    if model =='Tfidf':
        clf = BoG_model(X_train,y_train, clf)
        model_name = "Tfidf_model.pkl"
        with open(save_path + model_name, 'wb') as file:
            pickle.dump(model, file)
            
            
    print('==========================================')
    print('Model Name: ',model ,' Model')
    print('Training Accuracy: ', np.mean(clf.predict(X_train) == y_train))
    print('Validation Accuracy: ', np.mean(clf.predict(X_val) == y_val))
    print('Test Accuracy: ', np.mean(clf.predict(X_test) == y_test))
    print('==========================================')
        
    return clf

