from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pandas as pd
import os
import numpy as np
import pickle


def BoG_Logistic(X,y):

    pipe = Pipeline([
           ('BagOfWord', CountVectorizer()),
           ('clasiffier', LogisticRegression(max_iter = 1000))
           ])
           
    pipe.fit(X, y)
    return pipe

def Tfidf_Logistic(X,y):

    pipe = Pipeline([
           ('BagOfWord', CountVectorizer()),
           ('TfIdf',TfidfTransformer()),
           ('clasiffier', LogisticRegression(max_iter = 1000))
           ])
           
    pipe.fit(X, y)
    return pipe


def build_model(path,save_path, model ='BoG', train_size = 0.6):
    if model not in ['BoG', 'Tfidf']:
        print("Please select the right model: BoG or Tfidf")
        return
    
    df = pd.read_csv(path)
    X = df.summary
    Y = df.type_code
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = train_size)
    X_val,X_test, y_val,y_test = train_test_split(X_test, y_test, test_size = 0.5)
    
    if model =='BoG':
        clf = BoG_Logistic(X_train,y_train)
        model_name = "BoG_model.pkl"
        with open(save_path + model_name, 'wb') as file:
            pickle.dump(model, file)
            
    if model =='Tfidf':
        clf = Tfidf_Logistic(X_train,y_train)
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

