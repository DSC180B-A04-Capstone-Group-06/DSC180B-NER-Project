from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import pandas as pd
import os
import numpy as np


def BoG_Logistic(path):
    df = pd.read_csv(path)
    X = df.text
    Y = df.type_code

    pipe = Pipeline([
           ('BagOfWord', CountVectorizer()),
           ('clasiffier', LogisticRegression(max_iter = 1000))
           ])
           
    pipe.fit(X, Y)
    return pipe

def Tfidf_Logistic(path):

    df = pd.read_csv(path)
    X = df.text
    Y = df.type_code

    pipe = Pipeline([
           ('BagOfWord', CountVectorizer()),
           ('TfIdf',TfidfTransformer()),
           ('clasiffier', LogisticRegression(max_iter = 1000))
           ])
           
    pipe.fit(X, Y)
    return pipe



