import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize, FreqDist
from nltk.stem import WordNetLemmatizer 
import string
import re
import spacy
nlp = spacy.load('en_core_web_lg')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
import six
import sys
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import time

np.random.seed(0)
import pickle


def save_data(data, name):
    """
    Saves data.

    Args:
        data (obj): data that needs to be saved

        name (str): file path
    """
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_data(name):
    """
    loads data
    
    Args:
        name (str): file path
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


# spaCy tokenizer
def clean_text(text, stopwords=False, tweet=True):
    """
    Cleans and tokenizes tweet text data.
    Args:
        text (str): tweet text data
        
        stopwords (bool): True if stopwords needs to be removed
        
        tweet (bool): True if text data are tweets.
    
    Returns:
        tokens (array): Array of tokenized words from given text.
    """

    if tweet:
        text = re.sub(r'@\S+', '', text) # Gets rid of any mentions
        text = re.sub(r'RT\S+', '', text) # Gets rid of any retweets
        text = re.sub(r'#', '', text) # Gets rid of hashtag sign
        text = re.sub(r'https?:\/\/\S+', '', text) # Gets rid of any links
        text = re.sub(r'[0-9]+.?[0-9]+', '', text) # Gets rid of X.X where X are numbers
        text = re.sub(r'#?(sx|Sx|SX)\S+', '', text) # Gets rid common mentions
        text = re.sub(r'(&quot;|&Quot;)', '', text) # Gets rid of quotes
        text = re.sub(r'(&amp;|&Amp;)', '', text) # Gets rid of quotes
        text = re.sub(r'link', '', text) # Gets rid of quotes
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-': # if token is not a pronoun
            temp_token = token.lemma_.lower().strip()
        else:
            temp_token = token.lower_
        tokens.append(temp_token)
    
    if stopwords:
        tokens_stopped = [token for token in tokens if token not in stopwords_list and len(token)>2]
    else:
        tokens_stopped = [token for token in tokens if len(token)>2]
    
    return tokens_stopped

def get_vec(x):
    """
    Extracts vector out of string
    Args:
        x (str): string
    Returns:
        vec (array): word vector
    """

    doc = nlp(x)
    vec = doc.vector
    return vec

def evaluate_binary_model(label, model, data, vectorizer=None, params=None, run_type='tfidf', smote=False, show_result=True):
    """
    Cleans and creates NLP sentiment analysis model
    
    Args:
        label (str): Name of the model

        model (str): Name of a model OR actual model object.
                        possible names: {'RF', 'XGB', 'NB', 'SVM'}
        data (dict): dictionary that contains X and y. 
                     {'X': X, 'y': y}
        vectorizer (obj): vectorizer object if run_type is not 'we'

        params (dict): parameters for model

        run_type (str): type of nlp model
                        {'tfidf', 'we'}
        
        smote (bool): True if SMOTE is needed to be implemented

        show_result (bool): True if report is desired (classification report and confusion plot)
    
    Returns:
        model, report
    """
    # Splits the data
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], random_state=42)
    
    train_row = X_train.shape[0]
    test_row = X_test.shape[0]
    
    # initializes report dictionary
    report={'label': label,
            'run_time': None,
            'train_row': train_row,
            'test_row': test_row,
            'total_row': train_row+test_row,
            'negative_recall': None,
            'positive_recall': None,
            'test_accuracy': None,
            'average_time': None}
    
    # starting time
    start_time = time.time()
    
    if run_type == 'tfidf':
        # Pipeline
        clf = get_model(model)
        
        if smote:
            pipe = Pipeline([('tfidf', vectorizer),
                            ('smote', SMOTE()),
                            ('clf', clf)])
        else:
            pipe = Pipeline([('tfidf', vectorizer),
                            ('clf', clf)])
        # Fits and trains the model
        pipe.fit(X_train, y_train)
        
        # y_prediction
        y_pred =pipe.predict(X_test)
        
        report['test_accuracy'] = accuracy_score(y_test, y_pred)
        
        recall_scores = recall_score(y_test, y_pred, average=None)

        report['negative_recall'] = recall_scores[0]
        report['positive_recall'] = recall_scores[1]

        if show_result:
            # prints classification report
            print(classification_report(y_test, y_pred))

            # plots confusion matrix
            plot_confusion_matrix(pipe, X_test, y_test, normalize='true', cmap='Blues')

        ## Times stops here
        stop_time = time.time()

        report['run_time'] = stop_time - start_time
        report['average_time'] = report['run_time']/report['total_row']

        return pipe, report
    
    elif run_type == 'we':
        # word2vec transfomration
        X_train = transform_vec(X_train)
        X_test = transform_vec(X_test)
        
        clf = get_model(model, params)
        
#         pipe = Pipeline([('word2vec', transform_vec),
#                          ('clf', clf)])
#         # Fits and trains the model
        clf.fit(X_train, y_train)
        
        # y_prediction
        y_pred = clf.predict(X_test)
        
        report['test_accuracy'] = accuracy_score(y_test, y_pred)
        
        recall_scores = recall_score(y_test, y_pred, average=None)

        report['negative_recall'] = recall_scores[0]
        report['positive_recall'] = recall_scores[1]

        if show_result:
            # prints classification report
            print(classification_report(y_test, y_pred))

            # plots confusion matrix
            plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap='Blues')

        ## Times stops here
        stop_time = time.time()

        report['run_time'] = stop_time - start_time
        report['average_time'] = report['run_time']/report['total_row']

        return clf, report

def get_model(name, params=None):
    """
    Instanciate a model
    
    Args:
        name (str, obj): Name of a model OR actual model object.
                        possible names: {'RF', 'XGB', 'NB', 'SVM'}
                    
        params (dict): Dictionary of parameters for the model.
    
    Returns:
        model (object): ML Model
    """

    if name == 'RF':
        if params:
            clf = RandomForestClassifier(**params)
        else:
            clf = RandomForestClassifier()
    elif name == 'NB':
        if params:
            clf = MultinomialNB(**params)
        else:
            clf = MultinomialNB()
    elif name == 'XGB':
        if params:
            clf = XGBClassifier(**params)
        else:
            clf = XGBClassifier()
    elif name == 'SVM':
        if params:
            clf = LinearSVC(**params)
        else:
            clf = LinearSVC()
    elif name == 'LR':
        if params:
            clf = LogisticRegression(**params)
        else:
            clf = LogisticRegression()
    
    return clf

def transform_vec(X):
    X = X.apply(get_vec)
    
    # Defining X and y
    X = X.to_numpy()
    X = X.reshape(-1, 1)

    # Reset the shape properly
    X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 300)
    
    return X

# Calculates class weights
from sklearn.utils.class_weight import compute_class_weight
def get_class_weights(y):
    """
    Calculates class weight of target data.
    Args:
        y (list, array): target data
    Returns:
        class_weight (list): list of class_weight values.
        class_weight_dict (dict): dictionary version of class_weight along with its class names
    """
    class_weight = list(compute_class_weight(class_weight='balanced', 
                                             classes=['Negative', 'Positive'], 
                                             y=y))
    class_weight_dict = {'Negative': class_weight[0],
#                     'Neutral': class_weight[1],
                    'Positive': class_weight[1]}
    return class_weight, class_weight_dict