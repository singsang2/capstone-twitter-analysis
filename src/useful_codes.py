import pandas as pd
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
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

def get_percent_retained(df):
    """
    Prints percentage of data retained from original dataset.

    Args:
        df (Pandas.DataFrame): dataframe
    """
    original_length = 59400
    print(f'Original Length: {original_length}')
    print(f'Current Length: {df.shape[0]}')
    print(f'Percent Retained: {round(df.shape[0]/original_length * 100, 2)}%')

def drop_rows_na(df, col):
    """
    Drops rows with null value from given column.

    Args:
        df (Pandas.DataFrame): dataframe

        col (str): name of column from df.
    """
    indices = df[col].dropna().index
    return df.loc[indices,:]
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics

def evaluate_clf_model(model, X_train, y_train, X_test, y_test, classes=[], label='',
                       normalize='true', cmap='Blues'):
    """
    Evaluates a classifier model by providing
        [1] Metrics including accuracy, AUC, and cross validation score.
        [2] Classification report
        [3] Confusion Matrix

    Args:
        model (clf ojb): classifier model
        
        X_train (dataframe): Training dataset
        
        y_train (array): Training target
        
        X_test (dataframe): test dataset
        
        y_test (array): test target
        
        features (list): names of the features included in the test. (Default=None)
        
        classes (list): list of classes in the target. (Default=['functioning', 'needs repair', 'nonfunctioning'])
        
        prob (bool): True of model contains pred_prob values.
        
        feature_importance (bool): True if model provide feature_importance.
        
        normalize (str): 'true' if normalize confusion matrix annotated values.
        
        cmap (str): color map for the confusion matrix

        label (str): name of the classifier.
        
        cv (int): Number of cross folds for cross validation model.
    Returns:
        report: classfication report
        fig, ax: matplotlib object
    """
    ## Get Predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    table_header = "[i] CLASSIFICATION REPORT"
    
    # Add Label if given
    if len(label)>0:
        table_header += f" {label}"
    
    ## PRINT CLASSIFICATION REPORT
    dashes = '---'*20
    print(dashes,table_header,dashes,sep='\n')    
    print('Train Accuracy : ', round(metrics.accuracy_score(y_train, y_hat_train),4))
    print('Test Accuracy : ', round(metrics.accuracy_score(y_test, y_hat_test),4))
    

    print(metrics.classification_report(y_test,y_hat_test,
                                    target_names=classes))
    
    report = metrics.classification_report(y_test,y_hat_test,
                                               target_names=classes,
                                          output_dict=True)
    print(dashes+"\n\n")
    
    

    ## MAKE FIGURE
    fig, ax = plt.subplots(figsize=(10,4))
    ax.grid(False)
    
    ## Plot Confusion Matrix eva
    metrics.plot_confusion_matrix(model, X_test,y_test,
                                  display_labels=classes,
                                  normalize=normalize,
                                  cmap=cmap,ax=ax)
    ax.set(title='Confusion Matrix')
    plt.xticks(rotation=45)
    

    return report, fig, ax

def run_model(name, params=None, data=None, report=True):
    """
    Fits and evaluates model
    
    Args:
        name (str): Name of model.
                    {'RF', 'XGB', 'NB', 'SVM'}
                    
        params (dict): Dictionary of parameters for the model.
        
        **data (dataframe): X_train, X_test, y_train, y_test
        
        report (bool): True if classification report/confusion matrix is wanted.
    
    Returns:
        model (object): fitted model
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
        clf = XGBClassifier(**params)
    
    clf.fit(data['X_train'], data['y_train'])
    
    if report:
        evaluate_clf_model(clf, data['X_train'], data['y_train'],
                           data['X_test'], data['y_test'],
                           classes=['Negative emotion', 'Positive emotion'],
                           label=f'{name.capitalize()} Classifier');
    return clf