#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:12:52 2018

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
import csv
import os


ROOT_PATH = os.path.dirname(os.getcwd())
PROJECT_PATH = os.path.join(ROOT_PATH,'ml_taller5')

def get_data():
    data=np.genfromtxt("credit.csv",delimiter=",", dtype=np.unicode_)
    X = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    X[:,0] = encoded_col(data, 0)
    X[:,1] = data[1:,1].astype(float)
    X[:,2] = encoded_col(data, 2)    
    X[:,3] = encoded_col(data, 3)
    X[:,4] = data[1:,4].astype(float)
    X[:,5] = encoded_col(data, 5)
    X[:,6] = encoded_col(data, 6)
    X[:,7] = data[1:,7].astype(float)
    X[:,8] = encoded_col(data, 8)
    X[:,9] = encoded_col(data, 9)
    X[:,10] = data[1:,10].astype(float)
    X[:,11] = encoded_col(data, 11)
    X[:,12] = data[1:,12].astype(float)
    X[:,13] = encoded_col(data, 13)
    X[:,14] = encoded_col(data, 14)
    X[:,15] = data[1:,15].astype(float)
    y = data[1:,16].astype(float)
    X[:,16] = data[1:,17].astype(float)
    X[:,17] = encoded_col(data, 18)
    X[:,18] = encoded_col(data, 19)
    X[:,19] = encoded_col(data, 20)
    # Formula para estandarizar
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def encoded_col(data, num_col):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(data[1:,num_col]).astype(float)

def main():
    X_train, X_test, y_train, y_test = get_data()
    clf = tree.DecisionTreeClassifier()        
    
    clf.fit(X_train, y_train)   
    predictions = clf.predict(X_test)
    # Simple Report
    print('{}'.format(metrics.classification_report(y_test, predictions)))
    # Cross validation
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print("Scores {}".format(scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #ROC Curve
    roc(y_test, predictions, None, 'Default(no;yes)')
    
def roc(y_test, predictions, pos_label, label):
    filename = 'trees_{}_roc_curve_.png'.format(label) 
    filepath = os.path.join(PROJECT_PATH, filename)
    fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=pos_label)
    #ROC Curve
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic: {}'.format(label))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(filename)
    plt.show()
    

if __name__=="__main__":
    main()