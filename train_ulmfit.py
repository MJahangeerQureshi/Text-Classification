from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from ULMfit import *

import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap=sns.dark_palette("#2ecc71", as_cmap=True))
    plt.show()

def train():
    """
    Train ULMfit based on the dataset provided.
    args: 
      dataset_name: Name of the datasets to be trained.
                    2 Datasets are available
                    "LabBot"            23 distinct intents, examples = 1009
                    "WebApplication".   8 distinct intents, examples = 80
    """
    try:  
        os.mkdir('models/ULMfit/')
    except FileExistsError:
        pass
    try:
        os.mkdir('models/ULMfit/')
    except FileExistsError:
        pass
       
    train_ULMfit(dataset_path = 'data/train_endava.csv', 
    model_path = 'models/ULMfit/', 
    model_name = 'ULMfit_classifier')


def evaluate():
    """
    Test ULMfit based on the dataset provided.
    args: 
      dataset_name: Name of the datasets to be trained.
                    2 Datasets are available
                    "LabBot"            23 distinct intents, examples = 1009
                    "WebApplication".   8 distinct intents, examples = 80
    """
    
    with open('data/train_endava.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        test_text=[]
        labels=[]
        target=[]
        for row in csv_reader:
            test_text.append(row[0])
            target.append(row[1])
    labels = pd.DataFrame(target).unique().tolist()
    
    results=[]
    for i in test_text:
        r = parse_using_ULMfit(message = i, model_path = 'models/ULMfit/'+dataset_name+'/', model_name = 'ULMfit_classifier')
        results.append(np.array([str(r[0])]))

    cm_analysis(target, results, labels, ymap=None, figsize=(15,15))
    print(classification_report(target, results, labels=labels))

    

if __name__ == '__main__':
    train()
    evaluate()
