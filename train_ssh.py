from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

print('loading ssh scripts')
from subword_semantic_hashing import *


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

def train(train_path):
    try:  
        os.mkdir('models/')
    except FileExistsError:
        pass
    try:
        os.mkdir('models/ssh/')
    except FileExistsError:
        pass
    
    train_ssh(dataset_path = train_path, 
                        model_path = 'models/ssh/', 
                        model_name = 'ssh_classifier.sav')

def evaluate(test_path):
    test_df = pd.read_csv(test_path)

    labels = test_df['label'].unique().tolist()
    test_text = test_df['text'].tolist()
    
    results=parse_batch_using_ssh(messages = test_text, 
                model_path = 'models/', 
                model_name = 'ssh_classifier.sav', vectorizer_name = 'hash_vectorizer.sav', 
                dataset_path = 'data/train.csv')
    
    cm_analysis(test_df['label'], results, labels, ymap=None, figsize=(15,15))

    print(classification_report(test_df['label'], results, labels=labels))

    

if __name__ == '__main__':
    train('data/train.csv')
    evaluate('data/test.csv')
