from __future__ import unicode_literals
import re
import os
import codecs
import json
import csv
import spacy
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
import math
import random
from tqdm import tqdm
from nltk.corpus import wordnet
import pickle

import time

nlp=spacy.load('en_core_web_lg')

#HyperParameters
oversample = True             # Whether to oversample small classes or not. True in the paper
synonym_extra_samples = True # Whether to replace words by synonyms in the oversampled samples. True in the paper
augment_extra_samples = False  # Whether to add random spelling mistakes in the oversampled samples. False in the paper
additional_synonyms = 0       # How many extra synonym augmented sentences to add for each sentence. 0 in the paper
additional_augments = 0       # How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
mistake_distance = 2.1        # How far away on the keyboard a mistake can be
    
def get_synonyms(word, number= 3):
    synonyms = []
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name().lower().replace("_", " "))
    synonyms = list(OrderedDict.fromkeys(synonyms))
    return synonyms[:number]

nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
verbs = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')}

class Augment_Dataset():
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards
    
        It's the actual test set as defined in the paper that we comparing against."""

    def __init__(self, path):
        """ Instantiate the object.
            @param: dataset_path The directory which contains the data set."""
        self.path =  path
        self.X, self.y = self.load()
        self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                   'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                   'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                   'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                   's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                   'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                   'm': {'x': 6, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                   'h': {'x': 5, 'y': 1}, 'j': {'x': 6, 'y': 1}, 'k': {'x': 7, 'y': 1},
                                   'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2},
                                   'ß': {'x': 10,'y': 2}, 'ü': {'x': 10,'y': 2}, 'ä': {'x': 10,'y': 0},
                                   'ö': {'x': 11,'y': 0}}
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
        self.splits = self.stratified_split()

    def get_nearest_to_i(self, keyboard_cartesian):
        """ Get the nearest key to the one read.
            @params: keyboard_cartesian The layout of the QWERTY keyboard for English
            
            return dictionary of eaculidean distances for the characters"""
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i, j) < mistake_distance: #was > 1.2
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def _shuffle_word(self, word, cutoff=0.7):
        """ Rearange the given characters in a word simulating typos given a probability.
        
            @param: word A single word coming from a sentence
            @param: cutoff The cutoff probability to make a change (default 0.9)
            
            return The word rearranged 
            """
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc] in self.keyboard_cartesian:
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)

    def _euclidean_distance(self, a, b):
        """ Calculates the euclidean between 2 points in the keyboard
            @param: a Point one 
            @param: b Point two
            
            return The euclidean distance between the two points"""
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)

    def _get_augment_sentence(self, sentence):
        return ' '.join([self._shuffle_word(item) for item in sentence.split(' ')])
    
    def _augment_sentence(self, sentence, num_samples):
        """ Augment the dataset of file with a sentence shuffled
            @param: sentence The sentence from the set
            @param: num_samples The number of sentences to genererate
            
            return A set of augmented sentences"""
        sentences = []
        for _ in range(num_samples):
            sentences.append(self._get_augment_sentence(sentence))
        sentences = list(set(sentences))
        # print("sentences", sentences)
        return sentences + [sentence]

    def _augment_split(self, X, y, num_samples=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)
            
            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X, y):
            tmp_x = self._augment_sentence(X, num_samples)
            sample = [[Xs.append(item), ys.append(y)] for item in tmp_x]
            
        return Xs, ys

    # Randomly replaces the nouns and verbs by synonyms
    def _synonym_word(self, word, cutoff=0.5):
        if random.uniform(0, 1.0) > cutoff and len(get_synonyms(word)) > 0 and word in nouns and word in verbs:
            return random.choice(get_synonyms(word))
        return word
    
    # Randomly replace words (nouns and verbs) in sentence by synonyms
    def _get_synonym_sentence(self, sentence, cutoff = 0.5):
        return ' '.join([self._synonym_word(item, cutoff) for item in sentence.split(' ')])

    # For all classes except the largest ones; add duplicate (possibly augmented) samples until all classes have the same size
    def _oversample_split(self, X, y, synonym_extra_samples = False, augment_extra_samples = False):
        """ Split the oversampled train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
        
            return Oversampled training dataset"""
        
        classes = {}
        for X, y in zip(X, y):
            if y not in classes:
                classes[y] = []
            classes[y].append(X)
            
        max_class_size = max([len(entries) for entries in classes.values()])
        
        Xs, ys = [],[] 
        for y in classes.keys():
            for i in range(max_class_size):
                sentence = classes[y][i % len(classes[y])]
                if i >= len(classes[y]):
                    if synonym_extra_samples:
                        sentence = self._get_synonym_sentence(sentence)
                    if augment_extra_samples:
                        sentence = self._get_augment_sentence(sentence)
                Xs.append(sentence)
                ys.append(y)
               
        return Xs, ys
    
    def _synonym_split(self, X, y, num_samples=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)
            
            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X, y):
            sample = [[Xs.append(self._get_synonym_sentence(X)), ys.append(y)] for item in range(additional_synonyms)]
            
        return Xs, ys

    def load(self):
        """ Load the file
        return The vector separated in test, train and the labels for each one"""
        X ,y = get_data(self.path)
        return X, y

    def process_sentence(self, x):
        """ Clean the tokens from stop words in a sentence.
            @param x Sentence to get rid of stop words.
            
            returns clean string sentence"""
        clean_tokens = []
        doc = nlp.tokenizer(x)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    def process_batch(self, X):
        """See the progress as is coming along.
        
            return list[] of clean sentences"""
        return [self.process_sentence(a) for a in tqdm(X)]

    def stratified_split(self):
        """ Split data whole into stratified test and training sets, then remove stop word from sentences
        
            return list of dictionaries with keys train,test and values the x and y for each one"""
        self.X = ([preprocess(sentence) for sentence in self.X])
        if oversample:
            self.X, self.y = self._oversample_split(self.X, self.y, synonym_extra_samples, augment_extra_samples)
        if additional_synonyms > 0:
            self.X, self.y = self._synonym_split(self.X, self.y, additional_synonyms)
        if additional_augments > 0:
            self.X, self.y = self._augment_split(self.X, self.y, additional_augments)

        splits = {"X": self.X, "y": self.y}
        return splits

    def get_splits(self):
        """ Get the splitted sentences
            
            return splitted list of dictionaries"""
        return self.splits
    
#*****************************************************

def tokenize(doc): #tick
    """
    Returns a list of strings containing each token in `sentence`
    """
    #return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
    #                            doc) if i != '' and i != ' ' and i != '\n']
    tokens = []
    doc = nlp.tokenizer(doc)
    for token in doc:
        tokens.append(token.text)
    return tokens

def preprocess(doc): #tick
    clean_tokens = []
    doc = nlp(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)

def find_ngrams(input_list, n): #tick
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text): #tick
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens

def semhash_corpus(corpus): #tick
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str,tokens)))
    return new_corpus

def get_vectorizer(corpus, preprocessor=None, tokenizer=None): #tick
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer.fit(corpus)
    return vectorizer

def vectorize_data(corpus, sample, model_path): #cross
    vectorizer = get_vectorizer(corpus, preprocessor=preprocess, tokenizer=tokenize)
    
    filename = model_path+'hash_vectorizer.sav'
    
    pickle.dump(vectorizer, open(filename, 'wb'))
    
    corpus_hashed = vectorizer.transform(corpus).toarray()
    X_hashed = vectorizer.transform(sample).toarray()
    
    return corpus_hashed, X_hashed

def get_data(path): #tick
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X=[]
        y=[]
        for row in csv_reader:
            X.append(row[0])
            y.append(row[1])
    return X,y
    
def sklearn_numpy_warning_fix():
    """Fixes unecessary warnings emitted by sklearns use of numpy.

    Sklearn will fix the warnings in their next release in ~ August 2018.

    based on https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array"""
    import warnings

    warnings.filterwarnings(module='sklearn*', action='ignore',
                            category=DeprecationWarning)
sklearn_numpy_warning_fix()

def get_vectorized_data(dataset_path, model_path):
    #
    dataset = Augment_Dataset(dataset_path)
    
    splits = dataset.get_splits()
    
    X_str = splits['X'] ; y_label = splits['y']
    
    X_hash = semhash_corpus(X_str)
    
    corpus_hashed, X_vector = vectorize_data(X_hash, X_hash, model_path = model_path)
    #
    y_num = str2num(sample = y_label, training_data = y_label)
    
    return X_vector, y_num

def vectorize_message(text, filename):
    
    text_hash = semhash_corpus([text])
    
    vectorizer = pickle.load(open(filename, 'rb'))
    
    text_vector = vectorizer.transform(text_hash).toarray()[0]
    
    return text_vector

def num2str(sample, training_data):
    le = LabelEncoder()
    le.fit(training_data)
    sample = le.inverse_transform(sample)
    
    return sample
    
def str2num(sample, training_data):
    le = LabelEncoder()
    le.fit(training_data)
    sample = le.transform(sample)
    
    return sample

def train_ssh(dataset_path, model_path = 'models/ssh/', model_name = 'ssh_classifier.sav'):
    
    from sklearn.naive_bayes import BernoulliNB
    
    clf = BernoulliNB(alpha=.01)

    X, y = get_vectorized_data(dataset_path = dataset_path, model_path = model_path)
    
    clf.fit(X,y)
    
    filename = model_path+model_name
    
    pickle.dump(clf, open(filename, 'wb'))

def parse_using_ssh(message, model_path , model_name, vectorizer_name, dataset_path):
    
    filename_model = model_path+model_name
    filename_vectorizer = model_path+vectorizer_name
    
    clf = pickle.load(open(filename_model, 'rb'))
    
    message = vectorize_message(message, filename = filename_vectorizer)
    
    prediction_clf = clf.predict(message.reshape(1, -1))
    
    dataset = Augment_Dataset(dataset_path) ; splits = dataset.get_splits() ; y_label = splits['y']
    
    prediction_num2str = num2str(sample = prediction_clf, training_data = y_label)
    
    return prediction_num2str

def parse_batch_using_ssh(messages, model_path , model_name , vectorizer_name, dataset_path):
    
    filename_model = model_path+model_name
    filename_vectorizer = model_path+vectorizer_name
    
    clf = pickle.load(open(filename_model, 'rb'))
    
    prediction_clf=[]
    for i in messages:
        message = vectorize_message(i, filename = filename_vectorizer)
    
        prediction_clf.append(clf.predict(message.reshape(1, -1)))
    
    dataset = Augment_Dataset(dataset_path) ; splits = dataset.get_splits() ; y_label = splits['y']
    
    prediction_num2str=[]
    for i in prediction_clf:
        prediction_num2str.append(num2str(sample = i, training_data = y_label))
    
    return prediction_num2str
