import pandas as pd
import numpy as np
import fastai
from fastai import *
from fastai.text import * 
from functools import partial
import io
import os
import csv
import json

def train_ULMfit(dataset_path, model_path, model_name):
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        text=[]
        target=[]
        for row in csv_reader:
            text.append(row[0])
            target.append(row[1])
    
    dropout_multiplier = 0.5

    df = pd.DataFrame({'label':target,'text':text})

    data_lm = TextLMDataBunch.from_df(train_df = df, valid_df = df, path = "")
    data_clas = TextClasDataBunch.from_df(path = "", train_df = df, valid_df = df, vocab=data_lm.train_ds.vocab, bs=128)

    # data_lm.save('data/lm_ulmfit_data')
    # data_clas.save('data/class_ulmfit_data')

    text_classifier = text_classifier_learner(data_clas, drop_mult=dropout_multiplier)
    try:
        text_classifier.load_encoder(model_name+'LanguageModel_Encoder')
    except:
        language_model = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=dropout_multiplier)
    
        language_model.fit_one_cycle(1, 1e-2)

        # language_model.unfreeze()
        # language_model.fit_one_cycle(1, 1e-3)

        language_model.save_encoder(model_name+'LanguageModel_Encoder')
        text_classifier.load_encoder(model_name+'LanguageModel_Encoder')
        
    text_classifier.fit_one_cycle(10, 1e-2)
    for i in range(1,3,1):
        text_classifier.freeze_to(-i)
        text_classifier.fit_one_cycle(3, slice(1e-2/2, 1e-2))
    text_classifier.unfreeze()
    # text_classifier.fit_one_cycle(3, slice(1e-2/100, 1e-2))

    text_classifier.export(fname = model_path+model_name)

def parse_using_ULMfit(message, model_path, model_name):
    
    text_classifier = load_learner(path = model_path, fname = model_name)
    prediction = text_classifier.predict(message)
    #prediction = text_classifier.predict(message.text)
    
    return prediction
