#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:43:55 2018

@author: KaranJaisingh
"""

import keras
from keras.models import load_model

import pandas as pd
import numpy as np
import argparse
import random
import sys

maxlen = 50

parser = argparse.ArgumentParser(description='This is a Lyrics Generator program')
parser.add_argument("-s","--seed", type=str, help="Seed to generate text", default="life")
seed = parser.parse_args().seed

model = load_model('lyrics_model.h5')

dataset = pd.read_csv('lyrics.csv')
trainData = np.array(dataset.iloc[:, 5])
text = ""
for i in range(0, trainData.size):
    if(i % 10 == 0):
        text += str(trainData[i])
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


print('----- Generating text -----')
for diversity in [0.2, 0.5, 1.0]:
    
    print()
    generated = seed
    sentence = generated
    
    finalText = ''
    print('----- Diversity:', diversity, ' -----\n')
    
    for i in range(1000):
        
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        
        finalText += next_char
  
    fileName = "output-" + str(diversity) + ".txt"
    fo = open(fileName, "w")
    fo.write('----- Diversity: ')
    fo.write(str(diversity))
    fo.write(' -----\n')
    fo.write(finalText)
    fo.write("\n")
    fo.close()
    
print('----- Text generation complete! Open file to view -----')