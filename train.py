#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:34:07 2018

@author: KaranJaisingh
"""

import pandas as pd
import numpy as np
import random
import sys

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

dataset = pd.read_csv('lyrics.csv')
trainData = np.array(dataset.iloc[:, 5])
text = ""
for i in range(0, trainData.size):
    if(i % 10 == 0):
        text += str(trainData[i])
    
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of Sequences:', len(sentences))

# vectorize the input2/54
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# building an LSTM to train using the given
print('Building the LSTM Model')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# A function that allows us to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function that prints at the end of each epock
def on_epoch_end(epoch, logs):
    print()
    if(epoch > 0):
        print('----- Generating text after Epoch: %d\n' % epoch)
        for diversity in [0.2, 0.5, 1.0]:
            
            print('----- Diversity:', diversity, ' -----')
            generated = 'life'
            sentence = generated
        
            for i in range(1000):
                
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
        
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
                
            print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=2,
          callbacks=[print_callback])

print("LSTM Network Trained")
model.save('lyrics_model.h5')
del model
print("LSTM Network Saved")
