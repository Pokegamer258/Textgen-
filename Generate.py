#https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/model/load.py
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import keras
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import sys
import io
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import os
#get file
path = get_file('romeo+juliet.txt', origin='http://www.awesomefilm.com/script/romeo+juliet.txt')
#reads file
with io.open(path, encoding='latin-1') as f:
 #makes everything lowercase
    text = f.read().lower()   
#makes variables important to establish input shape    
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
#load model    
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

start_index = random.randint(0, len(text) - maxlen - 1)
f= open("output.txt","w+") #makes output text file
for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        #sentence = "What did the viola say?"
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(10000):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = loaded_model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char


            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


