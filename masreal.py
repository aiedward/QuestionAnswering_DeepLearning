#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
import pprint as pp

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '18000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/textomaspreguntasyrespuestas.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]


index_to_word1=[]
#index_to_word1.append(unknown_token)

print"--------------------index_to_word "

index_to_word=[unknown_token]

for sent in tokenized_sentences:
    index_to_word.extend(sent)


word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print(word_to_index[unknown_token])   

arreglo = []
x = []
y = []
#print("---------------------------------------------------------------------------")
par = 0
anterior = 0
mayor = 0
for sent in tokenized_sentences:
    arreglo = []
    for w in sent:
        if w in word_to_index:
            arreglo.append(word_to_index[w])
#    print arreglo

        if len(sent)>=116:
            print("A LE RTA AAAAAAAAAAAAAAAAAAAAAAAAAaa")
    while (len(arreglo)< 116) :
        a=word_to_index[unknown_token]
        arreglo.append(a)


    if par % 2 == 0:
        x.append(arreglo)
    else:
#        arreglo = np.asarray(arreglo)
        arreglo.sort()
#        arreglo = np.asarray(arreglo)
        y.append([arreglo[next((i for i, x in enumerate(arreglo) if x), None)], arreglo[-1]])
    par = par + 1

#print("---------------------------------------------------------------------------")

x1 = np.asarray([sent for sent in x])
X_train =  x1.astype(np.int32)
y1 = np.asarray([sent for sent in y])
y_train =  y1.astype(np.int32)
#Create the training data
#X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
#y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

print("---------------------------------------------------------------------------")
print("E N T R A D A len")
print(type(X_train))
#print("---------------------------------------------------------------------------")
#for a in X_train:
print (len(X_train))
print (X_train)
#print("---------------------------------------------------------------------------")
print("S A L I D A")
#print("---------------------------------------------------------------------------")
#for a in y_train:
print (len(y_train))
print (y_train)


 
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])


# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

