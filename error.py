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

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
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
with open('data/error.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s" % (x,) for x in sentences]

sentences1 = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Se obtuvo %d frases." % (len(sentences))
print "-------------------"
print"Tokenize_sentences palabras"
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences1]
print(tokenized_sentences)
print "-------------------"
index_to_word=[]
index_to_word.append(sentence_start_token)
index_to_word.append(sentence_end_token)

print"index_to_word "
for sent in sentences:
    index_to_word.extend(nltk.word_tokenize(sent))

print(index_to_word)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print "(word_to_index)"
print (word_to_index)


# Create the training data

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
print("X_train")
print(X_train)

y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


#model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#t1 = time.time()
#model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

#train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
