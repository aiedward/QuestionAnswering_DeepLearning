#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import string
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from rnn_numpy import RNNNumpy



_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '1600'))
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
with open('data/context-data.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Imprimiendo sentences"
    for x in sentences:
        print(x)
print "Parsed %d sentences." % (len(sentences))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
vocab1= list()
for one in vocab:
    if (len(one[0])>2):
        vocab1.append(one[0])       

index_to_word = [x for x in vocab1]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print("---------------------------------------------------------------------------")
print("index_to_world:")
print("---------------------------------------------------------------------------")
for a in index_to_word:
    print(a)
print("---------------------------------------------------------------------------")
print("---------------------------------------------------------------------------")
print("world_to_index:")
print("---------------------------------------------------------------------------")
for a,b in word_to_index.items():
    print (a,b)
print("---------------------------------------------------------------------------")
print("Busqueda en word_to index")

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#print "(X_train[3]) Antes"
#print(X_train[3])


#model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#t1 = time.time()
#model.sgd_step(X_train[3], y_train[3], _LEARNING_RATE)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

#train_with_sgd(model, X_train, y_train, nepoch=1, learning_rate=_LEARNING_RATE)

#predictions = model.predict(X_train[3])
#print "predictions.shape"
#print predictions.shape
#print "predictions"
#print predictions
#print "(X_train[3])"
#print(X_train[3])
#print "S-------------------------------------sss"

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs

print ("Colocar al menos 8000, tam veector, vocabulary_size")
print (vocabulary_size)
model = RNNTheano(vocabulary_size, hidden_dim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
#save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('./data/trained-model-theano.npz', model)            


def generate_sentence(model,word):
    
    # find in index word
    wordFind= [word_to_index[word]]
    predict = model.forward_propagation(wordFind)
    samples = np.random.multinomial(1, predict[-1])
    sampled_word = np.argmax(samples)
    print("Prediccion : ")
    print(sampled_word)
    #nextWord=index_to_word[sampled_word]
    print (index_to_word[sampled_word])
    res=[]
    # We start the sentence with the start t
    for sent in sentences:
        esta= sent.find(nextWord)
        if esta>0:
            res.append(sent)

    return res        

     

print("Length index_word")
print "---------------------------------------------------------------"
word = raw_input("Introduzca la palabra clave de la pregunta a buscar  \n")  
res= generate_sentence(model,word)
print "Frases donde se ubica la respuesta"
print(res)
