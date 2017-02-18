#! /usr/bin/env python
import csv
import itertools
import operator
import numpy as np
import nltk
#nltk.download()
import sys
import os
import time
from datetime import datetime
from utils import *


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading CSV file...")
with open('info2.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.__next__()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    for x in sentences:
    	print(x)
print ("Parsed %d sentences." % (len(sentences)))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
#print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
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
#print "Using vocabulary size %d." % vocabulary_size
#print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
print("---------------------------------------------------------------------------")
print("X   T R A I N")
print("---------------------------------------------------------------------------")
for a in X_train:
    print(a)
print("---------------------------------------------------------------------------")
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
print("---------------------------------------------------------------------------")
print("Y   T R A I N")
print("---------------------------------------------------------------------------")
for a in y_train:
    print(a)
print("---------------------------------------------------------------------------")