#! /usr/bin/env python
import csv
import itertools
import operator
import numpy as np
import nltk
import pprint as pp
#nltk.download()
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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


def bitLen(int_type):
    length = 0
    while (int_type):
        int_type >>= 1
        length += 1
    return(length)

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading CSV file...")
with open('data/textomaspreguntasyrespuestas.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    #reader.__next__()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s" % (x) for x in sentences]
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
#print "Using vocabulary size %d." % vocabulary_size
#print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])

arreglo = []
x = []
y = []
print("---------------------------------------------------------------------------")
par = 0
anterior = 0
mayor = 0
for sent in tokenized_sentences:
    arreglo = []
    for w in sent:
        if w in word_to_index:
            arreglo.append(word_to_index[w])
        if len(sent)>=116:
            print("A LE RTA AAAAAAAAAAAAAAAAAAAAAAAAAaa")
    while (len(arreglo)< 116) :
        a=word_to_index[unknown_token]
        arreglo.append(a)
    if par % 2 == 0:
        x.append(arreglo)
    else:
        arreglo.sort()
        arreglo = [arreglo[next((i for i, x in enumerate(arreglo) if x), None)], arreglo[-1]]
        cero=word_to_index[unknown_token]
        posIni= arreglo[0]
        PosFin= arreglo[1]
        while ((len(y)< 116)):
            if((len(y)< 114)):
                y.append(cero)
            else:
                y.append(posIni)
                y.append(PosFin)
                break
        pp.pprint(y)
    par = par + 1

print("---------------------------------------------------------------------------")

entrada = np.asarray(x)
salida = np.asarray(y)

print("---------------------------------------------------------------------------")
print("E N T R A D A")
print("---------------------------------------------------------------------------")
for a in entrada:
    print (a)
print("---------------------------------------------------------------------------")
print("S A L I D A")
print("---------------------------------------------------------------------------")
for a in salida:
    print (a)

print("---------------------------------------------------------------------------")
print(" P A R A   V E C T O R   X ")
print("---------------------------------------------------------------------------")
print("---------------------------------------------------------------------------")
print(" N O R M A L I Z A C I O N   D E   V E C T O R E S ")
print("---------------------------------------------------------------------------")
vectoresfusionados = zip(entrada, salida)
arregloparanormalizar =[]
for x in vectoresfusionados:
    arregloparanormalizar.append(list(itertools.chain.from_iterable(x)))
arreglito1 = np.asarray(arregloparanormalizar)
pp.pprint(arreglito1)

print("---------------------------------------------------------------------------")
print(" A R R E G L O   D E   V E C T O R E S   S U M A D O S ")
print("---------------------------------------------------------------------------")

vectoresnormalizados =[]
for x in arregloparanormalizar:
    vectoresnormalizados.append(sum(x))
arreglito2 = np.asarray(vectoresnormalizados)
print arreglito2

print("---------------------------------------------------------------------------")
print(" N O R M A L I Z A C I O N   D E   V E C T O R E S   E N T R E   C E R O  Y   U N O ")
print("---------------------------------------------------------------------------")

vectoresrenormalizados = []
for y in range(0,len(vectoresnormalizados)):
    vectoresrenormalizados.append([x / float(vectoresnormalizados[y]) for x in arregloparanormalizar[y]])
arreglito3 = np.asarray(vectoresrenormalizados)
pp.pprint(arreglito3)

print("---------------------------------------------------------------------------")
print(" P A R A   V E C T O R   Y ")
print("---------------------------------------------------------------------------")

mayorlongitudx = 0
for x in range(0, len(entrada)):
    if len(entrada[x]) > mayorlongitudx:
        mayorlongitudx = len(entrada[x])

doublexlength = mayorlongitudx*2
print("Maxima longitud en bits de los textos+preguntas", mayorlongitudx)

#mayorlongitudy = 0
#for x in range(0, len(entrada)):
#    if len(salida[x]) > mayorlongitudy:
#        mayorlongitudy = len(salida[x])

bitsdemayorlongitudx = bitLen(mayorlongitudx)
print("Maxima longitud en bits de los textos+preguntas", bitsdemayorlongitudx)
doublexlength = bitsdemayorlongitudx*2
print("el doble de la Maxima longitud en bits de los textos+preguntas", doublexlength)
#bitsdemayorlongitudy = bitLen(mayorlongitudy)