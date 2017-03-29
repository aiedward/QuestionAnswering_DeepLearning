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
import sys
from Tkinter import *

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '10000'))
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
    sentences = ["%s" % (x,) for x in sentences]

sentences1 = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Se obtuvo %d frases." % (len(sentences))
print "-------------------"
print"Tokenize_sentences palabras"
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences1]
#print(tokenized_sentences)
print "-------------------"
index_to_word=[]
index_to_word.append(unknown_token)
index_to_word.append(sentence_start_token)
index_to_word.append(sentence_end_token)

print"index_to_word "
for sent in sentences:
    index_to_word.extend(nltk.word_tokenize(sent))
#print(index_to_word)
print"--------------------tamano index_to_word "
print(len(index_to_word))


word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print "(word_to_index)"
#print (word_to_index)
print"----------------- tamano word to index "
#print(word_to_index)


# Create the training data

arreglo = []
x = []
y = []
print("---------------------------------------------------------------------------")
par = 0
anterior = 0
mayor = 0
posIni= ""
PosFin= ""
for sent in tokenized_sentences:
    arreglo = []
    for w in sent:
        if w in word_to_index:
            arreglo.append(word_to_index[w])
        if len(sent)>=116:
            print("---------------------------ALERTA------------------------------")
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
        #pp.pprint(y)
    par = par + 1
    #lalala ='{:b}'.format(posIni)
    #lelele ='{:b}'.format(posFin)
    #pp.pprint(lalala)
    #pp.pprint(lelele)
print("---------------------------------------------------------------------------")

X_train = np.asarray(x)
y_train= np.asarray(y)


#X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
#print("X_train")
#print(X_train)

#y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
#print("Y_train")
#print(y_train)








model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
 
#train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
load_model_parameters_theano('./data/final.npz', model)

def generate_sentence(model,bs):
    valida=0
    # We start the sentence with the start token
    new_sentence = [word_to_index[bs]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token] and valida==0:
        next_word_probs = model.forward_propagation(new_sentence)
        samples = np.random.multinomial(1, next_word_probs[-1])
        sampled_word = np.argmax(samples)
        if (sampled_word<len(word_to_index)):

            #return  index_to_word[sampled_word]
            return  sampled_word
            
    return null



def imprimir(arg):
    arg.sort()
    res=[]
    for x in arg:
        if len(index_to_word[x])>2:
            res.append(str(index_to_word[x]))
    print"La Respuesta es: "
    print (str(res))
    return str(res)    

def responder(pregunta):
    tokenized_sentences = nltk.word_tokenize(pregunta)    
    print (tokenized_sentences)
    sent=[]
    for w in tokenized_sentences:
        if len(w)>3:
            foo = word_to_index[w] if word_to_index.get(w) else False
            if foo:
                sent.append(generate_sentence(model,w))
    
    return imprimir(sent)            


def hacer_click():
 try:
  _valor = entrada_texto.get()
  #_valor = _valor
  resp=responder(str(_valor)) 
  etiqueta.config(text=resp)
 except ValueError:
  etiqueta.config(text="ERROR AL PROCESAR LA PREGUNTA")
 
app = Tk()
app.title("MiniProyecto-Preguntas y Respuestas")
 
#Ventana Principal
vp = Frame(app)
vp.grid(column=0, row=0, padx=(500,500), pady=(400,400))
vp.columnconfigure(0, weight=3)
vp.rowconfigure(0, weight=3)
 
etiqueta = Label(vp, text="Introduzca la pregunta")
etiqueta.grid(column=1, row=1, sticky=(W,E))
 
boton = Button(vp, text="Responder", command=hacer_click,bg='green',relief=RIDGE)
boton.grid(column=2, row=2,padx=(20,20), pady=(20,20))
 
valor = ""
entrada_texto = Entry(vp, width=10, textvariable=valor)
entrada_texto.grid(column=2, row=1)
 
app.mainloop()
sent=[]
#sent=responder("Which team won Super Bowl 50. Where was Super Bowl 50 held? The name of the NFL championship game is?")
#imprimir(sent) 

