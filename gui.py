#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from Tkinter import *


def responder():
	try:
		_valor = int(entrada_texto.get())
		_valor = _valor * 5
		etiqueta.config(text=_valor)
	except ValueError:
		etiqueta.config(text="Introduce un numero!")


app = Tk()
app.title("Sistema de Preguntas y Respuestas")

#Ventana Principal
vp = Frame(app)
vp.grid(column=0, row=0, padx=(50,50), pady=(10,10))
vp.columnconfigure(0, weight=1)
vp.rowconfigure(0, weight=1)

etiqueta = Label(vp, text="Introduzca una pregunta")
etiqueta.grid(column=2, row=2, sticky=(W,E))

boton = Button(vp, text="Responder", command=responder)
boton.grid(column=1, row=1)

valor = ""
entrada_texto = Entry(vp, width=10, textvariable=valor)
entrada_texto.grid(column=2, row=1)

app.mainloop()