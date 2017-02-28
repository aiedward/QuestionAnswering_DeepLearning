import json
import re
# http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


diccionario = dict()
indices = dict()
a=1
with open('contexto.txt','r') as f:
	for line in f:
		for word in line.split():
			print(word)
			if word in diccionario:
				print('voy a anadir')
				diccionario[word].add(a)
			else:
				diccionario[word] = set()
				diccionario[word].add(a)
			print (a)
			a+=1

print 'imprimir diccionario'
for x in diccionario:
	print (x, diccionario[x])

print 'voy a buscar valores'
lista = list()

###
### E S C R I B I R  A Q U I  L A  P R E G U N T A
###

var = 'who used the concept force'
for a in var.split():
	if a in diccionario:
		lista.extend(diccionario[a])
		print (a, diccionario[a])

print 'mi lista es', lista
lista =sorted(lista)
print 'lista ordenada', lista
arreglo= group_consecutives(lista)

print 'segmentos de conincidencia', arreglo 
elegido =0
for x in range(0, len(arreglo)):
	if elegido < len(arreglo[x]):
		a = len(arreglo[x])
print 'el elegido es', elegido
print 'la respuesta esta adyacente a estas palabras', arreglo[elegido]