import json
import re

with open('dev-original.json') as data_file:    
    data = json.load(data_file)

print("Nro de Contextos:", len(data["data"]))
for a in range(0, len(data["data"])):
	contexto = open('contexto.txt','w')
	preguntas = open('preguntas.txt','w') 
	print("------------------------------------------------------------------")
	print(" \n C O N T E X T -")
	print("------------------------------------------------------------------")
	print("Nro de Parrafos:", len(data["data"][a]["paragraphs"]))
	for b in range(0, len(data["data"][a]["paragraphs"])):
		print("------------------------------------------------------------------")
		print(" \n P A R R A F O S -")
		print("------------------------------------------------------------------")
		print(data["data"][a]["paragraphs"][b]["context"])
		print("ELIMINANDO PALABRAS")
		algo = data["data"][a]["paragraphs"][b]["context"]
		algo = re.sub(r'\b\w{1,2}\b', '', algo)
		print 'algo vale:', algo
		contexto.write(algo.encode('utf-8')) 
		contexto.write("\n")
		contexto.write("\n")
		print("------------------------------------------------------------------")
		print(" Q U E S T I O N S -")
		print("------------------------------------------------------------------")
		print("Nro de Preguntas:", len(data["data"][a]["paragraphs"][b]["qas"]))
		for c in range(0, len(data["data"][a]["paragraphs"][b]["qas"])):
			print(data["data"][a]["paragraphs"][b]["qas"][c]["question"])
			print("ELIMINANDO PALABRAS DE PREGUNTAS")
			algo = data["data"][a]["paragraphs"][b]["qas"][c]["question"]
			algo = re.sub(r'\b\w{1,2}\b', '', algo)
			print 'PREGUNTA algo vale:', algo
			preguntas.write(algo.encode('utf-8'))
			preguntas.write("\n")
			print("------------------------------------------------------------------")
		print(" E N D   Q U E S T I O N S  -")
		print("------------------------------------------------------------------")
	print("------------------------------------------------------------------")
	print(" \n P A R R A F O S -")
	print("------------------------------------------------------------------")
	print("------------------------------------------------------------------")
	print("E N D   C O N T E X T -")
	print("------------------------------------------------------------------", a)
contexto.close()
preguntas.close()