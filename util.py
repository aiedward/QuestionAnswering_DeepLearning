import json
import re
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#archivo csv
context_data = open('context-data.csv', 'w')
csvwriter = csv.writer(context_data, delimiter=" ")
preguntas_data = open('preguntas.csv', 'w')
csvwriterpreguntas = csv.writer(preguntas_data, delimiter=" ")

count = 0

with open('dev-original.json') as data_file:    
	data = json.load(data_file)
print("Nro de Contextos:", len(data["data"]))

#VOY A GENERAR EL ARCHIVO DE 50 TEXTOS
for a in range(0, len(data["data"])):
	contexto = open('contexto.txt','w')
	preguntas = open('preguntas.txt','w')
	respuestas = open('respuestas.txt','w')
	print("------------------------------------------------------------------")
	print(" \n C O N T E X T -")
	print("------------------------------------------------------------------")
	#CONFIGURACION PARA 50 parrafos
	print("Nro de Parrafos:", len(data["data"][a]["paragraphs"]))
	print("Nro de COntextos:", len(data["data"]))
	for b in range(0, len(data["data"][a]["paragraphs"])):
		print("------------------------------------------------------------------")
		print(" \n P A R R A F O S -")
		print("------------------------------------------------------------------")
		print(data["data"][a]["paragraphs"][b]["context"])
		print("ELIMINANDO PALABRAS")
		algo = data["data"][a]["paragraphs"][b]["context"]
		csvwriter.writerow([algo])
		#algo = re.sub(r'\b\w{1,2}\b', '', algo)
		#print 'algo vale:', algo
		contexto.write(algo.encode('utf-8')) 
		contexto.write("\n")
		contexto.write("\n")
		print("------------------------------------------------------------------")
		print(" Q U E S T I O N S -")
		print("------------------------------------------------------------------")
		print("Nro de Preguntas:", len(data["data"][a]["paragraphs"][b]["qas"]))
		for c in range(0, len(data["data"][a]["paragraphs"][b]["qas"])):
			print("------------------------------------------------------------------")
			print("R E S P U E S T A S")
			print("------------------------------------------------------------------")
			for d in range(0, len(data["data"][a]["paragraphs"][b]["qas"][c]["answers"])):
				print(data["data"][a]["paragraphs"][b]["qas"][c]["answers"][d]["answer_start"])
				print(data["data"][a]["paragraphs"][b]["qas"][c]["answers"][d]["text"])
			print("------------------------------------------------------------------")
			print("R E S P U E S T A S")
			print("------------------------------------------------------------------")
			print(data["data"][a]["paragraphs"][b]["qas"][c]["question"])
			print("ELIMINANDO PALABRAS DE PREGUNTAS")
			algo = data["data"][a]["paragraphs"][b]["qas"][c]["question"]
			csvwriterpreguntas.writerow([algo])
			#algo = re.sub(r'\b\w{1,2}\b', '', algo)
			print 'PREGUNTA algo vale:', algo
			preguntas.write(algo.encode('utf-8'))
			preguntas.write("\n")
		print("------------------------------------------------------------------")
		print(" E N D   Q U E S T I O N S  -")
		print("------------------------------------------------------------------")
		++count
	print("------------------------------------------------------------------")
	print(" \n P A R R A F O S -")
	print("------------------------------------------------------------------")
	print("------------------------------------------------------------------")
	print("E N D   C O N T E X T -")
	print("------------------------------------------------------------------", a)
contexto.close()
preguntas.close()
context_data.close()
preguntas_data.close()