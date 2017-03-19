import json
import re
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#archivo csv
all_data = open('all.csv', 'w')
csvwriterall = csv.writer(all_data, delimiter=" ")

context_data = open('context.csv', 'w')
csvwriter = csv.writer(context_data, delimiter=" ")

preguntas_data = open('preguntas.csv', 'w')
csvwriterpreguntas = csv.writer(preguntas_data, delimiter=" ")

respuestas_data = open('respuestas.csv', 'w')
csvwriterrespuestas = csv.writer(respuestas_data, delimiter=" ")

preguntasyrespuestas_data = open('preguntasyrespuestas.csv', 'w')
csvwriterpreguntasyrespuestas = csv.writer(preguntasyrespuestas_data, delimiter=" ")

count = 0

with open('data/dev-original.json') as data_file:    
	data = json.load(data_file)
print("Nro de Contextos:", len(data["data"]))

#VOY A GENERAR EL ARCHIVO DE 50 TEXTOS
for a in range(0, 1):
	print("------------------------------------------------------------------")
	print(" \n C O N T E X T -")
	print("------------------------------------------------------------------")
	#CONFIGURACION PARA 50 parrafos
	print("Nro de Parrafos:", len(data["data"][a]["paragraphs"]))
	print("Nro de COntextos:", len(data["data"]))
	for b in range(0, 50):
		print("------------------------------------------------------------------")
		print(" \n P A R R A F O S -")
		print("------------------------------------------------------------------")
		print(data["data"][a]["paragraphs"][b]["context"])
		print("ELIMINANDO PALABRAS")
		algo = data["data"][a]["paragraphs"][b]["context"]
		csvwriterall.writerow([algo])
		csvwriter.writerow([algo])
		#algo = re.sub(r'\b\w{1,2}\b', '', algo)
		#print 'algo vale:', algo
#		contexto.write(algo.encode('utf-8')) 
#		contexto.write("\n")
#		contexto.write("\n")
		print("------------------------------------------------------------------")
		print(" Q U E S T I O N S -")
		print("------------------------------------------------------------------")
		print("Nro de Preguntas:", len(data["data"][a]["paragraphs"][b]["qas"]))
		for c in range(0, len(data["data"][a]["paragraphs"][b]["qas"])):

			print(data["data"][a]["paragraphs"][b]["qas"][c]["question"])
			print("ELIMINANDO PALABRAS DE PREGUNTAS")
			algo = data["data"][a]["paragraphs"][b]["qas"][c]["question"]
			csvwriterall.writerow([algo])
			csvwriterpreguntas.writerow([algo])			
			csvwriterpreguntasyrespuestas.writerow([algo])
			#algo = re.sub(r'\b\w{1,2}\b', '', algo)
			print 'PREGUNTA algo vale:', algo
#			preguntas.write(algo.encode('utf-8'))
#			preguntas.write("\n")
			print("------------------------------------------------------------------")
			print("R E S P U E S T A S")
			print("------------------------------------------------------------------")
			algo = ""
			count = 0
			for d in range(0, len(data["data"][a]["paragraphs"][b]["qas"][c]["answers"])):
				print(data["data"][a]["paragraphs"][b]["qas"][c]["answers"][d]["answer_start"])
				print(data["data"][a]["paragraphs"][b]["qas"][c]["answers"][d]["text"])
				alguito = str(data["data"][a]["paragraphs"][b]["qas"][c]["answers"][d]["text"])
				print ("algo vale: ", algo)
				if(algo != alguito):
					if (count == 0):
						algo += alguito
					else:
						algo += " "
						algo += alguito
			csvwriterall.writerow([algo])
			csvwriterrespuestas.writerow([algo])
			csvwriterpreguntasyrespuestas.writerow([algo])
			print("------------------------------------------------------------------")
			print("R E S P U E S T A S")
			print("------------------------------------------------------------------")
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
all_data.close()
context_data.close()
preguntas_data.close()
respuestas_data.close()
preguntasyrespuestas_data.close()