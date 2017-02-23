import json
#from pprint import pprint

with open('dev-original.json') as data_file:    
    data = json.load(data_file)

print("Nro de COntextos:", len(data["data"]))
for a in range(0, len(data["data"])):
	print("------------------------------------------------------------------")
	print(" \n C O N T E X T -")
	print("------------------------------------------------------------------")
	print("Nro de Parrafos:", len(data["data"][a]["paragraphs"]))
	for b in range(0, len(data["data"][a]["paragraphs"])):
		print("------------------------------------------------------------------")
		print(" \n P A R R A F O S -")
		print("------------------------------------------------------------------")
		print(data["data"][a]["paragraphs"][b]["context"])
		print("------------------------------------------------------------------")
		print(" Q U E S T I O N S -")
		print("------------------------------------------------------------------")
		print("Nro de Preguntas:", len(data["data"][a]["paragraphs"][b]["qas"]))
		for c in range(0, len(data["data"][a]["paragraphs"][b]["qas"])):
			print(data["data"][a]["paragraphs"][b]["qas"][c]["question"])
			print("------------------------------------------------------------------")
		print(" E N D   Q U E S T I O N S  -")
		print("------------------------------------------------------------------")
	print("------------------------------------------------------------------")
	print(" \n P A R R A F O S -")
	print("------------------------------------------------------------------")
	print("------------------------------------------------------------------")
	print("E N D   C O N T E X T -")
	print("------------------------------------------------------------------", a)
