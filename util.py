import json
#from pprint import pprint

with open('dev-original.json') as data_file:    
    data = json.load(data_file)

print("Nro de Parrafos:", len(data["data"]))
for x in range(0, len(data["data"])):
	print("\n ----------------------------------------C O N T E X T ---------------------------------------\n")
	print(data["data"][x]["paragraphs"][x]["context"])
	print("\n ----------------------------------------E N D   C O N T E X T ---------------------------------------\n")


	print("Nro de Preguntas", len(data["data"][x]["paragraphs"][x]["qas"]))
	print("\n---------------------------------------- Q U E S T I O N S ---------------------------------------\n")
	for x in range(0, len(data["data"][x]["paragraphs"][x]["qas"])):
		print(data["data"][0]["paragraphs"][0]["qas"][x]["question"])
	print("\n---------------------------------------- E N D   Q U E S T I O N S ---------------------------------------\n")
