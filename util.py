import json
#from pprint import pprint

with open('data_example.json') as data_file:    
    data = json.load(data_file)

#pprint(data)
#print (data)
print("\n----------------------------------------C O N T E X T ---------------------------------------\n")
print (data["data"][0]["paragraphs"][0]["context"])
print("\n---------------------------------------- Q U E S T I O N S ---------------------------------------\n")
print("el tamano del arreglo es: ", len(data["data"][0]["paragraphs"][0]["qas"]))
for x in range(0, len(data["data"][0]["paragraphs"][0]["qas"])):
	print(data["data"][0]["paragraphs"][0]["qas"][x]["question"])
#    print "We're on time %d" % (x)
#print(data["data"][0]["paragraphs"][0]["qas"][1]["question"])

