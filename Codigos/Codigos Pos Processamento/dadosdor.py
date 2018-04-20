import os
import pickle
	
def recupera_dados(dicinfo):
	
	while('Trabalho-de-IA' not in os.listdir()):
		os.chdir('..')
	os.chdir('Trabalho-de-IA')
	os.chdir('Objetos')

	if(dicinfo['corpus']=='bbc'):
		os.chdir('ObjetosPreprocessados')
	else:
		os.chdir('Objetos Preprocessados Reuters')
	os.chdir(dicinfo['representacao'])
	os.chdir(dicinfo['tamanho'])
	os.chdir(dicinfo['processamento'])

	print("conversando com os amiguinhos ",os.listdir())
	amigo=None
	for a in os.listdir():
		if(dicinfo['LSA'] and 'LSA' in a):
			print("lsa otario")
			amigo=pickle.load(open(a,'rb'))
		if('LSA' not in a and dicinfo['LSA']==False):
			print("não lsa otario")
			amigo=pickle.load(open(a,'rb'))
	return amigo
