import os,sys,pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenizer import tokenizer, tokenizer_lemmatizer


text_path = '../../bbc/'
folders = os.listdir(text_path) # vai devolver os nomes da pasta
corpus = []

for folder in folders:
	for file in os.listdir(text_path+folder):
		f = open(text_path+folder+'/'+file).read()
		corpus.append(f) # add os arqvos textos

# # Pre-Processamento Binario
binaryVectorizer = CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, max_features=3000, binary = True)
# Parametro max_features -> Seleciona as X palavras mais frequente # // Objetivo Eliminar Ruido
binaryVectorizer2 = CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, binary = True)

binaryVector = binaryVectorizer.fit_transform(text for text in corpus)
binaryVecto2 = binaryVectorizer2.fit_transform(text for text in corpus)

pickle.dump(binaryVector, open("../../Objetos/ObjetosPreProcessados/Binario/binaryVector3kTokenizerLemmatizer.aug", "wb+"))
pickle.dump(binaryVecto2, open("../../Objetos/ObjetosPreProcessados/Binario/binaryVectorTotalTokenizerLemmatizer.aug", "wb+"))