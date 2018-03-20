import os,sys,pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenizer import tokenizer,tokenizer_lemmatizer


text_path = '../../bbc/'
folders = os.listdir(text_path) # vai devolver os nomes da pasta
corpus = []

for folder in folders:
	for file in os.listdir(text_path+folder):
		f = open(text_path+folder+'/'+file).read()
		corpus.append(f) # add os arqvos textos 


tfVectorizer = CountVectorizer(analyzer = "word",max_features=3000,tokenizer = tokenizer ,stop_words = 'english')
tfVectorizer2 = CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer)

tfVector = tfVectorizer.fit_transform(text for text in corpus)
tfVector2 = tfVectorizer2.fit_transform(text for text in corpus)

pickle.dump(tfVector, open("../../Objetos/ObjetosPreProcessados/TF/tfVector3kTokenizer.aug", "wb+"))
pickle.dump(tfVector2, open("../../Objetos/ObjetosPreProcessados/TF/tfVectorTotalTokenizer.aug", "wb+"))

