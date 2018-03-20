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

tfVectorizer = CountVectorizer(analyzer = "word",tokenizer = tokenizer_lemmatizer,max_features=3000, stop_words = 'english')
tfVectorizer2 = CountVectorizer(analyzer = "word",tokenizer = tokenizer_lemmatizer, stop_words = 'english')

tfVector = tfVectorizer.fit_transform(text for text in corpus)
tfVector2 = tfVectorizer2.fit_transform(text for text in corpus)

pickle.dump(tfVector, open("../../Objetos/ObjetosPreProcessados/TF/tfVector3kTokenizerLemmatizer.aug", "wb+"))
pickle.dump(tfVector2, open("../../Objetos/ObjetosPreProcessados/TF/tfVectorTotalTokenizerLemmatizer.aug", "wb+"))