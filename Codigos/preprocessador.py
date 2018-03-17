import os,sys,pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tokenizer

text_path = '../bbc/'
folders = os.listdir(text_path) # vai devolver os nomes da pasta
corpus = []

for folder in folders:
	for file in os.listdir(text_path+folder):
		f = open(text_path+folder+'/'+file).read()
		corpus.append(f) # add os arqvos textos 


# # Pre-Processamento Binario 
binaryVectorizer = CountVectorizer(analyzer = "word", stop_words = 'english', max_features=3000, binary = True)
# Parametro max_features -> Seleciona as X palavras mais frequente # // Objetivo Eliminar Ruido
binaryVectorizer2 = CountVectorizer(analyzer = "word", stop_words = 'english', binary = True)

binaryVector = binaryVectorizer.fit_transform(text for text in corpus)
binaryVecto2 = binaryVectorizer2.fit_transform(text for text in corpus)

pickle.dump(binaryVector, open("../ObjetosPreProcessados/binaryVector3k.aug", "wb+"))
pickle.dump(binaryVecto2, open("../ObjetosPreProcessados/binaryVectorTotal.aug", "wb+"))

#pickle.load() recupera o arquivo

# TF
tfVectorizer = CountVectorizer(analyzer = "word",max_features=3000, stop_words = 'english')
tfVectorizer2 = CountVectorizer(analyzer = "word", stop_words = 'english')

tfVector = tfVectorizer.fit_transform(text for text in corpus)
tfVector2 = tfVectorizer2.fit_transform(text for text in corpus)

pickle.dump(tfVector, open("../ObjetosPreProcessados/tfVector3k.aug", "wb+"))
pickle.dump(tfVector2, open("../ObjetosPreProcessados/tfVectorTotal.aug", "wb+"))

# TF-IDF

tfidfVectorizer = TfidfVectorizer(max_features=3000, stop_words = 'english')
tfidfVectorizer2 = TfidfVectorizer(stop_words = 'english')

tfidfVector = tfidfVectorizer.fit_transform(text for text in corpus)
tfidfVector2 = tfidfVectorizer2.fit_transform(text for text in corpus)

pickle.dump(tfidfVector, open("../ObjetosPreProcessados/tfidfVector3k.aug", "wb+"))
pickle.dump(tfidfVector2, open("../ObjetosPreProcessados/tfidfVectorTotal.aug", "wb+"))
