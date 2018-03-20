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

tfidfVectorizer = TfidfVectorizer(max_features=3000, stop_words = 'english')
tfidfVectorizer2 = TfidfVectorizer(stop_words = 'english')

tfidfVector = tfidfVectorizer.fit_transform(text for text in corpus)
tfidfVector2 = tfidfVectorizer2.fit_transform(text for text in corpus)

pickle.dump(tfidfVector, open("../ObjetosPreProcessados/tfidfVector3k.aug", "wb+"))
pickle.dump(tfidfVector2, open("../ObjetosPreProcessados/tfidfVectorTotal.aug", "wb+"))