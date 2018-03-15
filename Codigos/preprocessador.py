import os,sys,pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


text_path = '../bbc/'
folders = os.listdir(text_path)
corpus = []

for folder in folders:
	for file in os.listdir(text_path+folder):
		f = open(text_path+folder+'/'+file).read()
		corpus.append(f)

binaryVectorizer = CountVectorizer(analyzer = "word", stop_words = 'english', max_features=3000, binary = True)
binaryVector = binaryVectorizer.fit_transform(text for text in corpus)
pickle.dump(binaryVector, open("binaryVector.aug", "wb"))