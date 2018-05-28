import os, sys, nltk, logging, pickle, gensim
import numpy as np
import tokenizer 
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from word2vectools import word_averaging_list, w2v_tokenize_text, word_averaging
import nltk
from sklearn.decomposition import TruncatedSVD
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

text_path = '../../reuters/text/'
folders = os.listdir(text_path) # vai devolver os nomes da pasta
corpus = []

for file in folders:
	
	f = open(text_path+file).read()
	corpus.append(f) # add os arqvos textos


print("abrindo os Word2Vec")
wv = gensim.models.KeyedVectors.load_word2vec_format("../../GoogleNews-vectors-negative300.bin.gz",binary=True)
wv.init_sims(replace=True)

print("tokenizando")
corpusTokenizado = [w2v_tokenize_text(text) for text in corpus]
print("preprocessando")
corpusPreProcessado = [word_averaging(wv, tokenizedText) for tokenizedText in corpusTokenizado]

pickle.dump(corpusPreProcessado, open("../../Objetos/Objetos Preprocessados Reuters/Word2Vec.aug", "wb+"))
print("word2vec.aug salvo com sucesso!!!")

lsa = TruncatedSVD(n_components=100)
npCorpusPreProcessado = np.array(corpusPreProcessado)
corpusPreProcessadoLSA = lsa.fit_transform(npCorpusPreProcessado)
pickle.dump(corpusPreProcessadoLSA, open("../../Objetos/Objetos Preprocessados Reuters/Word2VecLSA.aug", "wb+"))
print("word2vecLSA.aug salvo com sucesso!!!")


