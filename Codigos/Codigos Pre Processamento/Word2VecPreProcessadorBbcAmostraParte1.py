import os, sys, nltk, logging, pickle, gensim
import numpy as np
import tokenizer 
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from word2vectools import word_averaging_list, w2v_tokenize_text, word_averaging

text_path = '../../bbc amostra/'
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

pickle.dump(corpusPreProcessado, open("../../Objetos/ObjetosPreProcessados Amostra/Word2Vec.aug", "wb+"))