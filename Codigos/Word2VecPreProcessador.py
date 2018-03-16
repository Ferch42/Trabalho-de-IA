import os, sys, nltk, logging, pickle, gensim
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

text_path = '../bbc/'
folders = os.listdir(text_path) # vai devolver os nomes da pasta
corpus = []

for folder in folders:
	for file in os.listdir(text_path+folder):
		f = open(text_path+folder+'/'+file).read()
		corpus.append(f) # add os arqvos textos

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        print("FUDEU!!")

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(word)
    return tokens

print("abrindo os Word2Vec")
wv = KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz", binary = True)
wv.init_sims(replace=True)

print("tokenizando")
corpusTokenizado = [w2v_tokenize_text(text) for text in corpus]
print("preprocessando")
corpusPreProcessado = [word_averaging_list(wv, tokenizedText) for tokenizedText in corpusTokenizado]

pickle.dump(corpusPreProcessado, open("../ObjetosPreProcessados/Word2Vec.aug", "wb+"))