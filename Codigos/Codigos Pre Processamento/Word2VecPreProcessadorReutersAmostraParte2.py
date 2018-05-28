import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tokenizer import tokenizer, tokenizer_lemmatizer
import numpy

f = open("../../Objetos/Objetos Preprocessados Reuters Amostra/Word2Vec.aug", "rb")
data = pickle.load(f)
data_np = numpy.array(data_np)
lsa = TruncatedSVD(n_components=100)
new_data = lsa.fit_transform(data_np)
pickle.dump(new_data, open("../../Objetos/Objetos Preprocessados Reuters Amostra/Word2VecLSA.aug", "wb+"))

