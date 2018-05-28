import os
import pickle
from sklearn.decomposition import TruncatedSVD
import numpy

f = open("../../Objetos/ObjetosPreProcessados Amostra/Word2Vec.aug", "rb")
data = pickle.load(f)
data_np = numpy.array(data)
lsa = TruncatedSVD(n_components = 100)
new_data = lsa.fit_transform(data_np)
pickle.dump(new_data, open("../../Objetos/ObjetosPreProcessados Amostra/Word2VecLSA.aug", "wb+"))