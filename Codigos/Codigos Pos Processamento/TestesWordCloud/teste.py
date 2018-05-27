import sys,os
import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib
import random
import Wordcloud

path_arquivos = "../../../Objetos/ObjetosPreProcessados/"
come_xuchu = pickle.load(open(path_arquivos + 'TFIDF' + "/" + '3k' + "/" + 'Tokenizer' + "/" + 'tfidfVector3kTokenizerLSA.aug' ,"rb"))
if(not isinstance(come_xuchu,np.ndarray)):
	come_xuchu=np.array(come_xuchu.todense(), dtype = np.float64)
kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=5, algoritmo = "media", distancia = "euclidiana")
kmeans.incluir(come_xuchu)
kmeans.inicializar()
kmeans.executar()

n=30
path='../../../bbc/'
lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, path, kmeans)
Wordcloud.plotar_word_cloud(lista_mais_frequentes)

