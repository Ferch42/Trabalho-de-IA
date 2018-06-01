import sys,os
import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
import matplotlib
import Wordcloud

class Kmeanz:
	pass

resultados = []
path = "../../../Objetos/ObjetosProcessadosSOM/"
for file in sorted(os.listdir(path),reverse=True):
	p = pickle.load(open(path + file, "rb"))
	for result in p:
		resultados.append(result)
n=20
corpus = 'bbc'
for i,resultado in enumerate(resultados):
	kmeans = resultado[1]
	lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, corpus, kmeans)
	Wordcloud.save_plot(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_bbc")
	Wordcloud.save_plots_por_cluster(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_bbc")



resultados = []
path = "../../../Objetos/ObjetosProcessadosSOMReuters/"
for file in os.listdir(path):
	p = pickle.load(open(path + file, "rb"))
	for result in p:
		resultados.append(result)
n=20
corpus = 'reuters'
for i,resultado in enumerate(resultados):
	kmeans = resultado[1]
	lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, corpus, kmeans)
	Wordcloud.save_plot(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
	Wordcloud.save_plots_por_cluster(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
