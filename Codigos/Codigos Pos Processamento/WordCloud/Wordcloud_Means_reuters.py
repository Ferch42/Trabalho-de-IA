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
'''
resultados = []
p = pickle.load(open("../../../Objetos/ObjetosProcessadosReutersMeansFinal/resualtado_final_reuters_normal.lai", "rb"))
for result in p:
	resultados.append(result)
resultados = sorted(resultados, key = lambda tupla: tupla[1], reverse = True)

n=20
corpus = 'reuters'
for i,resultado in enumerate(resultados):
	kmeans = resultado[2]
	lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, corpus, kmeans)
	Wordcloud.save_plot(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
	Wordcloud.save_plots_por_cluster(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
'''

#5+

resultados = []
p = pickle.load(open("../../../Objetos/ObjetosProcessadosReutersMeansFinal/resualtado_final_reuters_5+.lai", "rb"))
for result in p:
	resultados.append(result)
resultados = sorted(resultados, key = lambda tupla: tupla[1], reverse = True)
print(resultados)

n=20
corpus = 'reuters'
for i,resultado in enumerate(resultados):
	kmeans = resultado[2]
	lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, corpus, kmeans)
	Wordcloud.save_plot(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
	Wordcloud.save_plots_por_cluster(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_reuters")
