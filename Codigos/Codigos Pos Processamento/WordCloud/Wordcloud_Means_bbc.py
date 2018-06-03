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
p = pickle.load(open("../../../Objetos/ObjetosProcessadosBbcMeansFinal/resualtado_final_bbc_normal.lai", "rb"))
for result in p:
	resultados.append(result)
p = pickle.load(open("../../../Objetos/ObjetosProcessadosBbcMeansFinal/resualtado_final_bbc_x.lai", "rb"))
for result in p:
	resultados.append(result)
'''


'''
resultados = []
p = pickle.load(open("../../../Objetos/ObjetosProcessadosBbcMeansFinal/resualtado_final_bbc_5clusterz.lai", "rb"))
for result in p:
	resultados.append(result)
'''

resultados = []
p = pickle.load(open("../../../Objetos/ObjetosProcessadosBbcMeansFinal/resualtado_final_bbc_5clusterz_outliers.lai", "rb"))
for result in p:
	resultados.append(result)


resultados = sorted(resultados, key = lambda tupla: tupla[1], reverse = True)
n=20
corpus = 'bbc'
for i,resultado in enumerate(resultados):
	kmeans = resultado[2]
	lista_mais_frequentes = Wordcloud.PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, corpus, kmeans)
	Wordcloud.save_plot(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_bbc")
	Wordcloud.save_plots_por_cluster(lista_mais_frequentes, str(i+1) + "ªmelhor_representacao_bbc")
