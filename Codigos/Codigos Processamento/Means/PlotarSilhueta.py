import matplotlib.pyplot as plt
import silhuetaDInamico as sd
import numpy as np
import ultra_omega_alpha_kmeans as uoak
import pickle
import os
import pathlib


def plotar_silhueta(silhueta_dados):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    flat_silhueta_dados = []
    for cluster in silhueta_dados:
        cluster = list(cluster)
        cluster.sort(key=lambda x: x) #sort ordem decrescente
        for dado in cluster:
            flat_silhueta_dados.append(dado)
    Y = np.arange(0, len(flat_silhueta_dados))
    cores = plt.spectral(3/len(silhueta_dados))
    ax.fill_betweenx(Y, flat_silhueta_dados, facecolor = cores)
    plt.show()

if __name__ == "__main__":
    filepath = os.path.realpath(
                            "../../../Objetos/ObjetosProcessados/Word2Vec/Word2Vec.aug")
    abspath = pathlib.Path(filepath).absolute()
    with open(abspath, 'rb') as matter:
        dados = pickle.load(matter)
    dados = np.array(dados)
    kmeans = uoak.ultra_omega_alpha_kmeans(no_clusters=5, inicializacao='++')
    kmeans.incluir(dados)
    kmeans.inicializar()
    kmeans.executar()
    silhueta_dados = sd.calcularSilhuetaDados(kmeans)
    plotar_silhueta(silhueta_dados)    
    



