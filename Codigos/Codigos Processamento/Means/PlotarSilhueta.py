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
    Y = np.arange(0, len(silhueta_dados))
    ax.fill_betweenx(Y, 0, silhueta_dados)
    plt.plot()

if __name__ == "__main__":
    filepath = os.path.realpath(
                            "../../../Objetos/ObjetosProcessados/Word2Vec/Word2VecLSA.aug")
    abspath = pathlib.Path(filepath).absolute()
    with open(abspath, 'rb') as matter:
        dados = pickle.load(matter)
    kmeans = uoak.ultra_omega_alpha_kmeans(no_clusters=5, inicializacao='++')
    kmeans.incluir(dados)
    kmeans.inicializar()
    kmeans.executar()
    silhueta_dados = sd.calcularSilhuetaDados(kmeans)
    plotar_silhueta(silhueta_dados)    
    



