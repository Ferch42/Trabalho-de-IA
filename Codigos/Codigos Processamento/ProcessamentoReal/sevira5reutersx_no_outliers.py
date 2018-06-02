import silhuetaDInamico
import sys, os
import pathlib
import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import progressbar
import OutlierSleuthing as outs

if __name__ == "__main__":
    resposta = []
    print("xzando")
    # Melhores representacoes
    # 1
    come_xuchu1 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/TFIDF/3K/Lemma/arq_LSA.aug", "rb"))
    if (not isinstance(come_xuchu1, np.ndarray)):
        come_xuchu1 = np.array(come_xuchu1.todense(), dtype=np.float64)

    # 2
    come_xuchu2 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/Word2Vec/Word2VecLSA.aug", "rb"))
    if (not isinstance(come_xuchu2, np.ndarray)):
        come_xuchu2 = np.array(come_xuchu2.todense(), dtype=np.float64)
    # 3
    come_xuchu3 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/TFIDF/3K/Lemma/arq.aug", "rb"))
    if (not isinstance(come_xuchu3, np.ndarray)):
        come_xuchu3 = np.array(come_xuchu3.todense(), dtype=np.float64)

    top_3 = [(True, 'media', 'reuters', 'euclidiana', 'padrao', 'Lemma', 'TFIDF', '3k', come_xuchu1),
             (True, 'media', 'reuters', 'cosseno', '++', 'Lemma', 'Word2Vec', '3k', come_xuchu2),
             (False, 'media', 'reuters', 'euclidiana', '++', 'Lemma', 'TFIDF', '3k', come_xuchu3)]

    for i, melhores_dos_melhores in enumerate(top_3):
        print(str(i + 1) + "ª melhor configuracao")
        come_xuchu_dict = {}
        come_xuchu_dict["LSA"] = melhores_dos_melhores[0]
        come_xuchu_dict["algoritmo"] = melhores_dos_melhores[1]
        come_xuchu_dict["corpus"] = melhores_dos_melhores[2]
        come_xuchu_dict["distancia"] = melhores_dos_melhores[3]
        come_xuchu_dict["inicializacao"] = melhores_dos_melhores[4]
        come_xuchu_dict["processamento"] = melhores_dos_melhores[5]
        come_xuchu_dict["representacao"] = melhores_dos_melhores[6]
        come_xuchu_dict["tamanho"] = melhores_dos_melhores[7]

        best_score = -1000
        best_kmeans = None
        silhueta_acumulador = 0

        print("(~‾_‾)~  que shit..")
        for j in range(5):
            print(str(j + 1))
            flagg = True
            while (flagg):
                try:
                    kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2,
                                                                               distancia=melhores_dos_melhores[3],
                                                                               algoritmo=melhores_dos_melhores[1])

                    dados, mapeador = outs.return_new_ndarray_indices(melhores_dos_melhores[8], 0.07)
                    kmeans.incluir(dados)
                    kmeans.inicializar()

                    kmeans.executar_x_means(7)
                    # print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
                    # print(kmeans.no_clusters)

                    silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)
                    for cluster in kmeans.clusters:
                        for i in range(len(cluster)):
                            cluster[i] = mapeador[cluster[i]]

                    silhueta_acumulador = silhueta_acumulador + silhueta_final
                    flagg = False
                    if (silhueta_final > best_score):
                        best_score = silhueta_final
                        best_kmeans = kmeans

                except:
                    print("Error found, but remedied")
        silhueta_acumulador = silhueta_acumulador / 5

        come_xuchu_dict["ncluster"] = best_kmeans.no_clusters
        resposta.append((silhueta_acumulador, best_score, best_kmeans, come_xuchu_dict))

    pickle.dump(resposta, open("resualtado_final_reuters_x.lai", "wb"))
