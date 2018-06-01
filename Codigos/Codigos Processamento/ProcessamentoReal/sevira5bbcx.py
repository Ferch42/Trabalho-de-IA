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

if __name__ == "__main__":
    resposta = []
    print("xzando")
    #Melhores representacoes
    #1
    come_xuchu1 = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Normal/tfVector3kLSA.aug","rb"))
    if (not isinstance(come_xuchu1, np.ndarray)):
        come_xuchu1 = np.array(come_xuchu1.todense(), dtype=np.float64)
    #2
    come_xuchu2 = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Lemma/tfVector3kTokenizerLemmatizerLSA.aug","rb"))
    if (not isinstance(come_xuchu2, np.ndarray)):
        come_xuchu2 = np.array(come_xuchu2.todense(), dtype=np.float64)
    #3
    come_xuchu3 = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Lemma/tfVector3kTokenizerLemmatizerLSA.aug","rb"))
    if (not isinstance(come_xuchu3, np.ndarray)):
        come_xuchu3 = np.array(come_xuchu.todense(), dtype=np.float64)
    
    top_3 = [(True, 'media', 'bbc', 'manhattan', 'x', 'Normal', 'TF', '3k', come_xuchu1), (True, 'media', 'bbc', 'cosseno', 'x', 'Lemma', 'TF', '3k', come_xuchu2), (True, 'media', 'bbc', 'manhattan', 'x', 'Lemma', 'TF', '3k', come_xuchu3)]

    for i,melhores_dos_melhores in enumerate(top_3):
        print(str(i+1) + "ª melhor configuracao")
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
            print(str(j+1))
            flagg=True
            while(flagg):
                try:
                    kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2,distancia=melhores_dos_melhores[3],algoritmo=melhores_dos_melhores[1])
                                        
                    kmeans.incluir(melhores_dos_melhores[8])
                    kmeans.inicializar()

                    kmeans.executar_x_means(7)
                                        #print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
                                        #print(kmeans.no_clusters)

                    silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)


                    silhueta_acumulador = silhueta_acumulador + silhueta_final
                    flagg=False
                    if(silhueta_final > best_score):
                        best_score = silhueta_final
                        best_kmeans = kmeans

                except:
                    print("Error found, but remedied")
        silhueta_acumulador = silhueta_acumulador/5

        come_xuchu_dict["ncluster"] = best_kmeans.no_clusters
        resposta.append((silhueta_acumulador, best_score, best_kmeans, come_xuchu_dict))

    pickle.dump(resposta,open("resualtado_final_bbc_x.lai", "wb"))

    '''
    #################################################################################################################################################
    #Melhor representacao
    #######################################################################################################################################
    come_xuchu = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Normal/tfVector3kLSA.aug","rb"))
    if (not isinstance(come_xuchu, np.ndarray)):
        come_xuchu = np.array(come_xuchu.todense(), dtype=np.float64)
    print("Rodando... primeira melhor representacao :D")
    come_xuchu_dict = {}
    come_xuchu_dict["corpus"] = "bbc"
    come_xuchu_dict["representacao"] = "TF"
    come_xuchu_dict["tamanho"] = "3K"
    come_xuchu_dict["processamento"] = "Normal"
    come_xuchu_dict["LSA"] = True
    #come_xuchu_dict["ncluster"] = 
    come_xuchu_dict["algoritmo"] = "media"
    come_xuchu_dict["distancia"] = "manhattan"
    come_xuchu_dict["inicializacao"]="x"

    best_score = -1000
    best_kmeans = None
    silhueta_acumulador = 0

    print("(~‾_‾)~  que shit..")
    for i in range(5):
        print(str(i+1))
        flagg=True
        while(flagg):
            try:
                kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2,distancia="manhattan",algoritmo="media")
                                    
                kmeans.incluir(come_xuchu)
                kmeans.inicializar()

                kmeans.executar_x_means(7)
                                    #print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
                                    #print(kmeans.no_clusters)

                silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)


                silhueta_acumulador = silhueta_acumulador + silhueta_final
                flagg=False
                if(silhueta_final > best_score):
                    best_score = silhueta_final
                    best_kmeans = kmeans

            except:
                print("Error found, but remedied")
    silhueta_acumulador = silhueta_acumulador/5

    come_xuchu_dict["ncluster"] = best_kmeans.no_clusters
    resposta.append((silhueta_acumulador, come_xuchu_dict, best_kmeans))

    #################################################################################################################################################
    #Segunda melhor representacao
    #######################################################################################################################################
    
    come_xuchu = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Lemma/tfVector3kTokenizerLemmatizerLSA.aug","rb"))
    if (not isinstance(come_xuchu, np.ndarray)):
        come_xuchu = np.array(come_xuchu.todense(), dtype=np.float64)
    print("Rodando... Segunda melhor representacao :D")
    come_xuchu_dict = {}
    come_xuchu_dict["corpus"] = "bbc"
    come_xuchu_dict["representacao"] = "TF"
    come_xuchu_dict["tamanho"] = "3K"
    come_xuchu_dict["processamento"] = "Lemma"
    come_xuchu_dict["LSA"] = True
    #come_xuchu_dict["ncluster"] = 
    come_xuchu_dict["algoritmo"] = "media"
    come_xuchu_dict["distancia"] = "cosseno"
    come_xuchu_dict["inicializacao"]="x"

    best_score = -1000
    best_kmeans = None
    silhueta_acumulador = 0

    print("(~‾_‾)~  que shit..")
    for i in range(5):
        print(str(i+1))
        flagg=True
        while(flagg):
            try:
                kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2,distancia="cosseno",algoritmo="media")
                                    
                kmeans.incluir(come_xuchu)
                kmeans.inicializar()

                kmeans.executar_x_means(7)
                                    #print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
                                    #print(kmeans.no_clusters)

                silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)


                silhueta_acumulador = silhueta_acumulador + silhueta_final
                flagg=False
                if(silhueta_final > best_score):
                    best_score = silhueta_final
                    best_kmeans = kmeans

            except:
                print("Error found, but remedied")
    silhueta_acumulador = silhueta_acumulador/5
    come_xuchu_dict["ncluster"] = best_kmeans.no_clusters
    resposta.append((silhueta_acumulador, come_xuchu_dict, best_kmeans))

    #################################################################################################################################################
    #Segunda melhor representacao
    #######################################################################################################################################
    
    come_xuchu = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Lemma/tfVector3kTokenizerLemmatizerLSA.aug","rb"))
    if (not isinstance(come_xuchu, np.ndarray)):
        come_xuchu = np.array(come_xuchu.todense(), dtype=np.float64)
    print("Rodando... Terceira melhor representacao :D")
    come_xuchu_dict = {}
    come_xuchu_dict["corpus"] = "bbc"
    come_xuchu_dict["representacao"] = "TF"
    come_xuchu_dict["tamanho"] = "3K"
    come_xuchu_dict["processamento"] = "Lemma"
    come_xuchu_dict["LSA"] = True
    #come_xuchu_dict["ncluster"] = 
    come_xuchu_dict["algoritmo"] = "media"
    come_xuchu_dict["distancia"] = "manhattan"
    come_xuchu_dict["inicializacao"]="x"

    best_score = -1000
    best_kmeans = None
    silhueta_acumulador = 0

    print("(~‾_‾)~  que shit..")
    for i in range(5):
        print(str(i+1))
        flagg=True
        while(flagg):
            try:
                kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2,distancia="manhattan",algoritmo="media")
                                    
                kmeans.incluir(come_xuchu)
                kmeans.inicializar()

                kmeans.executar_x_means(7)
                                    #print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
                                    #print(kmeans.no_clusters)

                silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)


                silhueta_acumulador = silhueta_acumulador + silhueta_final
                flagg=False
                if(silhueta_final > best_score):
                    best_score = silhueta_final
                    best_kmeans = kmeans

            except:
                print("Error found, but remedied")
    silhueta_acumulador = silhueta_acumulador/5
    come_xuchu_dict["ncluster"] = best_kmeans.no_clusters
    resposta.append((silhueta_acumulador,best_score, come_xuchu_dict, best_kmeans))

    pickle.dump(resposta,open("resualtado_final_bbc_x.lai", "wb"))
'''



