
# coding: utf-8

# In[4]:

import somoclu
import sys, os
import pathlib
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import progressbar
from sklearn.cluster import KMeans
import silhuetaDInamico


# In[ ]:

class Kmeanz:
    pass


# In[5]:

def clusterizacao(som):
    resposta=[[] for _ in range(som.clusters.max()+1)]
    for pos,valor in enumerate(som.bmus):
        my_cluster= som.clusters[valor[0]][valor[1]]
        resposta[my_cluster].append(pos)
    return resposta


# In[2]:



if __name__ == '__main__':
    path_arquivos = "../../../Objetos/ObjetosPreProcessados Amostra/"
    # Retorna tudo oque tem dentro de ObjetosPreProcessados -> só há pastas
    tipos_de_representacao = os.listdir(path_arquivos)
    escolha_da_representacao = sys.argv[1]  # entrada via prompt (string)
    

    if escolha_da_representacao not in tipos_de_representacao:
        raise ValueError("Voce nao digitou uma entrada valida")

    #tipos_de_tamanho = os.listdir(path_arquivos + escolha_da_representacao)
    tipos_de_tamanho = ["3k"]

    algoritmos_do_kmenzao = ["media"]
    distancias_do_kmenzao = ["euclidiana", "manhattan", "cosseno"]

    resposta = []


    for tipo_de_tamanho in tipos_de_tamanho:
        #tipos_de_tipo = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho)
        tipos_de_tipo = ["Normal","Lemma"]
        for tipo_de_tipo in tipos_de_tipo: #Normal ou Lema
            objetos = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo)

            for objeto in objetos:
                with open(
                        path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo + "/" + objeto,
                        "rb") as f1:
                    come_xuchu = pickle.load(f1) #Abre Representa;áo

                if (not isinstance(come_xuchu, np.ndarray)):
                    come_xuchu = np.array(come_xuchu.todense(), dtype=np.float64)
                lsa = False

                # print("TSNING...")
                # tsne = TSNE(n_components=3)
                # transform_come_xuchu = tsne.fit_transform(come_xuchu)
                # pickle.dump(transform_come_xuchu,open("../../../Objetos/ObjetosProcessados/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/"+ "bbc_" + escolha_da_representacao + "_" + tipo_de_tamanho + "_" + tipo_de_tipo + "_LSA" + str(lsa) +".tsne","wb"))

                if ("LSA" in objeto):
                    lsa = True

                print("Somando...", escolha_da_representacao, tipo_de_tamanho, tipo_de_tipo, lsa, ":D") 
                for grid_size in [7,10,14]:
                    for learning_rate in [0.1,0.4,0.7]:
                        for neighboorhood_radius  in [int(grid_size/2),2,1]:
                            for r_cooling in ['linear','exponential']:
                                for a_cooling in ['linear','exponential']:
                                    print("grid: "+str(grid_size)+"; learning_rate: "+str(learning_rate)+"; neighboorhood_radius: "+str(neighboorhood_radius)+"; r_cooling: "+r_cooling+"; a_cooling: "+a_cooling)
                                    som = somoclu.Somoclu(grid_size, grid_size,verbose =2)
                                    som.train(data=come_xuchu,epochs=1000,radius0=neighboorhood_radius,radiusN=1,radiuscooling=r_cooling,scale0=learning_rate,scaleN=0.01,scalecooling=a_cooling)
                                    
                                    for n in range(2,9):
                                        print("kmeans: n = "+str(n))
                                        silhueta_acumulador=0
                                        for _ in range(30):
                                            flaag=True
                                            while(flaag):
                                                try:
                                                    kmeans=KMeans(n_clusters=n)
                                                    som.cluster(kmeans)
                                                    kmeanz = Kmeanz()
                                                    kmeanz.clusters= clusterizacao(som)
                                                    kmeanz.dados= come_xuchu
                                                    silhueta_final = silhuetaDInamico.calcularSilhueta(kmeanz)
                                                    silhueta_acumulador = silhueta_acumulador + silhueta_final
                                                    flaag=False
                                                except:
                                                    print("error founddd")
                                        
                                        silhueta_acumulador = silhueta_acumulador/30
                                        
                                        come_xuchu_dict = {}
                                        come_xuchu_dict["corpus"] = "bbc"
                                        come_xuchu_dict["representacao"] = escolha_da_representacao
                                        come_xuchu_dict["tamanho"] = tipo_de_tamanho
                                        come_xuchu_dict["processamento"] = tipo_de_tipo
                                        come_xuchu_dict["LSA"] = lsa
                                        come_xuchu_dict["ncluster"] = n
                                        come_xuchu_dict["grid_size"] = grid_size
                                        come_xuchu_dict["learning_rate"] = learning_rate
                                        come_xuchu_dict["neighboorhood_radius"] = neighboorhood_radius
                                        come_xuchu_dict["r_cooling"] = r_cooling
                                        come_xuchu_dict["a_cooling"] = a_cooling
                                        
                                        resposta.append((silhueta_acumulador, come_xuchu_dict))
    
    pickle.dump(resposta,open("som"+escolha_da_representacao + str(numero_de_cluster) + ".jojo", "wb"))
                                        
                        
            


