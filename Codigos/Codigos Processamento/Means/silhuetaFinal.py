import numpy as np
#import ultra_omega_alpha_kmeans
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import sys
#from dadosdor import recupera_dados 


distancia_euclidiana = lambda x,y: np.sqrt(((x-y)**2).sum())

def calcularSilhueta(kmeans):

    silhueta_dados = []

    my_cluster =  kmeans.clusters #coordenadas para os dados 
    my_dados = kmeans.dados # matriz de dados - Para acessar um dado devemos consultar a cordenada em my_cluters (lista de listas)

    conj_daora = criarConjunto(my_cluster, my_dados)
    
    #Limpando memória
    my_cluster = None
    my_dados = None

    silhueta_dados = SilhuetaDado(conj_daora) 
    
    silhueta_grupos = [SilhuetaGrupo(grupo) for grupo in silhueta_dados]

    silhueta_grupos = np.array(silhueta_grupos)

    result = SilhuetaTotal(silhueta_grupos)

    return result


def criarConjunto(clusters, dados):
    conj_daora = [[] for _ in range(len(clusters))]
    i = 0
    for my_cluster in clusters: #my_clus
        #conj_daora.append([])

        for coordenadas in my_cluster:

            mydado = dados[coordenadas]
            conj_daora[i].append(mydado)

        i = i+1
    return conj_daora

#
#devolve uma lista de organizada pelo numero do cluster
def distanciaMediaExtra(conj_cluster_dados): #silhueta para dados extracluster    

    resultado = [[] for _ in range(len(conj_cluster_dados))]
    
    hash_dist = {} 
    #STRUCT ---> key (cluster1_indiceDado1_cluster2_indiceDado2): value = distancia(dado1,dado2) <---
    

    for cN,cluster in enumerate(conj_cluster_dados):
        
        resultado_cluster = [] #resultado parcial das distancias extras para o cluster cN

        for dN,dado in enumerate(cluster): 

            soma_media_min = np.inf

            for cK,cluster2 in enumerate(conj_cluster_dados): #escolhendo outro cluster

                if(cN == cK): #garante que é um cluster diferente
                    continue

                soma_media_cluster2 = 0

                for dK, dado2 in enumerate(cluster2): 
                    #ordenando a key do dict
                    myKey = str(sorted([(cN,dN),(cK,dK)])) #Muito TOP #cria um
                    aux = 0 

                    if(myKey in hash_dist.keys()): #distancia conhecida? Sim, então não calcula distancia
                        aux = hash_dist[myKey]
                    else:
                        myDist = distancia_euclidiana(dado,dado2) # distancia conhecida? Não , então calcula dist
                        hash_dist[myKey] = myDist
                        aux = myDist

                    soma_media_cluster2 = soma_media_cluster2 + aux # soma dist dos dados

                soma_media_cluster2 = soma_media_cluster2/ len(cluster2) #calc media das distancia

                if(soma_media_cluster2 < soma_media_min): # é a menor média? 
                    soma_media_min = soma_media_cluster2 
            
            resultado_cluster.append(soma_media_min) #salvando o valor da menor média do dado em resultado_cluster

        resultado.append(resultado_cluster) #salva a média mínima  para todos os cluster e seus respectivos dados no resultado

    return resultado #retorno
            
            

def distanciaMediaIntra(conj_dados): #silhueta para dados 
       
    resultado = [] #guarda distancia médio de cada dado para os outros dados do conjunto de dados
    matriz_distancias = np.full((len(conj_dados),len(conj_dados)), np.nan)
    
    for i,dado in enumerate(conj_dados):
        dist_soma = 0

        for j,dado_2 in enumerate(conj_dados):
            aux = 0
            if(np.isnan(matriz_distancias[i][j])):
                aux =  distancia_euclidiana(dado,dado_2)
                matriz_distancias[i][j] = aux
                matriz_distancias[j][i] = aux

            else:
                aux = matriz_distancias[i][j]

            dist_soma = dist_soma + aux

        resultado.append(dist_soma/(len(conj_dados)-1)) ##ANALISAR A FORMULA

    return resultado


def SilhuetaDado(conj_cluster_dados):

    list_conj_a = [distanciaMediaIntra(conj_cluster_dados[x]) for x in range(len(conj_cluster_dados))] # passando o conjunto de dados referente ao cluster em avaliação
    
    list_conj_b = distanciaMediaExtra(conj_cluster_dados)
    

    resultado = []

    for a,b in zip(list_conj_a,list_conj_b):
        a = np.array(a) # lista de distancias intra
        b = np.array(b) # lista de distancias extra

        c = np.concatenate([[a],[b]], axis = 0).max(axis = 0) #este é o denominador da silhueta- é o maximo entre a e b para todos os valores do vetor
        
        silhueta = (b-a)/c #Daora
        
        resultado.append(silhueta)
    return np.array(resultado) #valor de silhueta para cada dado do cluster para todos os cluster

    
def SilhuetaGrupo(conj_silhueta_dados):
    return conj_silhueta_dados.sum()/len(conj_silhueta_dados)


def SilhuetaTotal(conj_silhueta_grupo):
    conj = np.array(conj_silhueta_grupo)
    return conj.sum()/len(conj) 






