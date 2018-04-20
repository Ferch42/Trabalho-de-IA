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

    for cluster in range(len(conj_daora)):
        silhueta_dados.append(SilhuetaDado(cluster, conj_daora))
        
    silhueta_grupos = SilhuetaGrupo(silhueta_dados)

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

def distanciaMediaExtra(numero_do_meu_cluster,conj_cluster_dados): #silhueta para dados extracluster

    conj_dados = conj_cluster_dados[numero_do_meu_cluster] #pegando os dados para o cluster numero_do_meu_cluster

    resultado = []

    for dado in conj_dados:

        dist_soma = 0
        soma_media_min = sys.maxsize
        
        i = 0
        for cluster_externo in conj_cluster_dados:

            if i == numero_do_meu_cluster: # é o cluster do dado que estamos calculando a distancia
                i = i + 1
                continue

            soma_temp = 0

            for dado_externo in cluster_externo: 
                soma_temp = soma_temp + distancia_euclidiana(dado,dado_externo)

            soma_temp = soma_temp/len(dado_externo) #calculando a distancia média 

            if soma_temp < soma_media_min :
                soma_media_min = soma_temp

            i = i + 1

        resultado.append(soma_media_min)
    
    return resultado

def distanciaMediaIntra(conj_dados): #silhueta para dados 
       
    resultado = [] #guarda distancia médio de cada dado para os outros dados do conjunto de dados

    for dado in conj_dados:
        dist_soma = 0
        for dado_2 in conj_dados:
            dist_soma = dist_soma + distancia_euclidiana(dado,dado_2)
        resultado.append(dist_soma/(len(conj_dados)-1)) ##ANALISAR A FORMULA

    return resultado


def SilhuetaDado(cluster_avaliado, conj_cluster_dados):

    conj_a = distanciaMediaIntra(conj_cluster_dados[cluster_avaliado]) # passando o conjunto de dados referente ao cluster em avaliação
    conj_b = distanciaMediaExtra(cluster_avaliado,conj_cluster_dados)

    resultado = []

    for a,b in zip(conj_a,conj_b):
        maior_value = 0
        if(a>b):
            maior_value = a
        else:
            maior_value = b
        print(a,b,maior_value)
        resultado.append((b-a)/maior_value)
    return np.array(resultado) #valor de silhueta para cada dado do cluster

    
def SilhuetaGrupo(conj_silhueta_dados):
    return conj_silhueta_dados.sum()/len(conj_silhueta_dados)


def SilhuetaTotal(conj_silhueta_grupo):
    
    return conj_silhueta_grupo.sum()/len(conj_silhueta_grupo) 






