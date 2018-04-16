# Silhueta de um ponto para cada ponto                    b(i)-a(i)
# Silhueta de um ponto para cada cluster                max{b(i),a(i)}
# Silhueta media das silhuetas para cada cluster total
import numpy as np
import ultra_omega_alpha_kmeans
import math

def calcularSilheta(kmeans):
    S_dos_cluster = []

   #atributos para kmeans    
    #cluster lista de listas
    #dados matriz de dados
    #centroid lista

    # 1 - Teste Silhueta Para cada ponto


distancia_euclidiana = lambda x,y: np.sqrt(((x-y)**2).sum())


#Representa a(i)
def distanciaMediaIntra(conj_dados): #silhueta para dados 
    resultado = [] #guarda distancia médio de cada dado para os outros dados do conjunto de dados
    for dado in conj_dados:
        dist_soma = 0
        for dado_2 in conj_dados:
            dist_soma = dist_soma + distancia_euclidiana(dado,dado_2)
        resultado.append(dist_soma/len(conj_dados)-1)

    return np.array(resultado)

def distanciaMediaExtra(conj_dados,conj_clusters_externo): #sulhueta para dados extracluster
    resultado = []
    for dado in conj_dados:
        dist_soma = 0
        soma_media_min = math.inf

        for clusterI in conj_clusters_externo:
            soma_temp = 0

            for dado_externo in clusterI: 
                soma_temp = soma_temp + distancia_euclidiana(dado,dado_externo)

            soma_temp = soma_temp/len(clusterI)

            if soma_temp < soma_media_min :
                soma_media_min = soma_temp
        resultado.append(soma_media_min)
    
    return np.array(resultado)


def SilhuetaDado(conj_a, conj_b):

    resultado = []
    for a,b in conj_a,conj_b:
        maior_value = 0
        if(a>b):
            maior_value = a
        else:
            maior_value = b
        
        resultado.append((b-a)/maior_value)
    return np.array(resultado)

def SilhuetaGrupo(conj_silhueta_dados):
    return conj_silhueta_dados.sum()/len(conj_silhueta_dados)