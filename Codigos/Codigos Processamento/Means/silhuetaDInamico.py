
# coding: utf-8

# In[6]:


import numpy as np
import ultra_omega_alpha_kmeans
import math
import pickle
from sklearn.metrics import euclidean_distances
import sys
from joblib import Parallel, delayed
#from dadosdor import recupera_dados 
#distancia_euclidiana = lambda x,y: np.sqrt(((x-y)**2).sum())



# In[ ]:

def calcular_silhueta_um_grupo(kmeans):
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

    return silhueta_grupos



# In[7]:


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


# In[8]:


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


# In[27]:


def distanciaMediaExtra4x4(conj_cluster_dados):   #paired_distances
    

   
    resultado = [] #resultado parcial das distancias extras para o cluster cN
    
    maxZ = np.max([len(x) for x in conj_cluster_dados]) #selected a maior lista que existe para um cluster
       
    
    matriz_distancias = np.full((len(conj_cluster_dados),maxZ,len(conj_cluster_dados),maxZ), np.nan)
    
    #dist de todos para todos
    distancia_geral = Parallel(n_jobs=-1,  backend="threading") (delayed(euclidean_distances)(conj_cluster_dados[x],conj_cluster_dados[y]) for x in range(len(conj_cluster_dados)) for y in range(len(conj_cluster_dados)) if(y > x))
    
    dist_inter = iter(distancia_geral)
    
    for i in range(len(conj_cluster_dados)):
        
        
        for j in range(len(conj_cluster_dados)):
            
            if(j>i):                
                
                cluster_analisado = next(dist_inter)
                
                for cI,clusterI in enumerate(cluster_analisado):
                    
                    for k in range(len(clusterI)):
                        
                        matriz_distancias[i][cI][j][k] = cluster_analisado[cI][k]
                        matriz_distancias[j][k][i][cI] = cluster_analisado[cI][k]
                        
                        
    for i in range(len(conj_cluster_dados)):
        
        resultado_cluster_parcial = []
        
        for j in range(len(conj_cluster_dados[i])):
            
            soma_media_min = np.inf
            
            for clusterII in range(len(conj_cluster_dados)):
                
                if(clusterII != i):
                    
                    media_distancia = 0
                    
                    for dadoII in range(maxZ):
                        
                        if(np.isnan(matriz_distancias[i][j][clusterII][dadoII])):
                            break
                            
                        else: 
                            media_distancia = media_distancia + matriz_distancias[i][j][clusterII][dadoII]
                            
                    media_distancia = media_distancia/len(conj_cluster_dados[clusterII])
                    
                    if(media_distancia < soma_media_min):
                        soma_media_min = media_distancia
                        
            resultado_cluster_parcial.append(soma_media_min)
            
        resultado.append(resultado_cluster_parcial)
        
    return np.array(resultado)
            
        
  


# In[ ]:


def distanciaMediaIntra(conj_dados): #silhueta para dados 

    distancia_geral = euclidean_distances(conj_dados, conj_dados)

    #cada linha é a distancia do ponto para todo resto
    #media d alinha

    return np.array([x.sum()/(len(x)-1) for x in distancia_geral]) 




# In[15]:


def SilhuetaDado(conj_cluster_dados):

    #list_conj_a = [distanciaMediaIntra(conj_cluster_dados[x]) for x in range(len(conj_cluster_dados))] # passando o conjunto de dados referente ao cluster em avaliação
    
    list_conj_a = Parallel(n_jobs=-1,  backend="threading")  (delayed(distanciaMediaIntra)(conj_cluster_dados[x]) for x in range(len(conj_cluster_dados))) # passando o conjunto de dados referente ao cluster em avaliação
    
    
    list_conj_b = distanciaMediaExtra4x4(conj_cluster_dados)
    

    resultado = []

    for a,b in zip(list_conj_a,list_conj_b):
        a = np.array(a) # lista de distancias intra
        b = np.array(b) # lista de distancias extra

        c = np.concatenate([[a],[b]], axis = 0).max(axis = 0) #este é o denominador da silhueta- é o maximo entre a e b para todos os valores do vetor
        
        silhueta = (b-a)/c #Daora
        
        resultado.append(silhueta)
    return np.array(resultado) #valor de silhueta para cada dado do cluster para todos os cluster


# In[16]:


def SilhuetaGrupo(conj_silhueta_dados):
   return conj_silhueta_dados.sum()/len(conj_silhueta_dados)


# In[17]:


def SilhuetaTotal(conj_silhueta_grupo):
    conj = np.array(conj_silhueta_grupo)
    return conj.sum()/len(conj) 


