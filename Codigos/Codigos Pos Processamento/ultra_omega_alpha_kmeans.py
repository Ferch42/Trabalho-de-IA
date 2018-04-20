import numpy as np
import random 
from sklearn.metrics.pairwise import cosine_similarity
import sys
import progressbar

class ultra_omega_alpha_kmeans:

    def __init__(self, no_clusters = 2, inicializacao = 'padrao', distancia = "euclidiana", algoritmo = 'media',no_iteracoes = 500):
        self.no_clusters = no_clusters
        self.inicializacao = inicializacao
        self.algoritmo = algoritmo
        self.no_iteracoes = no_iteracoes
        self.distancia = distancia
        self.clusters = []
        self.dados = None
        self.invalid_positions=None
        self.centroids = []
        self.historia= []
        self.distancia_total=0
        self.distancia_total_ant=-1
        self.distancia_total_ant_ant=-2
        

    def incluir(self, dados):
        #Qual a dimensao dos dados ? Linhas = numeros de instancias(documents) e Colunas = numeros atributos
       
        if not isinstance(dados, np.ndarray):
            dados = np.array(dados)


        if len(dados.shape) != 2 :
            raise ValueError("Problemas na dimensao do array")

        self.dados = dados

    def inicializar(self):
        metodo = self.inicializacao

        if metodo == "padrao":
            self.__inicializarPadrao()
        elif metodo == "++":
            self.__inicializarPlusPlus()
        elif metodo == "x":
            self.__inicializarX()
        else:
            raise ValueError("Escolha entre as opcoes disponiveis:'padrao','++','x' ")

    def __inicializarPadrao(self):
        indices_prototipos = []

        for _ in range(self.no_clusters):       
            #self.dados.shape[0] == numero de linhas da matriz de dados.
            # random.randrange escolhe um numero aleatoria do range especificado.
            indice_aleatorio = random.randrange(self.dados.shape[0])

            while indice_aleatorio in indices_prototipos:# verificando se haverá indices iguais em 
                indice_aleatorio = random.randrange(self.dados.shape[0])

            indices_prototipos.append(indice_aleatorio)

        self.centroids = np.array([self.dados[indice] for indice in indices_prototipos])

    
    
    def __inicializarPlusPlus(self):
        pass
    def __inicializarX(self):
        pass

    def __recalcular_centroid_media(self):
        cluster_contador = 0
        for cluster in self.clusters: # Aqui 'self.clusters' é uma matriz que possui em linhas os cluster e a coluna de cada linha os INDICES dos dados associados a este cluster
            arr_media = np.zeros(self.dados.shape[1])
            for indice in cluster:
                arr_media = arr_media + self.dados[indice]
            arr_media = arr_media/len(cluster)
            self.centroids[cluster_contador] = arr_media # reajustando o centroid de 'cluster' após o calculo da média
            cluster_contador += 1

    def __recalcular_centroid_mediana(self):
        self.centroids = np.array([np.median(np.array([self.dados[i] for i in cluster]),axis=0) for cluster in self.clusters])
    
    def calcula_distancia(self, distancia_selecionada): #O(n*c)
        self.clusters = [[] for i in range(self.no_clusters)] # Aqui armazena-se para cada 'i' em 'no_clusters' uma lista vazia em 'clusters'
        
        #enchendo os clusters com pelo menos aquele elemento que lhe é mais proximo
        self.invalid_positions=[]
        distancias_clusters=[np.array([distancia_selecionada(centroid,dado) for dado in self.dados]) for centroid in self.centroids]
        for c in range(self.no_clusters):
            distancias_cluster=[enu for enu in enumerate(distancias_clusters[c])]
            sorted_dist=sorted(distancias_cluster,key= lambda x:x[1])
            for si in sorted_dist:
                if(si in self.invalid_positions):
                    continue
                else:
                    self.invalid_positions.append(si[0])
                    break
            self.clusters[c].append(self.invalid_positions[-1])

        #Para cada dado em 'dados' calcula-se a distancia deste dado para cada centroid em 'centroids' e armazena em 'arr_distancias'
        distancias_clusters=[np.array([x]).T for x in distancias_clusters]
        arr_distancias = np.concatenate(distancias_clusters,axis=1)
        self.distancia_total=arr_distancias.sum()
        # axis =1 : realiza a operacao sobre cada elemento em uma linha (para cada linha)
        arr_associacoes = np.argmin(arr_distancias,axis=1)  # Aqui verifica-se qual dado pertence a qual centroid analisando pela distancia minima entre eles
        for i in range(len(arr_associacoes)):
            if(i not in self.invalid_positions):
                self.clusters[arr_associacoes[i]].append(i)
        '''
        cen_in = -1
        min_distancia = sys.maxsize
        for id in range(len(dados)):
            for ic in range(len(self.centroids)):
                aux_distancia = min_distancia
                min_distancia = distancia_selecionada(self.centroid[ic],dado)
                if min_distancia > aux_distancia:
                    min_distancia = aux_distancia
                else:
                    cen_in = ic
            self.clusters[cen_in].append(id)     
        '''
    
    #Retorna os clusters finais
    #lista de lista de indices - > Matriz esparca
    def executar(self): 
        
        distancia_euclidiana = lambda x,y: np.sqrt(((x-y)**2).sum())
        distancia_manhattan = lambda x,y:  np.abs(x-y).sum()        
        distancia_cosseno = lambda x,y: 1-cosine_similarity([x],[y])[0][0] # [0][0] retorna o numero puro
        
        dist = self.distancia #Qual distancia será utilizada pelo algoritmo

        distancia_selecionada = None
        if dist == "euclidiana": 
            distancia_selecionada = distancia_euclidiana
        elif dist == "manhattan":
            distancia_selecionada = distancia_manhattan
        elif dist == "cosseno":
            distancia_selecionada = distancia_cosseno
        else:
            raise ValueError("Escolha a distancia entre as seguinte opcoes: 'euclidiana', 'manhattan', 'cosseno'")
        
        alg = self.algoritmo #Vê qual o tipo de calculo será executado -> Media ou Mediana
        
        if alg == "media":
            for _ in progressbar.progressbar(range(self.no_iteracoes)):# no_iteracoes configuração padrão igual a 500
                #hsit={}
                #hsit['centroids']=self.centroids.copy()
                #hsit['clusters']=self.clusters.copy()
                #self.historia.append(hsit)
                if(self.distancia_total==self.distancia_total_ant or self.distancia_total==self.distancia_total_ant_ant):
                    break
                
                self.distancia_total_ant_ant=self.distancia_total_ant
                self.distancia_total_ant=self.distancia_total
                self.calcula_distancia(distancia_selecionada)
                self.__recalcular_centroid_media()
                
        elif alg == "mediana":
            for _ in progressbar.progressbar(range(self.no_iteracoes)):
                
                #hsit={}
                #hsit['centroids']=self.centroids
                #hsit['clusters']=self.clusters
                #self.historia.append(hsit)
                if(self.distancia_total==self.distancia_total_ant or self.distancia_total==self.distancia_total_ant_ant):
                    break
                
                self.distancia_total_ant_ant=self.distancia_total_ant
                self.distancia_total_ant=self.distancia_total
                self.calcula_distancia(distancia_selecionada)
                self.__recalcular_centroid_mediana()
                
        else:
            raise ValueError("Escolha o algoritmo entre as seguintes opcoes: 'media', 'mediana'")
        
        
        
