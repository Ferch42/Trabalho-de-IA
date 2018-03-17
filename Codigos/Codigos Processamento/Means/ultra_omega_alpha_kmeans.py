import numpy as np
import random 
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt


class ultra_omega_alpha_kmeans:

    def __init__(self, no_clusters = 2, inicializacao = 'padrao', distancia = "euclidiana", algoritmo = 'media',no_iteracoes = 500):
        self.no_clusters = no_clusters
        self.inicializacao = inicializacao
        self.algoritmo = algoritmo
        self.no_iteracoes = no_iteracoes
        self.distancia = distancia
        self.dados = None
        self.centroids = None

    def incluir(self, dados):
        #Qual a dimensao dos dados ? Linhas = numeros de instancias e Colunas = numeros atributos
       
        if not isinstance(dados, np.ndarray):
            dados = np.array(dados)

        if len(dados.shape) != 2 :
            raise ValueError("Problemas na dimensao do array")

        self.dados = dados

    def __inicializar(self):
        metodo = self.inicializacao

        if metodo == "padrao":
            self.__inicializarPadrao()
        elif metodo == "++":
            self.inicializarPlusPlus()
        elif metodo == "x":
            self.inicializarX()

        else:
            raise ValueError("Escolha entre as opcoes disponiveis:'padrao','++','x' ")

    def __inicializarPadrao(self):
        indices_prototipos = []

        for i in range(self.no_clusters):       
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

    


    
 
    #Calcula a proxima matriz
    def __prox_U(self):
        pass
    
    

    #Retorna os clusters finais
    #lista de lista de indices - > Matriz esparca
    def rodaObagulho(self):

        distancias = [[] for i in range(self.no_clusters)] 
        
        distancia_euclidiana = lambda x,y: sqrt((x-y)**2)
        distancia_manhattan = lambda x,y: abs(x-y)        
        distancia_cosseno = lambda x,y: 1-cosine_similarity([x],[y])[0][0] # [0][0] retorna o numero puro
        dist = self.distancia

        distancia_selecionada = None
        if (dist == "euclidiana"): 


   

        pass

    
    