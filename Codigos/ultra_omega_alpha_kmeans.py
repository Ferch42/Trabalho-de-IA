import numpy as np

class ultra_omega_alpha_kmeans():

    def __init__(self, dados = None,no_clusters = None,iniciacao = 'c-means++', algoritmo = 'media',no_iteracoes = 500):
        self.dados = dados
        self.no_clusters = no_clusters
        self.iniciacao = iniciacao
        self.algoritmo = algoritmo
        self.no_iteracoes = no_iteracoes

    #Calcula a proxima matriz
    def __prox_U(self):
        pass
    
    

    #Retorna os clusters finais
    def pega_no_rabo(self):
        pass
    