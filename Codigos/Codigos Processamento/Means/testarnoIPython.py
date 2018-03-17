import ultra_omega_alpha_kmeans
import pickle
import numpy as np

dados = pickle.load(open("../../../ObjetosPreProcessados/Word2Vec/Word2Vec.aug","rb"))
dados = np.array(dados)

kmeansao = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans()
kmeansao.incluir(dados)
kmeansao.inicializar()
kmeansao.centroids
kmeansao.executar()