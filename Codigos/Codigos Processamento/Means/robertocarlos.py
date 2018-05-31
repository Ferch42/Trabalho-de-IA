import ultra_omega_alpha_kmeans
import pickle
import numpy as np

come_xuchu = pickle.load(open("../../../Objetos/ObjetosPreProcessados/TF/3K/Normal/tfVector3kLSA.aug","rb"))
come_xuchu = np.array(come_xuchu, dtype=np.float64)

kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=2)
kmeans.incluir(come_xuchu)
kmeans.inicializar()
kmeans.executar_x_means(7)
a=[]
for f in kmeans.clusters:
    a=a+f
print(len(a))
print(kmeans.no_clusters)
print(len(kmeans.clusters))
print(str(kmeans.clusters))