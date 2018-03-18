import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dados = pickle.load(open("../../../ObjetosPreProcessados/TFIDF/tfidfVector3k.aug","rb"))
dados=np.array(dados.todense(),dtype=np.float64)

kmeansao = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=5)
kmeansao.incluir(dados)
kmeansao.inicializar()
kmeansao.centroids
kmeansao.executar()


tsne=TSNE(n_components=3)
transformeddados= tsne.fit_transform(dados)



fig = plt.figure()
ax = plt.axes(projection='3d')

clusters1=[transformeddados[i] for i in kmeansao.clusters[0]]
clusters2=[transformeddados[i] for i in kmeansao.clusters[1]]
clusters3=[transformeddados[i] for i in kmeansao.clusters[2]]
clusters4=[transformeddados[i] for i in kmeansao.clusters[3]]
clusters5=[transformeddados[i] for i in kmeansao.clusters[4]]

xvalues=[vector[0] for vector in clusters1]
yvalues=[vector[1] for vector in clusters1]
zvalues=[vector[2] for vector in clusters1]

ax.scatter(xvalues,yvalues,zvalues,c='r',marker='o')


xvalues=[vector[0] for vector in clusters2]
yvalues=[vector[1] for vector in clusters2]
zvalues=[vector[2] for vector in clusters2]

ax.scatter(xvalues,yvalues,zvalues,c='b',marker='o')


xvalues=[vector[0] for vector in clusters3]
yvalues=[vector[1] for vector in clusters3]
zvalues=[vector[2] for vector in clusters3]

ax.scatter(xvalues,yvalues,zvalues,c='darkgreen',marker='o')


xvalues=[vector[0] for vector in clusters4]
yvalues=[vector[1] for vector in clusters4]
zvalues=[vector[2] for vector in clusters4]

ax.scatter(xvalues,yvalues,zvalues,c='m',marker='o')


xvalues=[vector[0] for vector in clusters5]
yvalues=[vector[1] for vector in clusters5]
zvalues=[vector[2] for vector in clusters5]

ax.scatter(xvalues,yvalues,zvalues,c='y',marker='o')




plt.show()