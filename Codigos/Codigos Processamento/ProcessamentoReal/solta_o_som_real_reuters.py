import somoclu
import sys, os
import pathlib
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import progressbar
from sklearn.cluster import KMeans
import silhuetaDInamico
import time
from joblib import Parallel, delayed

# In[ ]:

class Kmeanz:
    pass


# In[5]:

def clusterizacao(som):
    resposta=[[] for _ in range(som.clusters.max()+1)]
    for pos,valor in enumerate(som.bmus):
        my_cluster= som.clusters[valor[0]][valor[1]]
        resposta[my_cluster].append(pos)
    return resposta


# In[2]:






if __name__ == '__main__':
    #Melhores representacoes
    #1
    come_xuchu1 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/Word2Vec/Word2VecLSA.aug","rb"))
    if (not isinstance(come_xuchu1, np.ndarray)):
        come_xuchu1 = np.array(come_xuchu1, dtype=np.float64)
    #2
    come_xuchu2 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/Word2Vec/Word2VecLSA.aug","rb"))
    if (not isinstance(come_xuchu2, np.ndarray)):
        come_xuchu2 = np.array(come_xuchu2, dtype=np.float64)
    #3
    come_xuchu3 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/Word2Vec/Word2Vec.aug","rb"))
    if (not isinstance(come_xuchu3, np.ndarray)):
        come_xuchu3 = np.array(come_xuchu3, dtype=np.float64)
    #4
    come_xuchu4 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/TF/3k/Normal/arq_LSA.aug","rb"))
    if (not isinstance(come_xuchu4, np.ndarray)):
        come_xuchu4 = np.array(come_xuchu4.todense(), dtype=np.float64)
    #5
    come_xuchu5 = pickle.load(open("../../../Objetos/Objetos Preprocessados Reuters/TFIDF/3k/Lemma/arq_LSA.aug","rb"))
    if (not isinstance(come_xuchu5, np.ndarray)):
        come_xuchu5 = np.array(come_xuchu5.todense(), dtype=np.float64)
    
    
    top_5 = [(True, 'exponential','reuters', 10, 0.7, 2, 3, 'Normal', 'linear', 'Word2Vec', '3k', come_xuchu1, 'bubble'), (True, 'exponential', 'reuters', 15, 0.1, 2, 5, 'Normal', 'linear', 'Word2Vec', '3k', come_xuchu2, 'bubble'), (False, 'exponential', 'reuters', 20, 0.1, 2, 7, 'Normal', 'linear', 'Word2Vec', '3k', come_xuchu3, 'bubble'), (True, 'linear', 'reuters', 10, 0.7, 2, 2, 'Normal', 'linear', 'TF', '3k', come_xuchu4, 'gaussian'), (True, 'exponential', 'reuters', 20, 0.1, 3, 7, 'Lemma', 'linear', 'TFIDF', '3k', come_xuchu5, 'bubble')]
    resposta = []

    for i,melhor in enumerate(top_5):
        print(str(i+1) + "ª melhor configuracao")
        
         best_score = -1000
        best_kmeanz = None
        best_som = None
        silhueta_acumulador = 0
        
        for j in range(5)
            print(j+1)
            som = somoclu.Somoclu(melhor[3], melhor[3], neighborhood=melhor[12])
            som.train(data=melhor[11],epochs=1000,radius0=melhor[6],radiusN=1,radiuscooling=melhor[8],scale0=melhor[4],scaleN=0.01,scalecooling=melhor[1])
            kmeans=KMeans(n_clusters=melhor[5])
            som.cluster(kmeans)
            kmeanz = Kmeanz()
            kmeanz.clusters= clusterizacao(som)
            kmeanz.dados= melhor[11]
            silhueta_final = silhuetaDInamico.calcularSilhueta(kmeanz)
            silhueta_acumulador = silhueta_acumulador + silhueta_final
            flagg=False
            if(silhueta_final > best_score):
                best_score = silhueta_final
                best_kmeanz = kmeanz
                best_som = som

        come_xuchu_dict = {}
        come_xuchu_dict["corpus"] = melhor[2]
        come_xuchu_dict["representacao"] = melhor[9]
        come_xuchu_dict["tamanho"] = melhor[10]
        come_xuchu_dict["processamento"] = melhor[7]
        come_xuchu_dict["LSA"] = melhor[0]
        come_xuchu_dict["ncluster"] = melhor[5]
        come_xuchu_dict["grid_size"] = melhor[3]
        come_xuchu_dict["learning_rate"] = melhor[4]
        come_xuchu_dict["neighboorhood_radius"] = melhor[6]
        come_xuchu_dict["r_cooling"] = melhor[8]
        come_xuchu_dict["a_cooling"] = melhor[1]
        come_xuchu_dict["neighborhood"] = melhor[12]
        
        silhueta_acumulador = silhueta_acumulador/5    
        resposta.append((best_som, best_kmeanz, silhueta_acumulador, best_score, come_xuchu_dict))

    pickle.dump(resposta,open("som_real_reuters.lai", "wb"))

        