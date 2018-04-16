import sys,os
import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ultra_omega_alpha_kmeans_multicore

path_arquivos = "../../../Objetos/Objetos Preprocessados Reuters/"
#Retorna tudo oque tem dentro de ObjetosPreProcessados -> só há pastas
tipos_de_representacao = os.listdir(path_arquivos)
escolha_da_representacao = sys.argv[1] #entrada via prompt (string)
numero_de_cluster = int(sys.argv[2])

if(escolha_da_representacao not in tipos_de_representacao): 
    raise ValueError("Voce nao digitou uma entrada valida")
    
tipos_de_tamanho = os.listdir(path_arquivos + escolha_da_representacao)

algoritmos_do_kmenzao = ["media","mediana"]
distancias_do_kmenzao = ["euclidiana","manhattan","cosseno"]

for tipo_de_tamanho in tipos_de_tamanho:
    tipos_de_tipo = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho)
    
    for tipo_de_tipo in tipos_de_tipo:
        objetos = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo )

        for objeto in objetos:
            come_xuchu = pickle.load(open(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo + "/" + objeto ,"rb"))
            if(not isinstance(come_xuchu,np.ndarray)):
            	come_xuchu=np.array(come_xuchu.todense(), dtype = np.float64)
            lsa = False

            print("TSNING...")
            tsne = TSNE(n_components=3)      
            #transform_come_xuchu = tsne.fit_transform(come_xuchu)     
            #pickle.dump(transform_come_xuchu,open("../../../Objetos/ObjetosProcessados/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/"+ "bbc_" + escolha_da_representacao + "_" + tipo_de_tamanho + "_" + tipo_de_tipo + "_LSA" + str(lsa) +".tsne","wb")) 

            if("LSA" in objeto):
                lsa = True

            print("Rodando...",escolha_da_representacao,tipo_de_tamanho,tipo_de_tipo,lsa,":D")

            for algoritmo in algoritmos_do_kmenzao:
                
                for distancia in distancias_do_kmenzao:
                	kmeans = None
                	if(tipo_de_tamanho == "Total"):
                		kmeans = ultra_omega_alpha_kmeans_multicore.ultra_omega_alpha_kmeans(no_clusters = numero_de_cluster, algoritmo = algoritmo, distancia = distancia)
                	else:
						kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters = numero_de_cluster, algoritmo = algoritmo, distancia = distancia)
                    
                    print("algoritmando",algoritmo,distancia,numero_de_cluster)
                    #Dicionario q guardará todas as infos desse objeto
                    come_xuchu_dict = {}
                    come_xuchu_dict["corpus"] = "bbc"
                    come_xuchu_dict["representacao"] = escolha_da_representacao
                    come_xuchu_dict["tamanho"] = tipo_de_tamanho
                    come_xuchu_dict["processamento"] = tipo_de_tipo
                    come_xuchu_dict["LSA"] = lsa
                    come_xuchu_dict["ncluster"] = numero_de_cluster
                    come_xuchu_dict["algoritmo"] = algoritmo
                    come_xuchu_dict["distancia"] = distancia                  
                    #Até aqui o objeto está carregado na memoria - OK
                    comeu_chuxu = "bbc_" + escolha_da_representacao + "_" + tipo_de_tamanho + "_" + tipo_de_tipo + "_LSA" + str(lsa) + "_" + algoritmo + "_" + distancia + "_" + str(numero_de_cluster)
                    
                    kmeans.incluir(come_xuchu)
                    kmeans.inicializar()
                    kmeans.executar()

                    os.mkdir("../../../Objetos/ObjetosProcessados Reuters/" + escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/" + comeu_chuxu)
                    pickle.dump(come_xuchu_dict,open("../../../Objetos/ObjetosProcessados Reuters/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/" + comeu_chuxu +"/"+ comeu_chuxu+".info","wb"))
                    pickle.dump(kmeans,open("../../../Objetos/ObjetosProcessados Reuters/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/" + comeu_chuxu +"/"+ comeu_chuxu+".cluster","wb"))
                    '''
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    kmeans_clusters_coloridos = [[transform_come_xuchu[i] for i in cluster] for cluster in kmeans.clusters]

                    cluster_count=1
                    for kmeans_cluster_colorido in kmeans_clusters_coloridos:
                        x = [v[0] for v in kmeans_cluster_colorido]
                        y = [v[1] for v in kmeans_cluster_colorido]
                        z = [v[2] for v in kmeans_cluster_colorido]

                        
                        ax.scatter(x,y,z,c=np.random.rand(4),marker='o',label='Cluster '+str(cluster_count))
                        cluster_count+=1

                    pickle.dump(fig,open("../../../Objetos/ObjetosProcessados/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/" + comeu_chuxu +"/"+ comeu_chuxu+".art","wb"))
                    ''' 


