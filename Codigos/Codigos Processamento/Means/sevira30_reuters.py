import silhuetaDInamico
import sys, os
import pathlib
import ultra_omega_alpha_kmeans
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import progressbar


if __name__ == '__main__':
    path_arquivos = "../../../Objetos/Objetos Preprocessados Reuters Amostra/"
    # Retorna tudo oque tem dentro de ObjetosPreProcessados -> só há pastas
    tipos_de_representacao = os.listdir(path_arquivos)
    escolha_da_representacao = sys.argv[1]  # entrada via prompt (string)
    numero_de_cluster = int(sys.argv[2])

    if escolha_da_representacao not in tipos_de_representacao:
        raise ValueError("Voce nao digitou uma entrada valida")

    #tipos_de_tamanho = os.listdir(path_arquivos + escolha_da_representacao)
    tipos_de_tamanho = ["3k"]

    algoritmos_do_kmenzao = ["media"]
    distancias_do_kmenzao = ["euclidiana", "manhattan", "cosseno"]

    resposta = []


    for tipo_de_tamanho in tipos_de_tamanho:
        #tipos_de_tipo = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho)
        tipos_de_tipo = ["Normal","Lemma"]
        for tipo_de_tipo in tipos_de_tipo: #Normal ou Lema
            objetos = os.listdir(path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo)

            for objeto in objetos:
                with open(
                        path_arquivos + escolha_da_representacao + "/" + tipo_de_tamanho + "/" + tipo_de_tipo + "/" + objeto,
                        "rb") as f1:
                    come_xuchu = pickle.load(f1) #Abre Representa;áo

                if (not isinstance(come_xuchu, np.ndarray)):
                    come_xuchu = np.array(come_xuchu.todense(), dtype=np.float64)
                lsa = False

                # print("TSNING...")
                # tsne = TSNE(n_components=3)
                # transform_come_xuchu = tsne.fit_transform(come_xuchu)
                # pickle.dump(transform_come_xuchu,open("../../../Objetos/ObjetosProcessados/"+escolha_da_representacao +"/"+tipo_de_tamanho +"/"+ tipo_de_tipo + "/"+ "bbc_" + escolha_da_representacao + "_" + tipo_de_tamanho + "_" + tipo_de_tipo + "_LSA" + str(lsa) +".tsne","wb"))

                if ("LSA" in objeto):
                    lsa = True

                print("Rodando...", escolha_da_representacao, tipo_de_tamanho, tipo_de_tipo, lsa, ":D")                

                for algoritmo in algoritmos_do_kmenzao:

                    for distancia in distancias_do_kmenzao:
                        
                        print("algoritmando", algoritmo, distancia, numero_de_cluster,"kmeans normal")
                        # Dicionario q guardará todas as infos desse objeto
                        come_xuchu_dict = {}
                        come_xuchu_dict["corpus"] = "reuters"
                        come_xuchu_dict["representacao"] = escolha_da_representacao
                        come_xuchu_dict["tamanho"] = tipo_de_tamanho
                        come_xuchu_dict["processamento"] = tipo_de_tipo
                        come_xuchu_dict["LSA"] = lsa
                        come_xuchu_dict["ncluster"] = numero_de_cluster
                        come_xuchu_dict["algoritmo"] = algoritmo
                        come_xuchu_dict["distancia"] = distancia
                        come_xuchu_dict["inicializacao"]="x"
                        # Até aqui o objeto está carregado na memoria - OK
                        comeu_chuxu = "reuters_" + escolha_da_representacao + "_" + tipo_de_tamanho + "_" + tipo_de_tipo + "_LSA" + str(
                            lsa) + "_" + algoritmo + "_" + distancia + "_" + str(numero_de_cluster)
                        
                        best_score = -1000
                        silhueta_acumulador = 0

                        print("(~‾_‾)~  que shit..")

                        for cont in range(30):

                        	flagg=True
                        	while(flagg):
                        		try:
		                            kmeans = ultra_omega_alpha_kmeans.ultra_omega_alpha_kmeans(no_clusters=numero_de_cluster,distancia=distancia,algoritmo=algoritmo)
		                            
		                            kmeans.incluir(come_xuchu)
		                            kmeans.inicializar()

		                            kmeans.executar_x_means()
		                            #print(len(kmeans.clusters[0]),len(kmeans.clusters[1]))
		                            #print(kmeans.no_clusters)

		                            silhueta_final = silhuetaDInamico.calcularSilhueta(kmeans)

		                            silhueta_acumulador = silhueta_acumulador + silhueta_final
		                            flagg=False
		                       	except:
		                       		print("Error found, but remedied")
                        silhueta_acumulador = silhueta_acumulador/30

                        resposta.append((silhueta_acumulador, come_xuchu_dict))




    pickle.dump(resposta,open(escolha_da_representacao + str(numero_de_cluster) + "x_reuters.jojo", "wb"))                    




