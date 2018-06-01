import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt



def get_mean(dados):
    mean_point = dados.mean(axis=0)
    return mean_point

def get_distancias(dados):
    distances = np.zeros(dados.shape[0])
    mean_point = get_mean(dados)
    for i in range(len(dados)):
        distances[i] = np.sqrt(((dados[i] - mean_point) ** 2).sum())
    return distances

def get_indices_para_rem(distances):
    portion = int(0.013*len(distances))
    sorted_distances = sorted(distances, reverse=True)[:portion]
    to_be_removed = pd.Series(distances)[pd.Series(distances).isin(sorted_distances)]
    indexes_to_be_removed = [ind for ind in to_be_removed.index]
    return indexes_to_be_removed




def plot_Inicial(distances):
    X = [i for i in range(len(distances))]
    Y = distances
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=X, y=Y)
    # plt.show()

def plot_apos_remocao(distances, indexes_to_be_removed):
    X = [i for i in range(len(distances)) if i not in indexes_to_be_removed]
    Y = [distances[i] for i in range(len(distances)) if i not in indexes_to_be_removed]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=X, y=Y)
    # plt.show()

def return_new_ndarray(dados):
    distances = np.zeros(dados.shape[0])
    distances = get_distancias(distances, dados)
    ind_removal = get_indices_para_rem(distances)
    ind_manter = [i for i in range(len(dados)) if i not in ind_removal]
    new_array = np.array([dados[i] for i in range(len(dados)) if i not in ind_removal])
    return new_array, ind_manter

# plot1()
if __name__ == '__main__':
    come_xuchu_binario = pickle.load(open("../../Objetos/ObjetosPreProcessados/Binario/3K/Normal/binaryVector3k.aug", "rb"))
    come_xuchu_binario = np.array(come_xuchu_binario.todense(), dtype=np.float64)
    come_xuch_tf = pickle.load(open("../../Objetos/ObjetosPreProcessados/TF/3K/Normal/tfVector3k.aug", "rb"))
    come_xuch_tf = np.array(come_xuch_tf.todense(), dtype=np.float64)
    come_xuch_tfidf = pickle.load(open("../../Objetos/ObjetosPreProcessados/TFIDF/3K/Normal/tfidfVector3k.aug", "rb"))
    come_xuch_tfidf = np.array(come_xuch_tfidf.todense(), dtype=np.float64)
    come_xuch_w2v = pickle.load(open("../../Objetos/ObjetosPreProcessados/Word2Vec/Word2Vec.aug", "rb"))
    come_xuch_w2v = np.array(come_xuch_w2v, dtype=np.float64)
    # dist_w2v = get_distancias(come_xuch_w2v)
    # rem_ind_w2v = get_indices_para_rem(dist_w2v)
    # plot_Inicial(dist_w2v)
    # plot_apos_remocao(dist_w2v, rem_ind_w2v)
    dist_bin = get_distancias(come_xuchu_binario)
    rem_bin = get_indices_para_rem(dist_bin)
    plot_Inicial(dist_bin)
    plot_apos_remocao(dist_bin, rem_bin)
    plt.show()
