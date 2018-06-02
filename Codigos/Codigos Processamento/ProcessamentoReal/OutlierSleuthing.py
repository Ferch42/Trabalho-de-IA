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

def get_indices_para_rem(distances, pct=None):
    if not pct:
        pct = 0.03
    portion = int(pct*len(distances))
    sorted_distances = sorted(distances, reverse=True)[:portion]
    to_be_removed = pd.Series(distances)[pd.Series(distances).isin(sorted_distances)]
    indexes_to_be_removed = [ind for ind in to_be_removed.index]
    return indexes_to_be_removed

def return_new_ndarray_indices(dados, pct=None):
    if not pct:
        pct = 0.03
    distances = get_distancias(dados)
    ind_removal = get_indices_para_rem(distances, pct)
    ind_manter = [i for i in range(len(dados)) if i not in ind_removal]
    new_array = np.array([dados[i] for i in range(len(dados)) if i not in ind_removal])
    ind_transform = dict()
    for i in range(len(new_array)):
        ind_transform[i] = ind_manter[i]
    return new_array, ind_transform


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


def plot_dual(distances, indexes_to_be_removed, title):
    X1 = [i for i in range(len(distances))]
    Y1 = distances
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(121)
    ax.set_title("Antes da remocao")
    ax.scatter(x=X1, y=Y1)
    X2 = [i for i in range(len(distances)) if i not in indexes_to_be_removed]
    Y2 = [distances[i] for i in range(len(distances)) if i not in indexes_to_be_removed]
    ax2 = fig.add_subplot(122)
    ax2.set_title("Apos remocao")
    ax2.scatter(x=X2, y=Y2)


# plot1()
if __name__ == '__main__':
    bbc_xuchu_binario = pickle.load(open("../../Objetos/ObjetosPreProcessados/Binario/3K/Normal/binaryVector3k.aug", "rb"))
    bbc_xuchu_binario = np.array(bbc_xuchu_binario.todense(), dtype=np.float64)
    bbc_xuchu_tf = pickle.load(open("../../Objetos/ObjetosPreProcessados/TF/3K/Normal/tfVector3k.aug", "rb"))
    bbc_xuchu_tf = np.array(bbc_xuchu_tf.todense(), dtype=np.float64)
    bbc_xuchu_tfidf = pickle.load(open("../../Objetos/ObjetosPreProcessados/TFIDF/3K/Normal/tfidfVector3k.aug", "rb"))
    bbc_xuchu_tfidf = np.array(bbc_xuchu_tfidf.todense(), dtype=np.float64)
    bbc_xuchu_w2v = pickle.load(open("../../Objetos/ObjetosPreProcessados/Word2Vec/Word2Vec.aug", "rb"))
    bbc_xuchu_w2v = np.array(bbc_xuchu_w2v, dtype=np.float64)

    # dist_w2v = get_distancias(bbc_xuchu_w2v)
    # rem_ind_w2v = get_indices_para_rem(dist_w2v)
    # plot_Inicial(dist_w2v)
    # plot_apos_remocao(dist_w2v, rem_ind_w2v)
    dist_bin = get_distancias(bbc_xuchu_binario)
    rem_bin = get_indices_para_rem(dist_bin, 0.013)
    trans = return_new_ndarray_indices(bbc_xuchu_binario)[1]
    plot_dual(dist_bin, rem_bin, "Binario 3K")
    # plt.show()
    print(trans[2196])
    print([rem_bin])