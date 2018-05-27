import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
# import pprint


def load_dir():
    answers = []
    for f in os.listdir("../../Objetos/Objetos Processados Amostra/Binario/"):
        with open("../../Objetos/Objetos Processados Amostra/Binario/" + f, "rb") as ans:
            answers = answers + pickle.load(ans)

    for f in os.listdir("../../Objetos/Objetos Processados Amostra/TF/"):
        if "8" not in f:
            with open("../../Objetos/Objetos Processados Amostra/TF/" + f, "rb") as ans:
                answers = answers + pickle.load(ans)

    # for f in os.listdir("../../Objetos/Objetos Processados Amostra/TFIDF/"):
    #     if "8" not in f:
    #         with open("../../Objetos/Objetos Processados Amostra/TFIDF/" + f, "rb") as ans:
    #             answers = answers + pickle.load(ans)
    return answers


# def Plot_Samples(samples, order_key, representation=True):
#     if order_key not in samples[0][1].keys():
#         raise ValueError("not a valid key")
#     sorted_sample = samples
#     dif_keys = []
#
#     for s in samples:
#         if s[1][order_key] not in dif_keys:
#             dif_keys.append(s[1][order_key])
#
#     for v in samples[0][1].keys():
#         if not v == order_key:
#             # print(v)
#             sorted_sample = sorted(sorted_sample, key=lambda x: x[1][v])
#     x = []
#     y = []
#     labell = ""
#     conf = set()
#     figz = []
#     acc = []
#     count = 0
#     for i, s in enumerate(sorted_sample):
#         count = count + 1
#         print(count, len(dif_keys))
#         if ((count % len(dif_keys)) == 0):
#             print("entrou")
#             x.append(s[1][order_key])
#             y.append(s[0])
#             print(x)
#             print(y)
#             if (verifica_consistencia(acc, order_key)):
#                 raise ValueError("kasdk")
#
#             plot = sorted([(x[i], y[i]) for i in range(len(x))])
#             x = []
#             y = []
#             acc = []
#
#             fig, ax = plt.subplots()
#             title = ""
#             for t in conf:
#                 title = title + " " + t
#             conf = set()
#             for v in samples[i][1].keys():
#                 if not v == order_key:
#                     conf.add(v + ": " + str(samples[i][1][v]))
#             fig.suptitle(title)
#             ax.set_ylabel("Silhueta")
#             ax.set_xlabel(order_key)
#             x_coordinates = np.arange(len(plot))
#             ax.bar(x_coordinates, [yy[1] for yy in plot], label=labell)
#             ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
#             # ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
#             ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#             figz.append(fig)
#         else:
#
#             x.append(s[1][order_key])
#             y.append(s[0])
#             acc.append(s)
#     return figz
#
#
# def verifica_consistencia(lista, variavel):
#     conf = set()
#     print(lista)
#     for f in lista[0][1].keys():
#         if not f == variavel:
#             conf.add(lista[0][1][f])
#     print(conf)
#     i = 0
#     for l in lista:
#         print(i)
#         # print(l)
#         i = i + 1
#         for f in l[1].keys():
#             if not f == variavel:
#                 # print(l[1][f])
#                 if l[1][f] not in conf:
#                     return True
#     return False


def criar_plots(data_tuples, order_key, representacoes: list = None):
    temp = data_tuples[:]
    # fig_list = []

    if not representacoes:
        representacoes = ['Binario', 'TF', 'TFIDF']

    keys = data_tuples[0][1].keys()
    if order_key not in keys:
        raise ValueError("not a valid key")
    for v in keys:
        if v != order_key and v != 'representacao':
            sorted_answers = sorted(data_tuples, key=lambda x: x[1][v])
            # f = open("order_" + str(order_key) + ".txt", "w")
            # [f.write(str(sorted_ans[1]) + "\n") for sorted_ans in sorted_answers]
            # f.close()

    cur_parameters = dict()
    Xs = [[] for i in representacoes]
    Ys = [[] for i in representacoes]
    gerador_temp = gerador_dados(temp)
    for elem in temp:
        if elem[1]['representacao'] == representacoes[0]:
            first_elem = elem
    for key in first_elem[1].keys():
        if key != order_key and key != 'representacao':
            cur_parameters[key] = first_elem[1][key]
    switch = False

    while len(temp) > 0:
        next_val, index = next(gerador_temp)
        diff = False
        if switch:
            plotar(Xs, Ys, representacoes, cur_parameters, order_key)
            Xs = [[] for i in representacoes]
            Ys = [[] for i in representacoes]
            for key in cur_parameters.keys():
                cur_parameters[key] = next_val[1][key]
            switch = False
        else:
            for key in cur_parameters.keys():
                if cur_parameters[key] != next_val[1][key]:
                    diff = True
        if index == len(temp) - 1:
            switch = True
        if diff:
            continue
        else:
            for i in range(len(representacoes)):
                if next_val[1]['representacao'] == representacoes[i]:
                    Xs[i].append(next_val[1][order_key])
                    Ys[i].append(next_val[0])
            temp.pop(index)


def plotar(Xs, Ys, representacoes, cur_parameters, order_key):
    fig, ax = plt.subplots()
    for i in range(len(Xs)):
        ax.plot(Xs[i], Ys[i], label=representacoes[i], c=np.random.rand(3,), marker=(6, 1, 0))
    ax.set_xlabel(str(order_key))
    ax.set_ylabel('silhueta')
    ax.legend()
    aux = [str(param)+"_" for param in cur_parameters.values()]
    filename = ''
    for f in aux:
        filename += f
    ax.set_title(filename[:-1])
    plt.savefig('Graficos/{}.png'.format(filename[:-1]), dpi=160)


# recebe as referencias de suas listas
def gerador_dados(lista):
    index = 0
    while True:
        if index >= len(lista):
            if len(lista) > 0:
                index = 0
            else:
                break
        next_val = lista[index]
        yield next_val, index
        index += 1


if __name__ == '__main__':
    answers = load_dir()
    # fig = Plot_Samples(answers, 'ncluster')
    criar_plots(answers, 'ncluster', representacoes=['Binario', 'TF']) #  nao precisa especificar representacao tecnicamente

# initially ordered by corpus, representacao, tamanho, processamento, LSA, ncluster, algoritmo, distancia, inicializacao
# answer format sample
# [(0.05473443043032169,
# {'LSA': False,
#  'algoritmo': 'media',
#  'corpus': 'bbc',
#  'distancia': 'euclidiana',
#  'inicializacao': 'padrao',
#  'ncluster': 2,
#  'processamento': 'Normal',
#  'representacao': 'Binario',
#  'tamanho': '3k'}), . . .
