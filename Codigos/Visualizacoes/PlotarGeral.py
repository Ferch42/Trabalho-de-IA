import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pprint


def load_dir():
    answers = []
    # for f in os.listdir("../../Objetos/ObjetosProcessadosReutersAmostra/Binario/"):
    #     with open("../../Objetos/ObjetosProcessadosReutersAmostra/Binario/" + f, "rb") as ans:
    #         answers = answers + pickle.load(ans)
    #
    # for f in os.listdir("../../Objetos/ObjetosProcessadosReutersAmostra/TF/"):
    #     if "8" not in f:
    #         with open("../../Objetos/ObjetosProcessadosReutersAmostra/TF/" + f, "rb") as ans:
    #             answers = answers + pickle.load(ans)
    #
    # for f in os.listdir("../../Objetos/ObjetosProcessadosReutersAmostra/TFIDF/"):
    #     if "8" not in f:
    #         with open("../../Objetos/ObjetosProcessadosReutersAmostra/TFIDF/" + f, "rb") as ans:
    #             answers = answers + pickle.load(ans)
    for f in os.listdir("../../Objetos/ObjetosProcessadosReutersAmostraSOM/REUTERS/"):
        if "8" not in f:
            with open("../../Objetos/ObjetosProcessadosReutersAmostraSOM/REUTERS/" + f, "rb") as ans:
                answers = answers + pickle.load(ans)

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


def criar_plots(data_tuples, order_key, representacoes=None):
    # fig_list = []

    if not representacoes:
        representacoes = ['Binario', 'TF', 'TFIDF', 'Word2Vec']

    keys = data_tuples[0][1].keys()
    if order_key not in keys:
        raise ValueError("not a valid key")
    # for v in keys:
    #     if v != order_key and v != 'representacao':
    #         sorted_answers = sorted(data_tuples, key=lambda x: x[1][v])
    #         # f = open("order_" + str(order_key) + ".txt", "w")
    #         # [f.write(str(sorted_ans[1]) + "\n") for sorted_ans in sorted_answers]
    #         # f.close()
    # sorted_answers = sorted(sorted_answers, key=lambda  x:x[1]['representacao'])
    temp = data_tuples[:]
    cur_parameters = dict()
    index = [0]
    Xs = [[] for i in representacoes]
    Ys = [[] for i in representacoes]
    gerador_temp = gerador_dados(temp, index)
    first_elem = temp[0]
    for key in first_elem[1].keys():
        if (key != order_key) and (key != 'representacao'):
            cur_parameters[key] = first_elem[1][key]
    switch = False
    while len(temp) > 0:
        next_val = next(gerador_temp)
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
                cur_p_v = cur_parameters[key]
                next_v_v = next_val[1][key]
                if cur_p_v != next_v_v:
                    diff = True
        if index[0] == (len(temp) - 1):
            switch = True
        if diff:
            continue
        else:
            for i in range(len(representacoes)):
                if next_val[1]['representacao'] == representacoes[i]:
                    Xs[i].append(next_val[1][order_key])
                    Ys[i].append(next_val[0])
            temp.pop(index[0])
            index[0] -= 1



def plotar(Xs, Ys, representacoes, cur_parameters, order_key):
    fig, ax = plt.subplots()
    for i in range(len(Xs)):
        inp = np.array(sorted(zip(Xs[i], Ys[i])))
        if len(inp) == 0:
            continue
        # ax.plot(Xs[i], Ys[i], label=representacoes[i], c=np.random.rand(3,), marker=(6, 1, 0))
        ax.plot(inp[:, 0], inp[:, 1], label=representacoes[i], c=np.random.rand(3, ), marker=(6, 1, 0))
    ax.set_xlabel(str(order_key))
    ax.set_ylabel('silhueta')
    ax.legend()
    aux = [str(param)+"_" for param in cur_parameters.values()]
    filename = ''
    for f in aux:
        filename += f
        ax.set_title(filename[:-1])
    plt.savefig('Graficos/{}.png'.format(filename[:-1]), dpi=160)
    plt.close(fig)


def writeAns(ans, name):
    f = open('order{}.txt'.format(name), 'w')
    # ans = sorted(ans, key= lambda x: x[1]['ncluster'])
    for tupl in ans:
        f.write(str(tupl[0]) + " | ")
        for value in tupl[1].values():
            f.write(str(value) + " | ")
        f.write("\n")

# recebe as referencias de suas listas
def gerador_dados(lista, index):
    index[0] = 0
    while True:
        if index[0] >= len(lista):
            if len(lista) > 0:
                index[0] = 0
            else:
                break
        next_val = lista[index[0]]
        yield next_val
        index[0] += 1


class IncrementalDict(dict):
    def __missing__(self, key):
        return 0

def mergeAnsDuplicates(ans):
    real_keys = [*ans[0][1].keys()]
    new_answer = []
    repeated_ans_sums = IncrementalDict()
    repetition_counts = IncrementalDict()
    for i in range(len(ans)):
        tupl = ans[i]
        cur_params = tuple(tupl[1].values())
        repeated_ans_sums[cur_params] += tupl[0]
        repetition_counts[cur_params] += 1
    for param_set in repeated_ans_sums.keys():
        repeated_ans_sums[param_set] /= repetition_counts[param_set]
        aux_dict = dict()
        for i in range(len(real_keys)):
            aux_dict[real_keys[i]] = param_set[i]
        new_answer.append((repeated_ans_sums[param_set], aux_dict))

    return new_answer, repetition_counts


if __name__ == '__main__':
    answers, repetitions = mergeAnsDuplicates(load_dir())
    # fig = Plot_Samples(answers, 'ncluster')
    criar_plots(answers, 'ncluster') #  nao precisa especificar representacao tecnicamente
    # pprint.PrettyPrinter().pprint(answers)
    # writeAns(answers, '_noRep')

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
