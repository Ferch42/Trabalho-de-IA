import pandas as pd
import sys
import pickle
import scipy


if __name__ == '__main__':
    arq = sys.argv[1]
    dados = open(arq, 'rb')
    dados = pickle.load(dados)
    try:
        dados = dados.todense()
    except:
        dados = open(arq, 'rb')
        dados = pickle.load(dados)
    rows = map(lambda x: "doc"+str(x), [*range(len(dados))])
    cols = [*range(dados.shape[1])]
    dados = pd.DataFrame(dados, index=rows, columns=cols)
    # arquivo salvo no mesmo local do endereco dado como argumento
    dados.to_csv(arq[:-3] + "csv")