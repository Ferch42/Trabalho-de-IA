from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import matplotlib
import random
import re
import io



class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.
       Uses wordcloud.get_single_color_func
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

class PalavrasMaisFrequentesCluster:
 
  #recebe o nome da pasta de um corpus e devolve o corpus com os textos formando uma lista
  def __get_corpus(nome_corpus):
    path = '../../../'
    corpus = []
    if(nome_corpus == 'bbc'):
      path = path + 'bbc/'
      folders = os.listdir(path) # vai devolver os nomes da pasta
      for folder in folders:
        for file in os.listdir(path+folder):
          f = open(path+folder+'/'+file).read()
          corpus.append(f.lower()) # add os arqvos textos
      return corpus
    elif(nome_corpus == 'bbc_amostra'):
      path = path + 'bbc amostra/'
    elif(nome_corpus == 'reuters'):
      path = path + 'reuters/text/'
    elif(nome_corpus == 'reuters_amostra'):
      path = path + 'reuters amostra/'
    else:
      print("Nome invalido!!!\nOpcoes validas: 'bbc', 'bbc_amostra','reuters', ou 'reuters_amostra'.")
      exit()
    folder = os.listdir(path)
    for file in folder:
      f = open(path + file).read()
      corpus.append(f.lower())
    return corpus


  #recebe uma lista do corpus e um objeto kmeans e devolve uma lista de listas de textos, em que cada lista interna representa um cluster
  def __get_corpus_clusterizado(corpus, kmeans):
    clusters = kmeans.clusters
    corpus_clusterizado = [[] for _ in range(len(clusters))]
    for n_cluster,cluster in enumerate(clusters):
      for element in cluster:
        corpus_clusterizado[n_cluster].append(corpus[element])
      
    return corpus_clusterizado

  """recebe a lista de listas de textos, transforma a lista de textos em lista de palavras 
  e devolve lista de listas de palavras, em que cada lista interna representa um cluster"""
  def __get_palavras_clusterizadas(corpus_clusterizado):
    
    stopWords = [x for x in ENGLISH_STOP_WORDS]
    otherCommonWords = ['make','year','years','new','people','said','say','time','brown','good','told','000','says','took','way','think','going','just','don','did','use','best','didn']
    for w in otherCommonWords:
      stopWords.append(w)
    palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
    for n_cluster,cluster in enumerate(corpus_clusterizado):
      string = ""
      for texto in cluster:
        string = string + " " + texto
      palavras = re.sub('[^ a-zA-Z0-9]', ' ', string)
    
      palavras = palavras.split(" ")
        
      for palavra in palavras:
        word = palavra.replace("\\s+","")
        
        if(word not in stopWords and not len(word) <= 2 and not word.isdigit()):
          palavras_clusterizadas[n_cluster].append(word)
      
    return palavras_clusterizadas

  #recebe uma lista simples de palavras e devolve uma lista de tuplas(palavra, frequencia)
  def __get_tupla_frequencia_palavras(lista_palavras): 
    lista = sorted(lista_palavras)
    lista_tuplas = []
    contador = 1
    atual = lista[0]
    for palavra in lista:
      if(atual == palavra):
        contador+=1
        continue
      lista_tuplas.append((atual, contador))
      atual = palavra
      contador = 1
    lista_tuplas.append((atual, contador))

    return sorted(lista_tuplas, key = lambda tupla: tupla[1], reverse = True)

  #recebe um numero n e lista de listas de palavras e retorna lista de listas das n palavras mais frequentes de cada cluster
  def __get_n_palavras_mais_frequentes_cluster(n, lista_palavras_clusterizadas):
    lista_mais_frequentes = [[] for _ in range(len(lista_palavras_clusterizadas))]
    for n_cluster,cluster in enumerate(lista_palavras_clusterizadas):
      tuplas = PalavrasMaisFrequentesCluster.__get_tupla_frequencia_palavras(cluster)
      cont = 0
      for contador,tupla in enumerate(tuplas):
        if(contador == n):
          break
        lista_mais_frequentes[n_cluster].append(tupla[0])
    return lista_mais_frequentes

  #método principal que recebe numero n e caminho path e devolve lista de listas das palavras ,ais frequentes por cluster
  def gerar_n_palavras_mais_frequentes_por_cluster(n, nome_corpus, kmeans):
    corpus = PalavrasMaisFrequentesCluster.__get_corpus(nome_corpus)
    corpus_clusterizado = PalavrasMaisFrequentesCluster.__get_corpus_clusterizado(corpus,kmeans)
    palavras_clusterizadas = PalavrasMaisFrequentesCluster.__get_palavras_clusterizadas(corpus_clusterizado)
    return PalavrasMaisFrequentesCluster.__get_n_palavras_mais_frequentes_cluster(n, palavras_clusterizadas)

def get_lista_cores():
  colors = ['midnightblue','salmon','red','green','purple','yellow','cyan']
  '''for name in matplotlib.colors.cnames.items():
    colors.append(name[0])'''
  return colors

def colorir_palavras_dos_clusters(lista_palavras_mais_frequentes_clusterizadas, lista_cores):
  colors = lista_cores
  color_to_words = {}
  for palavras in lista_palavras_mais_frequentes_clusterizadas:
    cor = random.choice(colors)
    colors.remove(cor)
    color_to_words[cor] = palavras
  
  return color_to_words

def colorir_palavras_de_um_cluster(lista_palavras_mais_frequentes_do_cluster, cor):
  color_to_words = {} 
  color_to_words[cor] = lista_palavras_mais_frequentes_do_cluster
  return color_to_words

def converter_listas_em_texto(lista_palavras_mais_frequentes_clusterizadas):
  text = ""
  for palavras in lista_palavras_mais_frequentes_clusterizadas:
    for palavra in palavras:
      text = text + " " + palavra
  return text

def converter_lista_simples_em_texto(lista_palavras):
  text = ""
  for palavra in lista_palavras:
    text = text + " " + palavra
  return text

def plotar_word_cloud(lista_palavras_mais_frequentes_clusterizadas):
  default_color = 'grey'
  colors = get_lista_cores()
  word_to_color = colorir_palavras_dos_clusters(lista_palavras_mais_frequentes_clusterizadas,colors)
  print(word_to_color)
  grouped_color_func = GroupedColorFunc(word_to_color, default_color)
  text = converter_listas_em_texto(lista_palavras_mais_frequentes_clusterizadas)
  wc = WordCloud(width=2000,height=1800, background_color='white', min_font_size=5).generate(text)

  wc.recolor(color_func=grouped_color_func)

  plt.figure()
  plt.imshow(wc, interpolation="bilinear")
  plt.axis("off")
  plt.show()

def save_plot(lista_palavras_mais_frequentes_clusterizadas, nome_do_arquivo):
  default_color = 'grey'
  colors = get_lista_cores()
  word_to_color = colorir_palavras_dos_clusters(lista_palavras_mais_frequentes_clusterizadas,colors)
  #print(word_to_color)
  grouped_color_func = GroupedColorFunc(word_to_color, default_color)
  text = converter_listas_em_texto(lista_palavras_mais_frequentes_clusterizadas)
  wc = WordCloud(width=2000,height=1800, background_color='white', min_font_size=5).generate(text)

  wc.recolor(color_func=grouped_color_func)
  plt.figure()
  plt.imshow(wc, interpolation="bilinear")
  plt.axis("off")
  wc.to_file(nome_do_arquivo + ".png")

def save_plots_por_cluster(lista_palavras_mais_frequentes_clusterizadas, nome_do_arquivo):
  default_color = 'grey'
  colors = get_lista_cores()
  for i,lista in enumerate(lista_palavras_mais_frequentes_clusterizadas):
    cor = random.choice(colors)
    colors.remove(cor)
    color_to_words = colorir_palavras_de_um_cluster(lista, cor)
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)
    text = converter_lista_simples_em_texto(lista)
    wc = WordCloud(width=2000,height=1800, background_color='white', min_font_size=5).generate(text)

    wc.recolor(color_func=grouped_color_func)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    wc.to_file(nome_do_arquivo + "_cluster_" + str(i+1) + ".png")



  
'''
colors = []
for name in matplotlib.colors.cnames.items():
  colors.append(name[0])
color_to_words = {}

n=3
path='../../bbc amostra/'
#lista = PalavrasMaisFrequentesCluster.gerar_n_palavras_mais_frequentes_por_cluster(n, path)
for i in range(n):
  cor = random.choice(colors)
  colors.remove(cor)
  color_to_words[cor] = lista[i]

print(color_to_words)
  
text = ""
for palavras in lista:
  for palavra in palavras:
    text = text + " " + palavra

default_color = 'pink'
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

wc = WordCloud(collocations=False).generate(text)

wc.recolor(color_func=grouped_color_func)

plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

'''







