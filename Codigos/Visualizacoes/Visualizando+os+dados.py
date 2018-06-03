
# coding: utf-8

# <h1>Visualizando os arquivos</h1>

# In[1]:


import os,pickle
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


os.listdir("..\..\Objetos\ObjetosPreProcessados\Binario\\3K\\Normal")


# <h1>Binary
# </h1>

# In[3]:


binaryVector3k = pickle.load(open('..\..\Objetos\ObjetosPreProcessados\Binario\\3K\\Normal\\binaryVector3k.aug','rb'))


# In[ ]:


binaryVector3k


# In[ ]:


tsne=TSNE(n_components=3)
transformedBinaryVector= tsne.fit_transform(binaryVector3k.todense())


# In[ ]:


transformedBinaryVector


# In[ ]:


xvalues=[vector[0] for vector in transformedBinaryVector]
yvalues=[vector[1] for vector in transformedBinaryVector]
zvalues=[vector[2] for vector in transformedBinaryVector]
    


# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='r',marker='o')
plt.show()


# # Term Frequency#

# In[ ]:


TermFrequencyVector3k = pickle.load(open('..\..\Objetos\ObjetosPreProcessados\TF\\3K\\Normal\\tfVector3k.aug','rb'))


# In[ ]:


TermFrequencyVector3k


# In[ ]:


tsne=TSNE(n_components=3)
transformedTermFrequency= tsne.fit_transform(TermFrequencyVector3k.todense())


# In[ ]:


transformedTermFrequency


# In[ ]:


xvalues=[vector[0] for vector in transformedTermFrequency]
yvalues=[vector[1] for vector in transformedTermFrequency]
zvalues=[vector[2] for vector in transformedTermFrequency]


# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='b',marker='o')
plt.show()


# # Term Frequency Inverse Document Frequency #

# In[ ]:


TFIDFVector3k = pickle.load(open('..\..\Objetos\ObjetosPreProcessados\TFIDF\\3K\\Normal\\tfidfVector3k.aug','rb'))


# In[ ]:


TFIDFVector3k


# In[ ]:


tsne=TSNE(n_components=3)
transformedTFIDF= tsne.fit_transform(TFIDFVector3k.todense())


# In[ ]:


transformedTFIDF


# In[ ]:


xvalues=[vector[0] for vector in transformedTFIDF]
yvalues=[vector[1] for vector in transformedTFIDF]
zvalues=[vector[2] for vector in transformedTFIDF]


# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='g',marker='o')
plt.show()


# # Word2Vec #

# In[ ]:


Word2Vec = pickle.load(open('..\..\Objetos\ObjetosPreProcessados\Word2Vec\Word2Vec.aug','rb'))


# In[ ]:


Word2Vec = np.array(Word2Vec)


# In[ ]:


tsne=TSNE(n_components=3)
transformedWord2Vec= tsne.fit_transform(Word2Vec)


# In[ ]:


xvalues=[vector[0] for vector in transformedWord2Vec]
yvalues=[vector[1] for vector in transformedWord2Vec]
zvalues=[vector[2] for vector in transformedWord2Vec]


# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='y',marker='o')
plt.show()

