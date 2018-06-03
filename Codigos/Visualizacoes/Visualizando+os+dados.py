
# coding: utf-8

# <h1>Visualizando os arquivos</h1>

# In[33]:


import os,pickle
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# <h1>Binary
# </h1>

# In[2]:


binaryVector3k = pickle.load(open('..\\ObjetosPreProcessados\\binaryVector3k.aug','rb'))


# In[3]:


binaryVector3k


# In[4]:


tsne=TSNE(n_components=3)
transformedBinaryVector= tsne.fit_transform(binaryVector3k.todense())


# In[5]:


transformedBinaryVector


# In[6]:


xvalues=[vector[0] for vector in transformedBinaryVector]
yvalues=[vector[1] for vector in transformedBinaryVector]
zvalues=[vector[2] for vector in transformedBinaryVector]
    


# In[7]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='r',marker='o')
plt.show()


# # Term Frequency#

# In[8]:


TermFrequencyVector3k = pickle.load(open('..\\ObjetosPreProcessados\\tfVector3k.aug','rb'))


# In[9]:


TermFrequencyVector3k


# In[19]:


tsne=TSNE(n_components=3)
transformedTermFrequency= tsne.fit_transform(TermFrequencyVector3k.todense())


# In[20]:


transformedTermFrequency


# In[22]:


xvalues=[vector[0] for vector in transformedTermFrequency]
yvalues=[vector[1] for vector in transformedTermFrequency]
zvalues=[vector[2] for vector in transformedTermFrequency]


# In[15]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='b',marker='o')
plt.show()


# # Term Frequency Inverse Document Frequency #

# In[16]:


TFIDFVector3k = pickle.load(open('..\\ObjetosPreProcessados\\tfidfVector3k.aug','rb'))


# In[17]:


TFIDFVector3k


# In[18]:


tsne=TSNE(n_components=3)
transformedTFIDF= tsne.fit_transform(TFIDFVector3k.todense())


# In[25]:


transformedTFIDF


# In[26]:


xvalues=[vector[0] for vector in transformedTFIDF]
yvalues=[vector[1] for vector in transformedTFIDF]
zvalues=[vector[2] for vector in transformedTFIDF]


# In[27]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='g',marker='o')
plt.show()


# # Word2Vec #

# In[56]:


Word2Vec = pickle.load(open('..\\ObjetosPreProcessados\\Word2Vec.aug','rb'))


# In[57]:


Word2Vec = np.array(Word2Vec)


# In[69]:


tsne=TSNE(n_components=3)
transformedWord2Vec= tsne.fit_transform(Word2Vec)


# In[70]:


xvalues=[vector[0] for vector in transformedWord2Vec]
yvalues=[vector[1] for vector in transformedWord2Vec]
zvalues=[vector[2] for vector in transformedWord2Vec]


# In[71]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xvalues,yvalues,zvalues,c='y',marker='o')
plt.show()

