import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tokenizer import tokenizer, tokenizer_lemmatizer
reuters_path='../../reuters/text/'

arquivos= os.listdir(reuters_path)

corpus=[]

for f in arquivos:
	corpus.append(open(reuters_path+f).read())

corpus=[c.lower() for c in corpus]

representacao=['Binario','TF','TFIDF']
tamanho=['3k', 'Total']
tokenizacao= ['Lemma', 'Normal', 'Tokenizer']

os.mkdir('Objetos Preprocessados Reuters')
os.chdir('Objetos Preprocessados Reuters')

for r in representacao:
	os.mkdir(r)
	os.chdir(r)
	for t in tamanho:
		os.mkdir(t)
		os.chdir(t)
		for to in tokenizacao:
			print(r+t+to)
			os.mkdir(to)
			os.chdir(to)
			if(r=='Binario'):
				if(t=='3k'):
					if(to=='Lemma'):
						print('1')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, max_features=3000, binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('2')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english', max_features=3000, binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('3')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer, max_features=3000, binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
				if(t=='Total'):
					if(to=='Lemma'):
						print('4')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('5')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english', binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('6')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer, binary = True)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))

			if(r=='TF'):
				if(t=='3k'):
					if(to=='Lemma'):
						print('7')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('8')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english', max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('9')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer, max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
				if(t=='Total'):
					if(to=='Lemma'):
						print('10')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('11')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english')
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('12')
						vec= CountVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))

			if(r=='TFIDF'):
				if(t=='3k'):
					if(to=='Lemma'):
						print('13')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer, max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('14')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english', max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('15')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer, max_features=3000)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
				if(t=='Total'):
					if(to=='Lemma'):
						print('16')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english',tokenizer = tokenizer_lemmatizer)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Normal'):
						print('17')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english')
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
					if(to=='Tokenizer'):
						print('18')
						vec= TfidfVectorizer(analyzer = "word", stop_words = 'english',tokenizer =tokenizer)
						vecc=vec.fit_transform(c for c in corpus)
						pickle.dump(vecc,open('arq.aug','wb'))
						lsa=TruncatedSVD(n_components=100)
						new_data= lsa.fit_transform(vecc)
						pickle.dump(new_data,open('arq_LSA.aug','wb'))
			os.chdir('..')
		os.chdir('..')
	os.chdir('..')
