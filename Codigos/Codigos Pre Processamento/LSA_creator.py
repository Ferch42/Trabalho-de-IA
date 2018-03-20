import os,pickle 
from sklearn.decomposition import TruncatedSVD
folders = os.listdir()

for folder in folders:
	files = os.listdir('./'+folder)
	for file in files:
		print(file)
		data=pickle.load(open('./'+folder+'/'+file,'rb'))
		lsa=TruncatedSVD(n_components=100)
		new_data= lsa.fit_transform(data)
		out_file= './'+folder+'/'+file.replace('.','LSA.')
		pickle.dump(new_data,open(out_file,'wb'))

