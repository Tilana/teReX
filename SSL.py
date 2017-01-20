import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def SSL():
	# PARAMETERS	
	RBF = 1
	gamma = 20 
	conversion = 'raw_tfidf' 	# one of None, 'tfidf', 'MM', 'raw_tfidf'
	cosSim = 1
	Laplace = 0

	nrLabeledData = 230 

	# Load Data	
	filename = 'processedDocuments/Newsgroup_guns_motorcycles_all.pkl'
	data = pd.read_pickle(filename)
	X = np.load('NormMatrix_all.npy')
	nrDocs = len(data)

	if conversion=='tfidf':
		DF = np.array(data.tfIdf.tolist())
		X[0:nrDocs, nrDocs+1:] = DF
		X[nrDocs+1:, 0:nrDocs] = np.transpose(DF)

	if conversion=='raw_tfidf':
		DF = np.array(data.tfIdf.tolist())
		X = DF

	if conversion=='MM':
		DD = X[0:nrDocs, 0:nrDocs]
		DF = X[0:nrDocs, nrDocs:]
		FF = X[nrDocs:, nrDocs:]
		X = np.dot(np.dot(DF, FF), np.transpose(DF))

	if cosSim:
		X = cosine_similarity(X,X)
	
	if Laplace:	
		X[np.diag_indices_from(X)] = 0
		rowsum = X.sum(axis=1)
		D = np.diag(rowsum)
		L = D-X
		X = L
	                                                                            
	
	print 'Label Propagation'
	if RBF:
		labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=200) 
	else:
		labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=200)
	
	print labelPropagation
	labels = np.ones([X.shape[0]])*-1
	trueLabelIndex = range(0,nrLabeledData)
	labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()
	
	labelPropagation.fit(X, labels)
	predictLabels = labelPropagation.transduction_
	
	print 'Total Accuracy: %f' % accuracy_score(data.category.tolist(), predictLabels.tolist()[0:len(data)])
	print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])


if __name__ =='__main__':
	SSL()
