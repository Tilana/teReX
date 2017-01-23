import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def SSL():
	# PARAMETERS	
	RBF = 1
	gammaArray = [0.5, 1, 5 , 10, 15, 20, 25, 50, 100, 200]
	conversion = 'tfidf' # one of None, 'tfidf', 'MM', 'raw_tfidf'
	cosSim = 0
	Laplace = 0

	nrLabeledData = 230 

	# Load Data	
	filename = 'processedDocuments/Newsgroup_guns_motorcycles_all.pkl'
	data = pd.read_pickle(filename)
	X = np.load('NormMatrix_all.npy')
	nrDocs = len(data)

	# Normalize
	#X[0:nrDocs, 0:nrDocs] = 0
	#rowsum = X.sum(axis=1)
	#X[nrDocs:, nrDocs:] = np.transpose(X[nrDocs:, nrDocs:])
	#X[-1,-1] = 1
	#FF = X[nrDocs:, nrDocs:]
	#FF_rowsum = FF.sum(axis=1)

	if conversion=='tfidf':
		DF = np.array(data.tf.tolist())
		X[0:nrDocs, nrDocs:-1] = DF
		X[nrDocs:-1, 0:nrDocs] = np.transpose(DF)
		X = X[:-1,:-1]

	if conversion=='raw_tfidf':
		DF = np.array(data.tfIdf.tolist())
		X = DF

	if conversion=='MM':
		#DD = X[0:nrDocs, 0:nrDocs]
		#DF = np.array(data.tfIdf.tolist())
		#X[0:nrDocs, nrDocs:-1] = DF
		DF = X[0:nrDocs, nrDocs:]
		FF = X[nrDocs:, nrDocs:]
		#X = np.dot(np.dot(DF, FF), np.transpose(DF))
		X = np.dot(DF,FF)

	if cosSim:
		X = cosine_similarity(X,X)
	
	if Laplace:	
		X[np.diag_indices_from(X)] = 0
		rowsum = X.sum(axis=1)
		D = np.diag(rowsum)
		L = D-X
		X = L

        labels = np.ones([X.shape[0]])*-1
        trueLabelIndex = range(0,nrLabeledData)
        labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()

	
	print 'Label Propagation'
	if RBF:
		accuracy = [] 
		for gamma in gammaArray:
			labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=200)
			print 'Gamma: %f' % gamma
			labelPropagation.fit(X, labels)
			predictLabels = labelPropagation.transduction_
			curr_acc = accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])
			accuracy.append(curr_acc)
			print 'Label Prop. Test Accuracy: %f' % curr_acc
			labelSpread = LabelSpreading('rbf', gamma=gamma)
			#Test
			labelSpread.fit(X,labels)
			predictLabels = labelSpread.transduction_
			curr_acc = accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])
			print 'Label Spread. Test Accuracy: %f' % curr_acc 
		print gammaArray
		print accuracy
	else:
		labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=200)
		print labelPropagation
		labelPropagation.fit(X, labels)
		predictLabels = labelPropagation.transduction_
		print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])


if __name__ =='__main__':
	SSL()
