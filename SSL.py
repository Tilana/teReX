import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from plotFunctions import surface
from helper import createFilename

def SSL():
	# PARAMETERS	
	RBF = 1
	gammaArray = [0.5, 1, 5, 10, 20, 50, 100]
	conversion = 'tfidf' # one of None, 'tfidf', 'MM', 'raw_tfidf'
	cosSim = 0
	Laplace = 0

	nrLabeledData = 230 

	# Load Data	
	filename = 'processedDocuments/Newsgroup_guns_motorcycles_all.pkl'
	resultFilename = createFilename(filename,RBF,conversion,cosSim, Laplace)
	
	data = pd.read_pickle(filename)
	X = np.load('NormMatrix_all.npy')
	nrDocs = len(data)
	
	# remove DD and FD
	X = X[:,nrDocs:]
	FF = X[nrDocs:,:]
	X[nrDocs:,:] = np.transpose(FF)
	
	# Remove posts with no features
	DF = X[:nrDocs,:]
	indZeroFeatures = np.where(DF.sum(axis=1)==0)[0]
	for ind in indZeroFeatures:
		X = np.delete(X,ind,0)
	data.drop(data.index[indZeroFeatures], inplace=True)
	data.index = range(len(data)) 
	nrDocs = len(data)
	
	# Normalize
	#DF = X[:nrDocs,:] 
	#FF = X[nrDocs:,:]
	#rowsum = DF.sum(axis=1)
	#X[nrDocs:, nrDocs:] = np.transpose(X[nrDocs:, nrDocs:])
	#X[-1,-1] = 1
	#FF = X[nrDocs:, nrDocs:]
	#FF_rowsum = FF.sum(axis=1)

	if conversion=='tfidf':
		DF = np.array(data.tf.tolist())
		X[:nrDocs, :-1] = DF
		X = X[:-1,:-1]

	if conversion=='raw_tfidf':
		DF = np.array(data.tfIdf.tolist())
		X = DF

	if conversion=='MM':
		DF = X[:nrDocs, :]
		FF = X[nrDocs:, :]
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
		labelProp_accuracy = []
		labelSpread_accuracy = []
		for gamma in gammaArray:
			print 'Gamma: %f' % gamma
			labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=100)
			labelPropagation.fit(X, labels)
			predictLabels = labelPropagation.transduction_
			curr_acc = accuracy_score(data.category.tolist()[nrLabeledData:], predictLabels.tolist()[nrLabeledData:len(data)])
			labelProp_accuracy.append(curr_acc)
			print 'Label Prop. Test Accuracy: %f' % curr_acc
			labelSpread = LabelSpreading('rbf', gamma=gamma, alpha=1)
			#Test
			labelSpread.fit(X,labels)
			predictLabels = labelSpread.transduction_
			curr_acc = accuracy_score(data.category.tolist()[nrLabeledData:], predictLabels.tolist()[nrLabeledData:len(data)])
			labelSpread_accuracy.append(curr_acc)
			print 'Label Spread. Test Accuracy: %f' % curr_acc
		results = pd.DataFrame({'gamma': gammaArray, 'LabelPropagation': labelProp_accuracy, 'LabelSpread': labelSpread_accuracy})
		results.to_csv(resultFilename, sep='\t', encoding='utf-8')
	else:
		labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=200)
		print labelPropagation
		labelPropagation.fit(X, labels)
		predictLabels = labelPropagation.transduction_
		print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])


if __name__ =='__main__':
	SSL()
