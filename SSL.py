import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from plotFunctions import surface
from helper import createConfigName, generateVocabulary
from itertools import product

def SSL():
	# PARAMETERS
	RBF = 1
	gammaArray = [0.5, 1, 5, 10, 20, 50, 100]
	conversion = 'tfidf'	# one of None, 'tfidf', 'MM', 'raw_tfidf'
	nrLabeledData = 250
	
	# Load Data
	name = 'NG_guns_motorcycles'
	filename = 'processedDocuments/' + name + '.pkl'
	resultFilename = 'results/' + name + '_' + str(conversion) + '_'+ str(nrLabeledData) + '.txt'

	results = pd.DataFrame(data = {'gamma': gammaArray})

	for params in product((0,1), repeat=3):
		cosSim = params[0]
		avgFF = params[1]
		useContextSim = params[2]

		configName = createConfigName(cosSim, avgFF, useContextSim)
		print configName
	
		data = pd.read_pickle(filename)
		vocabulary = generateVocabulary(data.sentences.tolist())
		X = np.load('matrices/' + name + '.npy')
		X = X[:-2,:-2]
		contextSim = np.load('matrices/' + name + '_contextSim.npy')
		nrDocs = len(data)

		labels = np.ones([X.shape[0]])*-1
		trueLabelIndex = range(0,nrLabeledData)
		labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()
		
		# remove DD and FD
		#X = X[:,nrDocs:]
		# remove $Start$ and $End$

		if useContextSim:
			X[nrDocs:, nrDocs:] = contextSim

		if avgFF:
			FF = np.add(X[nrDocs:, nrDocs:], contextSim)/2
			X[nrDocs:, nrDocs:] = FF 

		# Renormalize
		#if renormalize:
		#	DF = X[nrDocs:, :nrDocs]
		#	rowsums = DF.sum(axis=1) 
		#	for i in range(len(rowsums)):
		#		DF[i] = DF[i]/rowsums[i]
		#	X[nrDocs:,:nrDocs] = DF
		#	X[:nrDocs,nrDocs:] = np.transpose(DF)
		#FF = X[nrDocs:,:]
		#X[nrDocs:,:] = np.transpose(FF)
		
		# Remove posts with no features
		#DF = X[:nrDocs,:]
		#indZeroFeatures = np.where(DF.sum(axis=1)==0)[0]
		#for ind in indZeroFeatures:
		#	X = np.delete(X,ind,0)
		#data.drop(data.index[indZeroFeatures], inplace=True)
		#data.index = range(len(data)) 
		#nrDocs = len(data)
		
		# Normalize
		#DF = X[:nrDocs,:] 
		#FF = X[nrDocs:,:]
		#rowsum = DF.sum(axis=1)
		#X[nrDocs:, nrDocs:] = np.transpose(X[nrDocs:, nrDocs:])
		#X[-1,-1] = 1
		#FF = X[nrDocs:, nrDocs:]
		#FF_rowsum = FF.sum(axis=1)

		if conversion=='tfidf':
			DF = np.array(data.tfIdf.tolist())
			X[:nrDocs, nrDocs:] = DF
			#X[nrDocs:, :nrDocs] = np.transpose(DF)
			#X = X[:,:]

		if conversion=='raw_tfidf':
			DF = np.array(data.tfIdf.tolist())
			X = DF

		if conversion=='MM':
			DF = X[:nrDocs, nrDocs:]
			FF = X[nrDocs:, nrDocs:]
			X = np.dot(DF,FF)

		if cosSim:
			X = cosine_similarity(X,X)
		
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

			results[configName+'_LP'] = labelProp_accuracy
			results[configName+'_LS'] = labelSpread_accuracy
			
		else:
			labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=200)
			print labelPropagation
			labelPropagation.fit(X, labels)
			predictLabels = labelPropagation.transduction_
			print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])

	results.to_csv(resultFilename, sep='\t', encoding='utf-8')


if __name__ =='__main__':
	SSL()
