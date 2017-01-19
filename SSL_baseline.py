import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

def SSL_baseline():
	
	nrLabels = 300
	filename='processedDocuments/Newsgroup_guns_motorcycles.pkl'
	data = pd.read_pickle(filename) 
	#data = data.reindex(np.random.permutation(data.index))
	#features = data.CountVectors.tolist()
	features = data.tfIdf.tolist()
	cosSimDoc = cosine_similarity(features, features)
	features = cosSimDoc
	#featureDF = pd.DataFrame(features)
	#stat = featureDF.describe()
	#maxFeatureVal = stat.loc['max']
	#[occ, threshold] = np.histogram(maxFeatureVal)
	#tooLowIndices = [ind for ind,value in enumerate(maxFeatureVal) if value<threshold[1]]
	#featureDF.drop(featureDF.columns[tooLowIndices], axis=1, inplace=True)
	#features = featureDF.as_matrix()
	labels = np.ones([len(features)])*-1
	labels[0:nrLabels] = data.category.tolist()[0:nrLabels]
	#labelPropagation = LabelPropagation('rbf', gamma=152.5, max_iter=200, useInputMatrix=0)
	labelPropagation = LabelPropagation('rbf', gamma=0.5, max_iter=200, useInputMatrix=1)
	#labelPropagation = LabelPropagation('knn', n_neighbors=15, max_iter=100)
	labelPropagation.fit(np.matrix(features), labels)
	predictLabels = labelPropagation.transduction_
	print 'True Labels: '
	print labels[0:20]
	print 'Preditcted Labels: '
	print predictLabels[0:20]
	print 'Total Accuracy: %f' % accuracy_score(data.category.tolist(), predictLabels.tolist()[0:len(data)])                           	
	print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabels:], predictLabels.tolist()[nrLabels:len(data)])

# Run script
if __name__ == '__main__':
	SSL_baseline()
