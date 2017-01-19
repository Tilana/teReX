import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

#combinedMatrix = np.load('NormMatrix600samples.npy')
filename = 'processedDocuments/Newsgroup_guns_motorcycles_100.pkl'
data = pd.read_pickle(filename)
combinedMatrix = np.load('NormMatrix_100_tfidf.npy')
nrLabeledData = 20

nrDocs = len(data)
X = combinedMatrix

DF = np.array(data.tfIdf_docFeature.tolist())
X[0:nrDocs, nrDocs+1:] = DF
X[nrDocs+1:, 0:nrDocs] = np.transpose(DF)


#DD = X[0:nrDocs, 0:nrDocs]
#DF = X[0:nrDocs, nrDocs:]
#FF = X[nrDocs:, nrDocs:]

#X = np.dot(np.dot(DF, FF), np.transpose(DF))
#X = np.array(data.tfIdf.tolist())
#X = cosine_similarity(X,X)


# Remove diagonal elements
#X[np.diag_indices_from(X)] = 0
#
## Degree Matrix
#rowsum = X.sum(axis=1)
#D = np.diag(rowsum)
#L = D-X
#X = L
                                                                            
#sparseMatrix = sparse.csr_matrix(combinedMatrix)
print 'Label Propagation'
labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1 max_iter=200) 
labelPropagation = LabelPropagation('rbf', gamma=50, alpha=1, useInputMatrix=0, max_iter=200) 
print labelPropagation
labels = np.ones([X.shape[0]])*-1
trueLabelIndex = range(0,nrLabeledData)
labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()

#Tests
classes = np.unique(labels)
                                                                            
labelPropagation.fit(X, labels)
predictLabels = labelPropagation.transduction_

print 'True Labels: '
print labels[0:20]
print 'Preditcted Labels: '
print predictLabels[0:20]
print 'Total Accuracy: %f' % accuracy_score(data.category.tolist(), predictLabels.tolist()[0:len(data)])
print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData:], predictLabels.tolist()[nrLabeledData:len(data)])
