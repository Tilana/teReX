import numpy as np
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from scipy import sparse

#combinedMatrix = np.load('NormMatrix600samples.npy')
filename = 'processedDocuments/Newsgroup_guns_motorcycles.pkl'
data = pd.read_pickle(filename)
combinedMatrix = np.load('NormMatrix.npy')
nrLabeledData = 1000
X = combinedMatrix
                                                                            
#sparseMatrix = sparse.csr_matrix(combinedMatrix)
print 'Label Propagation'
labelPropagation = LabelPropagation(alpha=1) 
labels = np.ones([X.shape[0]])*-1
trueLabelIndex = range(0,nrLabeledData)
labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()
                                                                            
labelPropagation.fit(X, labels)
predictLabels = labelPropagation.transduction_
print 'Accuracy: %f' % accuracy_score(data.category.tolist(), predictLabels.tolist()[0:len(data)])
