from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np


def supervisedBaseline():

	nrLabels = 60 
	filename='processedDocuments/Newsgroup_guns_motorcycles_300.pkl'
	data = pd.read_pickle(filename) 
	n = len(data)
        
	#features = data.CountVectors.tolist()
	features = data.tfIdf.tolist()
	labels = data.category.tolist()

	clf = MultinomialNB(alpha=0.1)
	clf.fit(features[0:nrLabels], labels[0:nrLabels])
	pred = clf.predict(features[nrLabels+1:n])

	accuracy = accuracy_score(labels[nrLabels+1:n], pred)
	print 'Test Accuracy: %f' % accuracy 
	


if __name__ == '__main__':
    supervisedBaseline()
