from GraphDatabase import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
import pandas as pd 
import numpy as np
from numpy import transpose, identity
from preprocessing import standardPreprocessing, lemmatizeAll, tokenizeSentence
import os.path

def script():
    
    database  = GraphDatabase()
    filename = 'processedDocuments/Newsgroup_guns_motorcycles_10.pkl'
    minFrequency = 2

    if not os.path.exists(filename):
        print 'Load Documents'
        #data = fetch_20newsgroups(categories=['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))
	data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
	categories = data.target_names
	data = pd.DataFrame({'text': data['data'], 'category': data['target']})
	data = data[0:10]

        for index, category in enumerate(categories):
            print 'Category: ' + category + '   N: ' + str(len(data[data.category==index]))

        print 'Preprocessing'
        #standardPreprocessing(data, filename)
        docs = data.text.tolist()
        vectorizer = CountVectorizer(min_df=minFrequency, stop_words='english', token_pattern='[a-zA-Z]+')
        wordCounts = vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names()
        print('Number of Unique words: %d' % len(vocabulary))
        print('Minimal Frequency: %d' % minFrequency)

	#tfIdf = TfidfVectorizer(min_df = minFrequency, stop_words='english', token_pattern='[a-zA-Z]+')
	#tfIdfwords = tfIdf.fit_transform(docs)
	#vocabulary = tfIdf.get_feature_names()
	
	docsSplitInSentences = [sent_tokenize(doc) for doc in docs]
	tokenizedCollection = [[tokenizeSentence(sentence) for sentence in sentences] for sentences in docsSplitInSentences]

	cleanedTokens = [[[lemmatizeAll(word.lower()) for word in sentence if word.lower() in vocabulary] for sentence in doc] for doc in tokenizedCollection]
	cleanedTokens = [filter(None, doc) for doc in cleanedTokens]
	data['sentences'] = cleanedTokens

	allWords = [sum(elem, []) for elem in cleanedTokens]
	vocabulary = set()
	for article in allWords:
		vocabulary.update(article)
	vocabulary = dict(zip(vocabulary, range(0,len(vocabulary))))
	
	fullCleanText = [' '.join(sum(post, [])) for post in data.sentences.tolist()]
	data['cleanText'] = fullCleanText

	tfIdf = TfidfVectorizer(vocabulary=vocabulary)
	docs = data.cleanText.tolist()
	tfidf_vec = tfIdf.fit_transform(docs)
	data['tfIdf'] = [list(elem) for elem in tfidf_vec.toarray()]

	tf = CountVectorizer(vocabulary=vocabulary)
	tf_vec = tf.fit_transform(docs)
	data['tf'] = [list(elem) for elem in tf_vec.toarray()]

	# Remove posts with no features
	for index in range(len(data)):
		tfIdfSum = np.sum(data.loc[index, 'tfIdf'])
		if tfIdfSum==0:
			print index
			data.drop(index, inplace=True)
	data.index = range(len(data))
		
        data.to_pickle(filename)
	
	#data = pd.read_pickle(filename)
	#data = data[0:5]

	#toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
	#data = pd.DataFrame(toydata, columns=['category', 'sentences'])

	print 'Graph Construction'
	wordID = -1
	#wordID = 1
	startNode = database.createFeatureNode(wordID,'$Start$')
	for index, text in enumerate(data.sentences):
		print 'Document' + str(index)
		label = data.category.loc[index]
		docNode = database.createDocumentNode(index, label)
		for sentence in text:
			preceedingWord = startNode
			for word in sentence:
				exists = len(list(database.graph.find('Feature', property_key='word', property_value=word))) > 0
				if not exists:
					wordID = vocabulary[word]
					wordNode = database.createFeatureNode(wordID, word)
					wordID += 1
				else:
					wordNode = database.getFeatureNode(word)
					database.createWeightedRelation(wordNode, docNode, 'is_in')
					database.createWeightedRelation(preceedingWord, wordNode, 'followed_by')
					preceedingWord = wordNode

	print 'Normalize relationships'
	docNodes = database.getNodes('Document')
	database.normalizeRelationships(docNodes, 'is_in')
	
	featureNodes = database.getNodes('Feature')
	database.normalizeRelationships(featureNodes, 'followed_by')

	print 'Create Matrix'
	docMatrix = identity(len(docNodes))
	featureMatrix = database.getMatrix(featureNodes)
	featureDocMatrix = database.getMatrix(featureNodes, docNodes, 'is_in')
	docAll = np.concatenate((docMatrix, np.transpose(featureDocMatrix)), axis=1)
	featureAll = np.concatenate((featureDocMatrix, featureMatrix), axis=1)
	combinedMatrix = np.concatenate((docAll, featureAll))
	print combinedMatrix.shape
	np.save('guns_motorcycles_10', combinedMatrix)


if __name__ == '__main__':
    script()
