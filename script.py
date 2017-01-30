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
	name = 'NG_guns_motorcycles_10'
	filename = 'processedDocuments/'+ name +'.pkl'
	minFrequency = 2

	if not os.path.exists(filename):
        	print 'Load Documents'
        	#data = fetch_20newsgroups(categories=['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))
		data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
		categories = data.target_names
		data = pd.DataFrame({'text': data['data'], 'category': data['target']})
		#data = data[0:10]

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

		cleanedTokens = [[[lemmatizeAll(word.lower()) for word in sentence if word.lower() in vocabulary and len(word)>1] for sentence in doc] for doc in tokenizedCollection]
		cleanedTokens = [filter(None, doc) for doc in cleanedTokens]
		data['sentences'] = cleanedTokens

		allWords = [sum(elem, []) for elem in cleanedTokens]
		vocabulary = set()
		for article in allWords:
			vocabulary.update(article)
		vocabList = list(vocabulary)
		vocabList.sort()
		vocabMapping = zip(vocabList, range(len(vocabulary)))
		vocabulary = dict(vocabMapping)
		
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
	
	data = pd.read_pickle(filename)
	allWords = [sum(elem, []) for elem in data.sentences.tolist()]
	vocabulary = set()
	for article in allWords:
		vocabulary.update(article)
	vocabList = list(vocabulary)
	vocabList.sort()
	vocabMapping = zip(vocabList, range(len(vocabulary)))
	vocabulary = dict(vocabMapping)

	#toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
	#data = pd.DataFrame(toydata, columns=['category', 'sentences'])

	print 'Graph Construction'
	startNode = database.createFeatureNode(-1,'$Start$')
	endNode = database.createFeatureNode(len(vocabulary), '$End$')
	for index, text in enumerate(data.sentences):
		print 'Document' + str(index)
		label = data.category.loc[index]
		docNode = database.createDocumentNode(index, label)
		for sentence in text:
			preceedingWord = startNode
			database.createWeightedRelation(startNode,docNode, 'is_in')
			for ind,word in enumerate(sentence):
				exists = len(list(database.graph.find('Feature', property_key='word', property_value=word))) > 0
				if not exists:
					wordID = vocabulary[word]
					wordNode = database.createFeatureNode(wordID, word)
					#wordID += 1
				else:
					wordNode = database.getFeatureNode(word)
				database.createWeightedRelation(wordNode, docNode, 'is_in')
				database.createWeightedRelation(preceedingWord, wordNode, 'followed_by')
				preceedingWord = wordNode
				if ind==len(sentence)-1:
					database.createWeightedRelation(wordNode, endNode, 'followed_by')
					database.createWeightedRelation(endNode, docNode, 'is_in')

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
	np.save('matrices/' + name, combinedMatrix)

	print 'Set Context Similarity'
        database.cypherContextSim()
        contextSim = database.getMatrix(featureNodes, relation='related_to', propertyType = 'contextSim')
	#np.save('matrices/' + name + '_contextSim', contextSim)

	print 'Create Context Similarity Matrix'
	c = len(vocabulary)
	contextSimilarity = np.zeros([c,c])
	m = 0
	for elem1 in vocabMapping:
		print 'Row' + str(elem1[1])
		for ind in range(m):
			contextSimilarity[elem1[1], ind] = database.contextSimilarity(elem1[0], vocabMapping[ind][0])
		m = m+1
	#np.save('matrices/' + name + '_contextSim', contextSimilarity)


if __name__ == '__main__':
    script()
