from GraphDatabase import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
import pandas as pd 
import numpy as np
from numpy import transpose, identity
from preprocessing import standardPreprocessing, lemmatizeAll, tokenizeSentence
import os.path
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from scipy import sparse

def script():
    
    database  = GraphDatabase()
    filename = 'processedDocuments/Newsgroup_guns_motorcycles.pkl'

    minFrequency = 3

    if not os.path.exists(filename):
        print 'Load Documents'
        data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
        categories = data.target_names
        data = pd.DataFrame({'text': data['data'], 'category': data['target']})

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
        
        docsSplitInSentences = [sent_tokenize(doc) for doc in docs]

        tokenizedCollection = [[tokenizeSentence(sentence) for sentence in sentences] for sentences in docsSplitInSentences]

        cleanedTokens = [[[lemmatizeAll(word.lower()) for word in sentence if word.lower() in vocabulary] for sentence in doc] for doc in tokenizedCollection] 
        cleanedTokens = [filter(None, doc) for doc in cleanedTokens]
        data['sentences'] = cleanedTokens
        data.to_pickle(filename)
        
    data = pd.read_pickle(filename)

    #toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
    #data = pd.DataFrame(toydata, columns=['category', 'sentences'])

    print 'Graph Construction'
    wordID = 1
    startNode = database.createFeatureNode(0,'$Start$')
    for index, text in enumerate(data.sentences):
        print 'Document' + str(index)
        label = data.category.loc[index]
        docNode = database.createDocumentNode(index, label)
        for sentence in text:
            preceedingWord = startNode 
            for word in sentence:
                exists = len(list(database.graph.find('Feature', property_key='word', property_value=word))) > 0
                if not exists:
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
    featureMatrix = database.getMatrix(featureNodes)
    docFeatureMatrix = database.getMatrix(featureNodes, docNodes, 'is_in')
    combinedMatrix = np.concatenate((np.transpose(docFeatureMatrix),featureMatrix))
    print combinedMatrix
    np.save('NormMatrix', combinedMatrix)

    #combinedMatrix = np.load('NormMatrix600samples.npy')    
    #docNr = 300;
    #docdoc = identity(docNr)
    #docFeatures = combinedMatrix[0:docNr,:]
    #featureDoc = transpose(docFeatures)
    #featureFeature = combinedMatrix[601:601+docNr,:]

    #docAll = np.concatenate((docdoc, docFeatures), axis=1)
    #featureAll = np.concatenate((featureDoc, featureFeature), axis=1)
    #

    #X = np.concatenate((docAll,featureAll)) 
    #nrLabeledData = 100

    ##sparseMatrix = sparse.csr_matrix(combinedMatrix)
    #print 'Label Propagation'
    #labelPropagation = LabelPropagation(alpha=1) 
    #labels = np.ones([X.shape[0]])*-1
    #trueLabelIndex = range(0,nrLabeledData)
    #labels[trueLabelIndex] = data.loc[trueLabelIndex, 'category'].tolist()

    #labelPropagation.fit(X, labels)
    #predictLabels = labelPropagation.transduction_
    #print predictLabels
    #print accuracy_score(data.category[0:600].tolist(), predictLabels.tolist())


if __name__ == '__main__':
    script()
