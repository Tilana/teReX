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
    docMatrix = identity(len(docNodes))
    featureMatrix = database.getMatrix(featureNodes)
    featureDocMatrix = database.getMatrix(featureNodes, docNodes, 'is_in')
    docAll = np.concatenate((docMatrix, np.transpose(featureDocMatrix)), axis=1)
    featureAll = np.concatenate((featureDocMatrix, featureMatrix), axis=1)
    combinedMatrix = np.concatenate((docAll, featureAll))
    print combinedMatrix.shape
    np.save('NormMatrix', combinedMatrix)



if __name__ == '__main__':
    script()
