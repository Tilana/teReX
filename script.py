from GraphDatabase import GraphDatabase
from py3neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
import pandas as pd 
import numpy as np
from preprocessing import standardPreprocessing 
import os.path

def script():
    
    database  = GraphDatabase()
    filename = 'processedDocuments/Newsgroup_guns_motorcycles.pkl'

    if not os.path.exists(filename):
        print 'Load Documents'
        data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
        categories = data.target_names
        data = pd.DataFrame({'text': data['data'], 'category': data['target']})

        for index, category in enumerate(categories):
            print 'Category: ' + category + '   N: ' + str(len(data[data.category==index]))

        print 'Preprocessing'
        standardPreprocessing(data, filename)
        
    data = pd.read_pickle(filename)

    toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
    data = pd.DataFrame(toydata, columns=['category', 'sentences'])

    print 'Graph Construction'
    wordID = 1
    startNode = database.createFeatureNode(0,'$Start$')
    for index, text in enumerate(data.sentences[0:2]):
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


if __name__ == '__main__':
    script()
