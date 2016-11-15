from GraphDatabase import GraphDatabase
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from nltk.tokenize import sent_tokenize
from preprocessing import lemmatize, lowerAndTokenize, split, removeStopwords
import os.path

def script():

    database  = GraphDatabase()
    filename = 'processedDocuments/Newsgroup_guns_motorcycles.pkl'

    if not os.path.exists(filename):
        print 'Pre-Processing Documents'
        data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
        categories = data.target_names
        data = pd.DataFrame({'text': data['data'], 'category': data['target']})

        print 'Number of Documents'
        for index, category in enumerate(categories):
            print 'Category: ' + category + '   N: ' + str(len(data[data.category==index]))
        
        data['tokens'] = data['text'].apply(lowerAndTokenize) 
        
        data['cleanText'] = data['tokens'].apply(lemmatize) 
        data['cleanText'] = data['cleanText'].apply(removeStopwords) 
        data['cleanText'] = data['cleanText'].apply(' '.join)

        data['sentences'] = data['cleanText'].apply(sent_tokenize)
        data['sentences'] = data['sentences'].apply(split)
                                                          
        data.to_pickle(filename)
    
    data = pd.read_pickle(filename)

    #toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
    #data = pd.DataFrame(toydata, columns=['category', 'sentences'])

    print 'Graph Construction'
    wordID = 0
    for index, text in enumerate(data.sentences[0:2]):
        print 'Document' + str(index)
        label = data.category.loc[index]
        docNode = database.graph.merge_one('Document', 'name', 'Doc '+str(index))
        database.updateNode(docNode, {'id':index, 'label':label, 'in-weight':0, 'out-weight':0})
        for sentence in text:
            preceedingWord = []
            for word in sentence:
                exists = len(list(database.graph.find('Feature', property_key='word', property_value=word))) > 0
                if not exists:
                    wordNode = Node('Feature', word=word)
                    database.graph.create(wordNode)
                    database.updateNode(wordNode, {'in-weight':0, 'out-weight':0, 'id':wordID})
                    wordID += 1 
                else:
                    wordNode = list(database.graph.find('Feature', property_key='word', property_value=word))[0]
                database.createWeightedRelation(wordNode, docNode, 'is_in')
                if preceedingWord:
                    database.createWeightedRelation(preceedingWord, wordNode, 'followed_by')
                preceedingWord = wordNode


    print 'Normalize relationships'
    docNodes = database.getNodes('Document')
    database.normalizeRelationships(docNodes, 'is_in')

    featureNodes = database.getNodes('Feature')
    database.normalizeRelationships(featureNodes, 'followed_by')

    print 'Create Matrix'
    import numpy as np
    nrFeatures = len(featureNodes)
    matrix = np.zeros([nrFeatures,nrFeatures])
    for node in featureNodes:
        rowIndex = node['id']
        for relation in node.match_outgoing('followed_by'):
               colIndex = relation.end_node['id']
               weight = relation['norm_weight']
               matrix[rowIndex, colIndex] = weight

    print matrix



if __name__ == '__main__':
    script()
