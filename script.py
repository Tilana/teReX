from __future__ import division
from GraphDatabase import GraphDatabase
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
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

    toydata = [[0, [['This','is','it','.']]], [1,[['it','is','here','is','.']]]]
    data = pd.DataFrame(toydata, columns=['category', 'sentences'])

    print 'Graph Construction' 
    for index, text in enumerate(data.sentences[0:2]):
        print 'Document' + str(index)
        label = data.category.loc[index]
        docNode = database.graph.merge_one('Document', 'name', 'Doc '+str(index))
        database.updateNode(docNode, {'id':index, 'label':label, 'in-weight':0, 'out-weight':0})
        
        for sentence in text:
            preceedingWord = []
            for word in sentence:
                wordNode = database.graph.merge_one('Feature', 'word', word)
                if not wordNode.properties['in-weight']:
                    database.updateNode(wordNode, {'in-weight':0, 'out-weight':0})
                database.createWeightedRelation(wordNode, docNode, 'is_in')
                if preceedingWord:
                    database.createWeightedRelation(preceedingWord, wordNode, 'followed_by')
                preceedingWord = wordNode

    print 'Normalize relationships'
    docNodes = database.getNodes('Document')
    database.normalizeRelationships(docNodes, 'is_in')

    featureNodes = database.getNodes('Feature')
    database.normalizeRelationships(featureNodes, 'followed_by')



def split(text):
    return [nltk.word_tokenize(sentence.strip()) for sentence in text]

def lemmatize(tokens):
    WordNet = WordNetLemmatizer()
    return [WordNet.lemmatize(word) for word in tokens]

def lowerAndTokenize(text):
    return nltk.word_tokenize(text.lower())

def removeStopwords(tokens):
    stopwords = set(stop_words.ENGLISH_STOP_WORDS)
    stopwords.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])     
    return [word for word in tokens if word not in stopwords]

def jaccard(a,b):
    intSize = len(a.intersection(b))
    unionSize = len(a.union(b))
    return intSize / unionSize


def paradigSimilarity(database, w1, w2):
    return (jaccard(database.getNeighbours(w1,left=1), database.getNeighbours(w2, left=1)) + jaccard(database.getNeighbours(w1), database.getNeighbours(w2))) / 2.0


if __name__ == '__main__':
    script()
