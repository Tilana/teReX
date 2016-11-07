#encoding=utf8
from GraphDatabase import GraphDatabase
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import os.path

def script_wine():

    database  = GraphDatabase()
    filename = 'processedWineReviews.pkl'

    if not os.path.exists(filename): 
        print 'Pre-Processing Documents'
        path = 'Documents/WineReviews/metamatrix3b.random.csv'
        columns = ['filename', 'website', 'color', 'fortfied', 'sparkling', 'name', 'year', 'norm_price', 'price', 'norm_variety', 'variety', 'producer', 'designation', 'ABV', 'region', 'country', 'import', 'rating', 'rating_norm', 'expert_name', 'review', 'date', 'user_review']
        
        data = pd.read_csv('Documents/WineReviews/metamatrix3b.random.csv', sep='\t', lineterminator='\n', header=None, names=columns, encoding='utf8')
        data = data.drop(['filename', 'website', 'fortfied', 'sparkling', 'norm_variety', 'variety', 'producer', 'designation', 'ABV', 'import', 'expert_name', 'date'], axis=1)
        data = data.replace(r'unk', np.nan, regex=True)
        data = data.dropna(subset=['color', 'review', 'country'])

   
        data['tokens'] = data['review'].apply(lowerAndTokenize) 
        
        data['cleanText'] = data['tokens'].apply(lemmatize) 
        data['cleanText'] = data['cleanText'].apply(removeStopwords) 
        data['cleanText'] = data['cleanText'].apply(' '.join)

        data['sentences'] = data['cleanText'].apply(sent_tokenize)
        data['sentences'] = data['sentences'].apply(split)

        data.to_pickle(filename)
   
    
    data = pd.read_pickle(filename)

    print 'Graph Construction' 
    for index, text in enumerate(data.sentences[0:3]):
        print 'Review' + str(index)
        label = data.color.loc[index]
        docNode = database.graph.merge_one('Review', 'name', 'review '+ str(index))
        docNode.properties.update({'id':index, 'label':label})
        database.graph.push(docNode)
        for sentence in text:
            preceedingWord = []
            for word in sentence:
                wordNode = database.graph.merge_one('Feature', 'word', word)
                database.createWeightedRelation(docNode, wordNode, 'contains')
                if preceedingWord:
                    database.createWeightedRelation(preceedingWord, wordNode, 'followed by')
                preceedingWord = wordNode


    #distance = paradigSimilarity(database, 'cable', 'guitar')
    #print distance
    
    #distance = paradigSimilarity(database, 'cheap', 'good')
    #print distance


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
    script_wine()
