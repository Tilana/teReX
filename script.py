from GraphDatabase import GraphDatabase
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

def script():

    database  = GraphDatabase()
    reviews = pd.read_pickle('Documents/reviews_MusicalInstruments')
    reviewText = reviews['reviewText'].tolist()[15:17]
    tokenizedReviews = [sent_tokenize(text) for text in reviewText]

    for review in tokenizedReviews:
        for sentence in review:
            processedSentence = preprocess(sentence)
            wordPairs = createWordPairs(processedSentence)
            database.createWordPairNodes(wordPairs)
    database.graph.open_browser()
    
    distance = paradigSimilarity(database, 'cable', 'guitar')
    print distance
    
    distance = paradigSimilarity(database, 'cheap', 'good')
    print distance


def preprocess(sentence):
    return nltk.word_tokenize(sentence.lower().strip())

def createWordPairs(sentence):
    tupleList = []
    for i,word in enumerate(sentence):
        if i+1 < len(sentence):
            tupleList.append((word, sentence[i+1]))
    return tupleList

def jaccard(a,b):
    intSize = len(a.intersection(b))
    unionSize = len(a.union(b))
    return intSize / unionSize


def paradigSimilarity(database, w1, w2):
    return (jaccard(database.getNeighbours(w1,left=1), database.getNeighbours(w2, left=1)) + jaccard(database.getNeighbours(w1), database.getNeighbours(w2))) / 2.0


if __name__ == '__main__':
    script()
