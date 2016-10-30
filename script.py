from GraphDatabase import GraphDatabase
from py2neo import Graph, Node, Relationship
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def script():

    database  = GraphDatabase()

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
    
    
    for index, text in enumerate(data.sentences[0:5]):
        print 'Document' + str(index)
        label = data.category.loc[index]
        docNode = database.graph.merge_one('Document', 'name', 'Doc '+str(index))
        docNode.properties.update({'id':index, 'label':label})
        database.graph.push(docNode)
        for sentence in text:
            processedSentence = preprocess(sentence)
            wordPairs = createWordPairs(processedSentence)
            for wordPair in wordPairs:
                word1 = database.graph.merge_one('Feature', 'word', wordPair[0])
                word2 = database.graph.merge_one('Feature', 'word', wordPair[1])
                database.graph.create(Relationship(word1, 'followed by', word2))
                database.graph.create((docNode, 'contains', word1))
                database.graph.create((docNode, 'contains', word2))

            #database.createWordPairNodes(wordPairs)
    
    #distance = paradigSimilarity(database, 'cable', 'guitar')
    #print distance
    
    #distance = paradigSimilarity(database, 'cheap', 'good')
    #print distance


def preprocess(sentence):
    return nltk.word_tokenize(sentence.strip())

def lemmatize(tokens):
    WordNet = WordNetLemmatizer()
    return [WordNet.lemmatize(word) for word in tokens]

def lowerAndTokenize(text):
    return nltk.word_tokenize(text.lower())

def removeStopwords(tokens):
    stopwords = set(stop_words.ENGLISH_STOP_WORDS)
    stopwords.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])     
    return [word for word in tokens if word not in stopwords]

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
