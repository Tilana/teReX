#from py2neo  import Graph, Node, Relationship
from GraphDatabase import GraphDatabase

database  = GraphDatabase()

data = ['My cat eats fish on Sunday', 'Her dog is called Susan', 'He likes fish', 'The dog bites on Sunday']

for sentence in data:
    processedSentence = preprocess(sentence)
    wordPairs = createWordPairs(processedSentence)
    database.createWordPairNodes(wordPairs)




def preprocess(sentence):
    return sentence.lower().strip().split()

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


def paradigSimilarity(w1, w2):
    return (jaccard(getNeighbours(w1,LEFT1_QUERY), getNeighbours(w2, LEFT1_QUERY)) + jaccard(getNeighbours(w1, RIGHT1_QUERY), getNeighbours(w2, RIGHT1_QUERY))) / 2.0

