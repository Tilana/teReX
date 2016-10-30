from py2neo import Graph, Node, Relationship
import webbrowser

class GraphDatabase():

    def __init__(self):
        #os.system('~/neo4j-community-3.0.6/bin/neo4j console')
        webbrowser.open('http://localhost:7474/')
        
        self.graph = Graph('http://neo4j:zxmga21@localhost:7474/db/data')
        self.graph.delete_all()


    def getNeighbours(self, word, left=0):
        QUERY = '''MATCH (s:Word {word: {word}})
                MATCH (w:Word)<-[:NEXT_WORD]-(s)
                RETURN w.word as word'''
        if left:
            QUERY = '''MATCH (s:Word {word: {word}})
            MATCH (w:Word)-[:NEXT_WORD]->(s)
            RETURN w.word as word'''
            
        params = {'word': word}
        cypher= self.graph.cypher
        result = cypher.execute(QUERY, params) 
        return set(self.cypherRecordList2List(result))

    def cypherRecordList2List(self, recordList):
        return [elem.word for elem in recordList]

    def createWordPairNodes(self, tupleList):
        INSERT_WORDPAIR = '''
            FOREACH (t IN {wordPairs} |
                MERGE (w0:Word {word: t[0]})
                MERGE (w1:Word {word: t[1]})
                CREATE (w0)-[:NEXT_WORD]->(w1)
                )
        '''
        params = {'wordPairs': tupleList}
        self.graph.cypher.execute(INSERT_WORDPAIR, params)

    def createDocumentNode(self, ind):
        CREATE_DOCUMENT = '''CREATE (doc:Document {index: ind})'''
        params = {'index': ind}
        self.graph.cypher.execute(CREATE_DOCUMENT, params)



