from __future__ import division
from py2neo import Graph, Node, Relationship, authenticate
import webbrowser
import numpy as np
import decimal

class GraphDatabase():

    	def __init__(self):
        	self.graph = Graph('http://neo4j:zxmga21@localhost:7474/db/data')
        	self.graph.delete_all()


    	def createDocumentNode(self, index, label):
		docNode = self.graph.merge_one('Document', 'name', 'Doc '+str(index))
        	self.updateNode(docNode, {'id':index, 'label':label, 'in-weight':0, 'out-weight':0})
        	return docNode


    	def createFeatureNode(self, index, word):
        	wordNode = Node('Feature', word=word) 
        	self.graph.create(wordNode)
        	self.updateNode(wordNode, {'in-weight':0, 'out-weight':0, 'id':index})
        	return wordNode


    	def getFeatureNode(self, word):
        	return list(self.graph.find('Feature', property_key='word', property_value=word))[0]


	def createWeightedRelation(self, node1, node2, relation):
        	match = self.graph.match(start_node=node1, rel_type=relation, end_node=node2) 
        	numberOfRelations= sum(1 for x in match)
        	if numberOfRelations >= 1:
        	    match = self.graph.match(start_node=node1, rel_type=relation, end_node=node2) 
        	    for relationship in match: 
        	        self.increaseWeight(relationship)
        	        self.increaseWeight(node1, 'out-weight')
        	        self.increaseWeight(node2, 'in-weight')
        	else:
        	    newRelation = Relationship(node1, relation, node2, weight=1)
        	    self.graph.create(newRelation)
        	    self.increaseWeight(node1, 'out-weight')
        	    self.increaseWeight(node2, 'in-weight')


    	def increaseWeight(self, entity, weight='weight'):
    	    	entity[weight] = entity[weight]+1
    	    	self.graph.push(entity)

    	def 	updateNode(self, node, propertyDict):
    	    	node.properties.update(propertyDict)
    	    	self.graph.push(node)

    	def 	normalizeRelationships(self, nodes, relation):
    	    	for node in nodes:
    	    	    for rel in node.match_incoming(relation):
    	    	        rel['norm_weight'] = rel['weight']/node['in-weight']
    	    	        self.graph.push(rel)

    	def 	getNodes(self, feature):
    	    	recordList = self.graph.cypher.execute('MATCH (node:%s) RETURN node' % feature)
    	    	return [record.node for record in recordList]


    	def 	getMatrix(self, nodesX, nodesY=None, relation='followed_by', propertyType='norm_weight'):
    	    	if nodesY == None:
    	    	    nodesY = nodesX
    	    	matrix = np.zeros([len(nodesX),len(nodesY)])
    	    	for node in nodesX:
    	    	    rowIndex = node['id']
    	    	    for outRelation in node.match_outgoing(relation):
    	    	           colIndex = outRelation.end_node['id']
    	    	           weight = outRelation[propertyType]
    	    	           matrix[rowIndex, colIndex] = weight
    	    	return matrix


	def cypherContextSim(self):
		tx = self.graph.cypher.begin()
		tx.append(CONTEXT_SIM)
		tx.commit()


CONTEXT_SIM = '''
	MATCH (s:Feature)
	// Get right1, left1
	MATCH (w:Feature)-[:followed_by]->(s)
	WITH collect(DISTINCT w.word) as left1, s
	MATCH (w:Feature)<-[:followed_by]-(s)
	WITH left1, s, collect(DISTINCT w.word) as right1
	// Match every other word
	MATCH (o:Feature) WHERE NOT s = o
	WITH left1, right1, s, o
	// Get other right, other left1
	MATCH (w:Feature)-[:followed_by]->(o)
	WITH collect(DISTINCT w.word) as left1_o, s, o, right1, left1
	MATCH (w:Feature)<-[:followed_by]-(o)
	WITH left1_o, s, o, right1, left1, collect(DISTINCT w.word) as right1_o
	// compute right1 union, intersect
	WITH FILTER(x IN right1 WHERE x IN right1_o) as r1_intersect,
	  (right1 + right1_o) AS r1_union, s, o, right1, left1, right1_o, left1_o
	// compute left1 union, intersect
	WITH FILTER(x IN left1 WHERE x IN left1_o) as l1_intersect,
	  (left1 + left1_o) AS l1_union, r1_intersect, r1_union, s, o
	WITH DISTINCT r1_union as r1_union, l1_union as l1_union, r1_intersect, l1_intersect, s, o
	WITH 1.0*size(r1_intersect) / size(r1_union) as r1_jaccard,
	  1.0*size(l1_intersect) / size(l1_union) as l1_jaccard,
	  s, o
	WITH s, o, r1_jaccard, l1_jaccard, r1_jaccard + l1_jaccard as sim
	CREATE UNIQUE (s)-[r:related_to]->(o) SET r.contextSim= sim;
	'''
