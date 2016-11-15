from __future__ import division
from py2neo import Graph, Node, Relationship
import webbrowser
import numpy as np

class GraphDatabase():

    def __init__(self):
        #os.system('~/neo4j-community-3.0.6/bin/neo4j console')
        webbrowser.open('http://localhost:7474/')
        self.graph = Graph('http://neo4j:zxmga21@localhost:7474/db/data')
        self.graph.delete_all()

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

    def updateNode(self, node, propertyDict):
        node.properties.update(propertyDict)
        self.graph.push(node)

    def normalizeRelationships(self, nodes, relation):
        for node in nodes:
            for rel in node.match_incoming(relation):
                rel['norm_weight'] = rel['weight']/node['in-weight']
                self.graph.push(rel)

    def getNodes(self, feature):
        recordList = self.graph.cypher.execute('MATCH (node:%s) RETURN node' % feature)
        return [record.node for record in recordList]

