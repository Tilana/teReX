import unittest
from py2neo import Node
from teReX import GraphDatabase

class test_GraphDatabase(unittest.TestCase):

    def setUp(self):
        self.database = GraphDatabase()

        toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
        data = pd.DataFrame(toydata, columns=['category', 'sentences'])


    #def test_updateNode(self):
    #    node = Node('Feature', word='test')
    #    target = Node('Feature', word='test', new_property='length')
    #    updatedNode = self.database.updateNode(node, {'new_property':'length'})
    #    self.assertEqual(
        
        




if __name__ == '__main__':
    uniitest.main()
