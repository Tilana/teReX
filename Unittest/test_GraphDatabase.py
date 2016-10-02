import unittest
from teReX import GraphDatabase

class test_GraphDatabase(unittest.TestCase):

    def setUp(self):
        self.database = GraphDatabase()
        self.database.createWordPairNodes([('This', 'is'), ('is', 'a'), ('a', 'test'), ('test', 'sentence')])

    def test_getNeighbours(self):
        self.assertEqual(self.database.getNeighbours('test'), set(['sentence']))
        self.assertEqual(self.database.getNeighbours('test', left=1), set(['a']))



if __name__ == '__main__':
    uniitest.main()
