import unittest
from . import GraphDatabase

class test_GraphDatabase(unittest.TestCase):

    def setUp(self):
        self.database = GraphDatabase()
        self.database.creatWordPairNodes([('This', 'is'), ('is', 'a'), ('a', 'test'), ('test', 'sentence')])

    def test_getNeighbours(self):
        self.assertEqual(self.database.getNeighbours('test'), 'sentence')


if __name__ == '__main__':
    uniitest.main()
