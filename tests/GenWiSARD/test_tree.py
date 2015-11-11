import unittest
from PyWANN.GenWiSARD import Tree


class TestTree(unittest.TestCase):

    def test_create_tree(self):

        config_path = "tests/GenWiSARD/config.yaml"
        t = Tree(config_path = config_path, confidence_threshold=0.10)
        self.assertNotEqual(t, None)

    
    def test_leaf_on_tree(self):
        config_path = "tests/GenWiSARD/config.yaml"
        t = Tree(config_path = config_path, confidence_threshold=0.10)
        
        self.assertTrue("neto1" in t._Tree__node_leaf_index)
        self.assertTrue("neto2" in t._Tree__node_leaf_index)
        self.assertTrue("filho2" in t._Tree__node_leaf_index)

    def test_tree_structure(self):

        config_path = "tests/GenWiSARD/config.yaml"
        t = Tree(config_path = config_path, confidence_threshold=0.10)


        neto1 = t._Tree__node_leaf_index['neto1']
        self.assertEqual("neto1", neto1.get_name())

        filho1 = neto1.get_parent()
        self.assertEqual("filho1", filho1.get_name())
        
        root = filho1.get_parent()
        self.assertEqual("root", root.get_name())

        ########################################################

        neto2 = t._Tree__node_leaf_index['neto2']
        self.assertEqual("neto1", neto1.get_name())

        filho1 = neto2.get_parent()
        self.assertEqual("filho1", filho1.get_name())
        
        root = filho1.get_parent()
        self.assertEqual("root", root.get_name())

        ########################################################

        filho2 = t._Tree__node_leaf_index['filho2']
        self.assertEqual("filho2", filho2.get_name())
        
        root = filho2.get_parent()
        self.assertEqual("root", root.get_name())




if __name__ == "__main__":
    unittest.main()    
