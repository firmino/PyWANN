import unittest
from PyWANN.DeepWiSARD import Node
from samples import *

class TestNode(unittest.TestCase):

    def test_create_node(self):
        
        retina_length = 16
        num_bits_addr = 2

        n = Node(name="teste1", 
                 retina_length=retina_length, 
                 num_bits_addr=num_bits_addr)     
        self.assertNotEqual(n,None)

    def test_train_and_classify(self):
        retina_length = 64
        num_bits_addr = 2
        
        node_A = Node(name="node_A", 
                  retina_length=retina_length, 
                  num_bits_addr=num_bits_addr) 


        node_T = Node(name="node_A", 
                  retina_length=retina_length, 
                  num_bits_addr=num_bits_addr) 

        node_A.fit(A_samples[0:-2])
        node_T.fit(T_samples[0:-2])
        
        self.assertTrue( node_A.predict(A_samples[-1]) > node_T.predict(A_samples[-1]) )
        self.assertTrue( node_T.predict(T_samples[-1]) > node_A.predict(T_samples[-1]) )

    def test_add_and_get_child(self):

        retina_length = 64
        num_bits_addr = 2
        
        node_parent = Node(name="node_parent", 
                           retina_length=retina_length, 
                           num_bits_addr=num_bits_addr) 


        node_child = Node(name="node_child", 
                          retina_length=retina_length, 
                          num_bits_addr=num_bits_addr) 

        node_parent.add_child(node_child)
        self.assertEqual(id(node_child), id(node_parent.get_children()[0]))



if __name__ == "__main__":
    unittest.main()    
