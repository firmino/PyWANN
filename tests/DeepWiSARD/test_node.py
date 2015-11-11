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

    def test_classify_node(self):
		retina_length = 64
		num_bits_addr = 2

		
		node1 = Node(name="teste1", 
        		  retina_length=retina_length, 
        		  num_bits_addr=num_bits_addr) 


		node2 = Node(name="teste2", 
        		  retina_length=retina_length, 
        		  num_bits_addr=num_bits_addr) 


		node1.fit(A_samples[0:-2])
		node2.fit(T_samples[0:-2])
		    

		self.assertTrue( node1.predict(A_samples[-1]) > node2.predict(A_samples[-1]) )
		self.assertTrue( node2.predict(T_samples[-1]) > node1.predict(T_samples[-1]) )



    # def test_get_name(self):
    #     n = Node("teste1")
    #     self.assertEqual(n.get_name(), "teste1")

    # def test_set_parente(self):
    #     parent = Node("parent")
        
    #     child1 = Node("child1")
    #     child2 = Node("child2")

    #     child1.set_parent(parent)
    #     child2.set_parent(parent)

    #     #  son to parent
    #     self.assertEqual(child1.get_parent().get_name(), parent.get_name())
    #     self.assertEqual(child2.get_parent().get_name(), parent.get_name())

    # def test_propagate_value(self):

    #     parent = Node("parent")
        
    #     child1 = Node("child1")
    #     child1.set_value(10)

    #     child2 = Node("child2")
    #     child2.set_value(5)

    #     child1.set_parent(parent)
    #     child2.set_parent(parent)

    #     child1.get_parent().propagate_value( child1 ) 
    #     child2.get_parent().propagate_value( child2 ) 

    #     self.assertEqual(parent.get_best_child().get_name(), "child1")
    #     self.assertEqual(parent.get_best_child().get_value(), 10)
        
    # def test_confidence(self):
        
    #     parent = Node("parent")
        
    #     child1 = Node("child1")
    #     child1.set_value(10)

    #     child2 = Node("child2")
    #     child2.set_value(5)

    #     child1.set_parent(parent)
    #     child2.set_parent(parent)

    #     child1.get_parent().propagate_value( child1 ) 
    #     child2.get_parent().propagate_value( child2 ) 

    #     self.assertEqual(parent.get_confidence(), (1 - 5.0/10.0) )

    # def test_reset(self):

    #     parent = Node("parent")
        
    #     child1 = Node("child1")
    #     child1.set_value(10)

    #     child2 = Node("child2")
    #     child2.set_value(5)

    #     child1.set_parent(parent)
    #     child2.set_parent(parent)

    #     child1.get_parent().propagate_value( child1 ) 
    #     child2.get_parent().propagate_value( child2 ) 


    #     parent.reset()

    #     self.assertEqual(parent.get_best_child(), None)
    #     self.assertEqual(parent._Node__best_value, 0)
    #     self.assertEqual(parent._Node__second_best_value, 0)
        





if __name__ == "__main__":
    unittest.main()    
