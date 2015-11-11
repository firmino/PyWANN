import unittest
from PyWANN.DeepWiSARD import Cluster
from samples import *

class TestCluster(unittest.TestCase):

    def test_create_cluster(self):
        
        retina_length = 16
        num_bits_addr = 3

        c = Cluster(retina_length, num_bits_addr)

        self.assertNotEquals(c, None)
    

    def test_add_training(self):

        retina_length = 16
        num_bits_addr = 2
        c = Cluster(retina_length, num_bits_addr)

        train1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        train2 = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


        #  how cluster is void, a new discriminator have be created
        c.add_training(train1)  
        self.assertEquals( len(c._Cluster__cluster), 1)

        #  adding a retina similar to train1, training happening in the same discriminator
        c.add_training(train2)  
        self.assertEquals( len(c._Cluster__cluster), 1)

        #  adding a diferent retina, a new discriminator have to be created
        train3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        c.add_training (train3)
        self.assertEquals( len(c._Cluster__cluster), 2)

        #  adding a retina similar to train3, two discriminator have to coexist
        train4 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        c.add_training (train4)
        self.assertEquals( len(c._Cluster__cluster), 2)


    def test_classify(self):

        retina_length = 64
        num_bits_addr = 2
        
        A_cluster = Cluster(retina_length, num_bits_addr, coverage_threshold = 0.95)

        T_cluster = Cluster(retina_length, num_bits_addr, coverage_threshold = 0.95)
        
        
        for retina_a in A_samples[0:-2]:
            A_cluster.add_training(retina_a)

        for retina_t in T_samples[0:-2]:
            T_cluster.add_training(retina_t)

        self.assertTrue( A_cluster.classify(A_samples[-1]) > T_cluster.classify(A_samples[-1]) )
        self.assertTrue( T_cluster.classify(T_samples[-1]) > A_cluster.classify(T_samples[-1]) )

