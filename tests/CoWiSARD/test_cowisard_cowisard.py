import unittest

from PyWANN.CoWiSARD  import CoWiSARD
from samples import *


class TestDiscriminator(unittest.TestCase):

    
    def test_classify(self):
        
        retina_width = 8
        retina_height = 8

        conv_width = 3
        conv_height = 3
        
        

        cw = CoWiSARD(retina_width,
                      retina_height,
                      conv_width, 
                      conv_height)        

        conv_matrix_1 = [[1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1]]

        conv_matrix_2 = [[1,0,1,1],
                        [1,1,1,0],
                        [1,1,1,1],
                        [1,0,1,1]]

        conv_matrix_3 = [[0,1,1,0],
                        [1,1,1,1],
                        [0,1,1,1],
                        [1,1,1,0]]


        cw.add_conv_matrix(conv_matrix_1)
        cw.add_conv_matrix(conv_matrix_2)
        cw.add_conv_matrix(conv_matrix_3)

        cw.create_discriminator("A")
        cw.create_discriminator("T")


        # training discriminators
        for ex in A_samples[0:-2]:
            cw.train_discriminator("A", ex)

        for ex in T_samples[0:-2]:
            cw.train_discriminator("T", ex)

        # classifying
        A_test = cw.classify(A_samples[-1])  
        T_test = cw.classify(T_samples[-1])  

        print A_test
        print "-"*20
        print T_test


        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A'])) 

    
if __name__ == "__main__":

    unittest.main()
