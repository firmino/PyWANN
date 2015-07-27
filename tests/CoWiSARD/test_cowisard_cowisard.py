import unittest

from PyWANN.CoWiSARD  import CoWiSARD
from samples import *


class TestDiscriminator(unittest.TestCase):


    '''
    def test_classify(self):
        
        retina_width = 8
        retina_height = 8
        
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        conv_3 = [[1,0],[0,1]]
        list_conv = [conv_1, conv_2, conv_3]

        conv_box = (len(conv_1), len(conv_1[0]))

        cw = CoWiSARD(retina_width,
                       retina_height,
                       list_conv,
                       (2,2))

        cw.create_discriminator("A")
        cw.create_discriminator("T")


        # training discriminators
        for ex in A_samples[0:-2]:
            cw.train_discriminator("A", ex)

        for ex in T_samples[0:-2]:
            cw.train_discriminator("T", ex)

        # classifying
        A_test = cw.classify(A_samples[-1])  
        #T_test = cw.classify(T_samples[-1])  

        print A_test
        print "-"*20
        #print T_test


        #self.assertTrue((A_test['A'] > A_test['T']) and
        #                (T_test['T'] > T_test['A'])) 
    '''
    
if __name__ == "__main__":

    unittest.main()
