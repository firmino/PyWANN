import unittest

from PyWANN.CoWiSARD  import CoWiSARD
from samples import *


class TestDiscriminator(unittest.TestCase):


    '''      
    def test_classify(self):
        
        retina_width = 8
        retina_height = 8
        
        list_conv_matrix = [  [[ -1,  0,  1],
                               [ -1,  0,  1],
                               [ -1,  0,  1]],

                              [[ -1, -1, -1],
                               [  0,  0,  0],
                               [  1,  1,  1]],

                              [[  0,  1,  1],
                               [ -1,  0,  1],
                               [ -1, -1,  0]],

                               [[ 1,  1,   0],
                                [ 1,  0,  -1],
                                [ 0, -1,  -1]] ]


        retina_width = len(A_samples[0])
        retina_height = len(A_samples[0][0])
        num_bits = 2
        num_memo_to_combine = 2

        cw = CoWiSARD(retina_width,
                      retina_height,
                      num_bits,
                      list_conv_matrix,
                      num_memo_to_combine)
        

        cw.create_discriminator("A")
        cw.create_discriminator("T")


        # training discriminators
        for ex in A_samples[0:-2]:
            cw.add_trainning("A", ex)

        for ex in T_samples[0:-2]:
            cw.add_trainning("T", ex)

        # classifying
        A_test = cw.classify(A_samples[-1])  
        T_test = cw.classify(T_samples[-1])  

        

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A'])) 
    '''
    
if __name__ == "__main__":

    unittest.main()
