import unittest

from PyWANN.Memory  import Memory
from PyWANN.Discriminator  import Discriminator

import numpy as np
from samples import *


class TestDiscriminator(unittest.TestCase):

    def test_create_discriminator(self):
        num_bits_addr = 3
        retina_length = 10
        d = Discriminator(retina_length, num_bits_addr)
        self.assertIsNotNone(d)

    def test_check_correct_number_of_memories_exact_mapping(self):
        retina_length = 16
        num_bits_addr = 4
        d = Discriminator(retina_length, num_bits_addr)
        self.assertEquals(len(d._Discriminator__memories), 4)


    def test_check_correct_number_of_memories_not_mult_num_bits(self):
        retina_length = 15
        num_bits_addr = 4
        
        d = Discriminator(retina_length,num_bits_addr) 

        #  last memory will have 3 bits instead 4
        self.assertEquals(len(d._Discriminator__memories), 4)


    def test_classifier_positive(self):
        # example of T classes in a grig of 3x3
        data1 = [1, 1, 1,
                 0, 1, 0,
                 0, 1, 0]

        data2 = [1, 1, 1,
                 0, 1, 0,
                 0, 0, 0]

        test_positive = [1, 1, 1,
                         0, 1, 0,
                         0, 1, 0]

        num_bits_addr = 3
        retina_length = 9

        d = Discriminator(retina_length,
                          num_bits_addr,
                          random_positions=False)

        d.add_training(data1)
        d.add_training(data2)
        d.add_training(2)


        result = d.predict(test_positive)
        
        self.assertTrue(np.array_equal(result,[2,2,1] ))

    # def test_drasiw(self):

    #     # example of T classes in a grig of 3x3
    #     data1 = [1, 1, 1,
    #              0, 1, 0,
    #              0, 1, 0]

    #     data2 = [1, 1, 1,
    #              0, 1, 0,
    #              0, 0, 0]

    #     test_positive = [1, 1, 1,
    #                      0, 1, 0,
    #                      0, 1, 0]

    #     num_bits_addr = 3
    #     retina_length = 9

    #     positions = np.arange(retina_length)

    #     mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
    #                           for i in xrange(0, retina_length, num_bits_addr) }


    #     memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr])) \
    #                  for i in xrange(0,retina_length, num_bits_addr) }  


    #     d = Discriminator(retina_length, mapping_positions, memories)

    #     d.add_training(data1)
    #     d.add_training(data2)

    #     DRASiW =  d.get_DRASiW()

    #     self.assertTrue(np.array_equal(DRASiW, [2,2,2,
    #                                             0,2,0,
    #                                             0,1,0]))


if __name__ == "__main__":
     unittest.main()
