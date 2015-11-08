import unittest

from PyWANN.WiSARD import Discriminator, Memory

import numpy as np
from samples import *


class TestDiscriminator(unittest.TestCase):

    def test_create_discriminator(self):

        num_bits_addr = 3
        retina_length = 10

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }

        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr] )) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        self.assertIsNotNone(d)

    def test_check_correct_number_of_memories_exact_mapping(self):

        num_bits_addr = 4
        retina_length = 16

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }

        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr] )) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        self.assertEquals(len(d.get_memories()), 4)


    def test_check_correct_number_of_memories_not_mult_num_bits(self):

        num_bits_addr = 4
        retina_length = 15

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }

        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr] )) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        #  last memory will have 3 bits instead 4
        self.assertEquals(len(d.get_memories()), 4)


    def test_training_no_cumulative(self):

        # example of T classes in a grig of 3x3
        data1 = [1, 1, 1,
                 0, 1, 0,
                 0, 1, 0]

        data2 = [1, 1, 1,
                 0, 1, 0,
                 0, 0, 0]

        num_bits_addr = 3
        retina_length = 9
        is_cummulative = False

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }


        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr]), is_cummulative) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        d.add_training(data1)
        d.add_training(data2)

        #  first three elements of the retina
        mem_0 = d.get_memory(0)

        self.assertEquals (mem_0.get_value(0), 0)
        self.assertEquals (mem_0.get_value(1), 0)
        self.assertEquals (mem_0.get_value(2), 0)
        self.assertEquals (mem_0.get_value(3), 0)
        self.assertEquals (mem_0.get_value(4), 0)
        self.assertEquals (mem_0.get_value(5), 0)
        self.assertEquals (mem_0.get_value(6), 0)
        self.assertEquals (mem_0.get_value(7), 1)

        mem_1 = d.get_memory(1)
        self.assertEquals (mem_1.get_value(0), 0)
        self.assertEquals (mem_1.get_value(1), 0)
        self.assertEquals (mem_1.get_value(2), 1)
        self.assertEquals (mem_1.get_value(3), 0)
        self.assertEquals (mem_1.get_value(4), 0)
        self.assertEquals (mem_1.get_value(5), 0)
        self.assertEquals (mem_1.get_value(6), 0)
        self.assertEquals (mem_1.get_value(7), 0)        
    
        mem_2 = d.get_memory(2)
        self.assertEquals (mem_2.get_value(0), 1)
        self.assertEquals (mem_2.get_value(1), 0)
        self.assertEquals (mem_2.get_value(2), 1)
        self.assertEquals (mem_2.get_value(3), 0)
        self.assertEquals (mem_2.get_value(4), 0)
        self.assertEquals (mem_2.get_value(5), 0)
        self.assertEquals (mem_2.get_value(6), 0)
        self.assertEquals (mem_2.get_value(7), 0)

    def test_training_no_cumulative(self):

        # example of T classes in a grig of 3x3
        data1 = [1, 1, 1,
                 0, 1, 0,
                 0, 1, 0]

        data2 = [1, 1, 1,
                 0, 1, 0,
                 0, 0, 0]

        num_bits_addr = 3
        retina_length = 9
        is_cummulative = True

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }


        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr]), is_cummulative) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        d.add_training(data1)
        d.add_training(data2)

        #  first three elements of the retina
        mem_0 = d.get_memory(0)

        self.assertEquals (mem_0.get_value(0), 0)
        self.assertEquals (mem_0.get_value(1), 0)
        self.assertEquals (mem_0.get_value(2), 0)
        self.assertEquals (mem_0.get_value(3), 0)
        self.assertEquals (mem_0.get_value(4), 0)
        self.assertEquals (mem_0.get_value(5), 0)
        self.assertEquals (mem_0.get_value(6), 0)
        self.assertEquals (mem_0.get_value(7), 2)

        mem_1 = d.get_memory(1)
        self.assertEquals (mem_1.get_value(0), 0)
        self.assertEquals (mem_1.get_value(1), 0)
        self.assertEquals (mem_1.get_value(2), 2)
        self.assertEquals (mem_1.get_value(3), 0)
        self.assertEquals (mem_1.get_value(4), 0)
        self.assertEquals (mem_1.get_value(5), 0)
        self.assertEquals (mem_1.get_value(6), 0)
        self.assertEquals (mem_1.get_value(7), 0)        
    
        mem_2 = d.get_memory(2)
        self.assertEquals (mem_2.get_value(0), 1)
        self.assertEquals (mem_2.get_value(1), 0)
        self.assertEquals (mem_2.get_value(2), 1)
        self.assertEquals (mem_2.get_value(3), 0)
        self.assertEquals (mem_2.get_value(4), 0)
        self.assertEquals (mem_2.get_value(5), 0)
        self.assertEquals (mem_2.get_value(6), 0)
        self.assertEquals (mem_2.get_value(7), 0)


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

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }


        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr])) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        d.add_training(data1)
        d.add_training(data2)

        result = d.classify(test_positive)

        self.assertTrue(np.array_equal(result, [2,2,1]))

    def test_drasiw(self):

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

        positions = np.arange(retina_length)

        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                              for i in xrange(0, retina_length, num_bits_addr) }


        memories = { i/num_bits_addr:  Memory( len(mapping_positions[i/num_bits_addr])) \
                     for i in xrange(0,retina_length, num_bits_addr) }  


        d = Discriminator(retina_length, mapping_positions, memories)

        d.add_training(data1)
        d.add_training(data2)

        DRASiW =  d.get_DRASiW()

        self.assertTrue(np.array_equal(DRASiW, [2,2,2,
                                                0,2,0,
                                                0,1,0]))


if __name__ == "__main__":
     unittest.main()
