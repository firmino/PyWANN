import unittest
from PyWANN.WiSARD import Discriminator, Retina


class TestDiscriminator(unittest.TestCase):

    def setUp(self):

        # creating a default mapping for all discriminators tests
        self.mapping_positions_9 = range(9)
        self.mapping_positions_14 = range(14)

    def test_create_discriminator(self):

        d = Discriminator(9, 3, self.mapping_positions_9)
        self.assertIsNotNone(d)

    def test_check_correct_number_of_memories_exact_mapping(self):

        # retina lenght is a multiple of number of address' bits
        d = Discriminator(12, 3, self.mapping_positions_9)
        num_mem = len(d.get_memories())
        self.assertEquals(num_mem, 4)

    def test_check_correct_number_of_memories_not_mult_num_bits(self):

        # retina lenght is not a multiple of number of address' bits
        d = Discriminator(14, 3, self.mapping_positions_14)
        num_mem = len(d.get_memories())
        self.assertEquals(num_mem, 5)  # a addicional memory

    def test_training_no_cumulative(self):

        # example of T classes in a grig of 3x3
        data1 = [[1, 1, 1],
                 [0, 1, 0],
                 [0, 3, 0]]

        data2 = [[4, 4, 4],
                 [0, 2, 0],
                 [0, 0, 0]]

        # creating retinas with data examples
        r1 = Retina(data1)
        r2 = Retina(data2)

        # will generate two memories (one with 5 bits and another with 4 bits
        # of addressing)
        d = Discriminator(9, 5, self.mapping_positions_9)
        d.train(r1)
        d.train(r2)

        # testing correct mapping
        expected_result = [1, 1, 1,
                           0, 1, 0,
                           0, 1, 0]

        # For example, in this case we have three memories so, if the
        # address' for the first memory (mem-1) is [3, 7, 8],
        # second memory (mem-2) is [4, 2, 5] and for the third (mem-3)
        # is [0, 1, 6] we will have the follow configuration for examples data1
        # and data2
        #
        # | MEMORY |  MAPPING | DATA1-Mapped | DATA2-Mapped
        # | mem-1  |  [3,7,8] | [0 1 0]      | [0 0 0]
        # | mem-2  |  [4,2,5] | [0 1 0]      | [1 1 0]
        # | mem-3  |  [0,1,6] | [1 1 0]      | [1 1 0]
        #
        # If we apply the mapping process for the expected result (that
        # represents the same class) we have to find 1 in all memories
        # positions

        is_corrected_mapped = True

        for key in d.get_memories_mapping():

            if(len(d.get_memories_mapping()[key]) == 4):
                posi_0 = d.get_memories_mapping()[key][0]  # get 1th position
                posi_1 = d.get_memories_mapping()[key][1]  # get 2th position
                posi_2 = d.get_memories_mapping()[key][2]  # get 3th position
                posi_3 = d.get_memories_mapping()[key][3]  # get 4th position

                addr = [expected_result[posi_0], expected_result[posi_1],
                        expected_result[posi_2], expected_result[posi_3]]
                if d.get_memory(key).get_value(addr) != 1:
                    is_corrected_mapped = False
                    break

            if(len(d.get_memories_mapping()[key]) == 5):
                posi_0 = d.get_memories_mapping()[key][0]  # get 1th position
                posi_1 = d.get_memories_mapping()[key][1]  # get 2th position
                posi_2 = d.get_memories_mapping()[key][2]  # get 3th position
                posi_3 = d.get_memories_mapping()[key][3]  # get 4th position
                posi_4 = d.get_memories_mapping()[key][4]  # get 5th position

                addr = [expected_result[posi_0], expected_result[posi_1],
                        expected_result[posi_2], expected_result[posi_3],
                        expected_result[posi_4]]

                if d.get_memory(key).get_value(addr) != 1:
                    is_corrected_mapped = False
                    break

        self.assertTrue(is_corrected_mapped)

    def test_training_cumulative(self):

        # example of T classes in a grig of 3x3
        data1 = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

        data2 = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

        # creating retinas with data examples
        r1 = Retina(data1)
        r2 = Retina(data2)

        d = Discriminator(9, 3, self.mapping_positions_9,
                          memories_values_cummulative=True)

        d.train(r1)
        d.train(r2)

        # as all positions in retinas are selected, it is just necessary check
        # if memories addressed by 1,1,1 have value 2

        is_corrected_mapped = True
        for memory in d.get_memories().values():

            if memory.get_value([1, 1, 1]) < 2:
                is_corrected_mapped = False
                break

        self.assertTrue(is_corrected_mapped)

    def test_classifier_positive(self):
        example_t1 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        example_t2 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        r1 = Retina(example_t1)
        r2 = Retina(example_t2)

        d = Discriminator(9, 3, self.mapping_positions_9)
        
        d.train(r1)
        d.train(r2)

        test_positive = [[1, 1, 1],
                         [0, 1, 0],
                         [0, 1, 0]]

        t_test = Retina(test_positive)

        list_memories_result = d.classify(t_test)
        self.assertEqual(sum(list_memories_result), 3)

    def test_classifier_negative(self):
        example_t1 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        example_t2 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        r1 = Retina(example_t1)
        r2 = Retina(example_t2)

        d = Discriminator(9, 3, self.mapping_positions_9)
        d.train(r1)
        d.train(r1)

        test_positive = [[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]]

        t_test = Retina(test_positive)

        list_memories_result = d.classify(t_test)
        self.assertNotEqual(sum(list_memories_result), 3)


if __name__ == "__main__":

    unittest.main()
