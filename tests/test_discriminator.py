import unittest

from PyWANN import Discriminator, Retina


class TestDiscriminator(unittest.TestCase):

    def test_create_discriminator(self):
        d = Discriminator(9, 3)
        self.assertIsNotNone(d)

    def test_check_correct_number_of_memories_exact_mapping(self):
        d = Discriminator(12, 3) # retina lenght is a multiple of number of address' bits
        num_mem = len(d.get_memories())
        self.assertEquals(num_mem, 4)

    def test_check_correct_number_of_memories_not_exact_mapping(self):
        d = Discriminator(14, 3) # retina lenght is not a multiple of number of address'
                                 # bits
        num_mem = len(d.get_memories())
        self.assertEquals(num_mem, 5) # a addicional memory

    def test_random_mapping(self):
        d = Discriminator(12, 3)

        is_organized = True
        list_organized_positions = str.join("", map(str, range(12)) )

        str_key = ""
        for memory_mapping in d._Discriminator__memories_mapping.values():
            str_key += str.join("", map(str, memory_mapping))

        if str_key not in list_organized_positions:
            is_organized = False

        self.assertFalse(is_organized)

    def test_mapping_without_randomize_position(self):
        d = Discriminator(12, 3, memories_values_cummulative=False, randomize_positions=False)

        is_organized = True
        list_organized_positions = str.join("", map(str, range(12)) )

        str_key = ""
        for memory_mapping in d._Discriminator__memories_mapping.values():
            str_key += str.join("", map(str, memory_mapping))

        if str_key not in list_organized_positions:
            is_organized = False

        self.assertTrue(is_organized)


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

        d = Discriminator(9, 5) # will generate two memories (one with 5 bits and another
                                # with 4 bits of addressing)
        d.training([r1, r2])

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
        # If we apply the mapping process for the expected result (that represents the
        # same class) we have to find 1 in all memories positions

        is_corrected_mapped = True

        for key in d.get_memories_mapping():

            if(len(d.get_memories_mapping()[key]) == 4):
                posi_0 = d.get_memories_mapping()[key][0]  # get first mapping position
                posi_1 = d.get_memories_mapping()[key][1]  # get second mapping position
                posi_2 = d.get_memories_mapping()[key][2]  # get third mapping position
                posi_3 = d.get_memories_mapping()[key][3]  # get fourth mapping position

                addr = [expected_result[posi_0], expected_result[posi_1],
                        expected_result[posi_2], expected_result[posi_3]]
                if d.get_memory(key).get_value(addr) != 1:
                    is_corrected_mapped = False
                    break

            if(len(d.get_memories_mapping()[key]) == 5):
                posi_0 = d.get_memories_mapping()[key][0]  # get first mapping position
                posi_1 = d.get_memories_mapping()[key][1]  # get second mapping position
                posi_2 = d.get_memories_mapping()[key][2]  # get third mapping position
                posi_3 = d.get_memories_mapping()[key][3]  # get fourth mapping position
                posi_4 = d.get_memories_mapping()[key][4]  # get fifth mapping position

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

        d = Discriminator(9, 3, memories_values_cummulative=True)
        d.training([r1, r2])

        # as all positions in retinas are selected, it is just necessary check if
        # memories addressed by 1,1,1 have value 2

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

        d = Discriminator(9, 3)
        d.training([r1, r2])

        test_positive = [[1, 1, 1],
                         [0, 1, 0],
                         [0, 1, 0]]

        t_test = Retina(test_positive)

        self.assertEqual(d.classifier(t_test), 3)

    def test_classifier_negative(self):
        example_t1 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        example_t2 = [[1, 1, 1],
                      [0, 1, 0],
                      [0, 1, 0]]

        r1 = Retina(example_t1)
        r2 = Retina(example_t2)

        d = Discriminator(9, 3)
        d.training([r1, r2])

        test_positive = [[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]]

        t_test = Retina(test_positive)

        self.assertNotEqual(d.classifier(t_test), 3)


if __name__ == "__main__":

    unittest.main()
