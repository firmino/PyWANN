import unittest

from PyWANN import Discriminator, Retina


class TestDiscriminator(unittest.TestCase):

    def test_create_discriminator(self):
        d = Discriminator(9, 3)
        self.assertIsNotNone(d)

    def test_invalid_number_of_bits_addrs_for_retina_size(self):
        self.assertRaises(Exception, Discriminator, (8, 3))

    def test_check_correct_number_of_memories(self):
        d = Discriminator(12, 3)
        num_mem = len(d.get_memories())
        self.assertEquals(num_mem, 4)

    def test_random_mapping(self):
        d = Discriminator(12, 3)

        is_organized = True
        list_organized_positions = range(12)

        for memory_mapping in d._Discriminator__memories_mapping.values():
            str_key_a = str.join("", map(str, list_organized_positions))
            str_key_b = str.join("", map(str, memory_mapping))

            if str_key_b not in str_key_a:
                is_organized = False
                break

        self.assertFalse(is_organized)

    def test_mapping_without_randomize_position(self):
        d = Discriminator(12, 3, memories_values_cummulative=False, randomize_positions=False)

        is_organized = True
        list_organized_positions = range(12)

        for memory_mapping in d._Discriminator__memories_mapping.values():
            str_key_a = str.join("", map(str, list_organized_positions))
            str_key_b = str.join("", map(str, memory_mapping))

            if str_key_b not in str_key_a:
                is_organized = False
                break

        self.assertTrue(is_organized)

    def test_training_no_cumulative(self):

        # example of T classes in a grig of 3x3
        data1 = [[1, 1, 1],
                 [0, 0, 0],
                 [0, 3, 0]]

        data2 = [[4, 4, 4],
                 [0, 2, 0],
                 [0, 0, 0]]

        # creating retinas with data examples
        r1 = Retina(data1)
        r2 = Retina(data2)

        d = Discriminator(9, 3)
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
        # If we apply the mapping process for the expected result (that represents the same class)
        # we have to find 1 in all memories positions

        is_corrected_mapped = True
        for key in d.get_memories_mapping():
            posi_0 = d.get_memories_mapping()[key][0]  # get first mapping position
            posi_1 = d.get_memories_mapping()[key][1]  # get second mapping position
            posi_2 = d.get_memories_mapping()[key][2]  # get third mapping position

            addr = [expected_result[posi_0], expected_result[posi_1], expected_result[posi_2]]
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


if __name__ == "__main__":

    unittest.main()
