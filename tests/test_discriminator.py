import unittest

from PyWANN import Discriminator


class TestDiscriminator(unittest.TestCase):

    def test_create_discriminator(self):
        d = Discriminator(9, 3)
        self.assertIsNotNone(d)

    def test_invalid_number_of_bits_addrs_for_retina_size(self):
        self.assertRaises(Exception, Discriminator, (8, 3))

    def test_check_correct_number_of_memories(self):
        d = Discriminator(12 , 3)
        num_mem = len(d.get_memory())
        self.assertEquals(num_mem, 4)
        

if __name__ == "__main__":
    
    unittest.main()