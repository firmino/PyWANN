import unittest
from PyWANN.DeepWiSARD import DiscriminatorFactory
from PyWANN.WiSARD import Discriminator

class TestDiscriminatorFactory(unittest.TestCase):

    def test_create_discriminator(self):
        
        retina_length = 14
        num_bits_addr = 3
        randomize_positions = True

        disc_factory = DiscriminatorFactory( retina_length=retina_length, 
        								     num_bits_addr=num_bits_addr,
        								     randomize_positions=randomize_positions )

        discriminator1 = disc_factory.generate_discriminator()
        self.assertNotEqual(discriminator1, None)



if __name__ == "__main__":
    unittest.main()    
