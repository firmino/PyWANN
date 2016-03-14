# import unittest

# from PyWANN.WiSARD  import WiSARD
# import numpy as np
# from samples import *


# class TestWiSARD(unittest.TestCase):

#     def test_T_versus_A_without_bleaching(self):

#         retina_length = 64
#         num_bits_addr = 2
#         bleaching = False
        
#         w = WiSARD(retina_length, num_bits_addr, bleaching)

#         w.create_discriminator("A")
#         w.create_discriminator("T")

#         # training discriminators
#         w.fit(A_samples[0:-2], ["A"]*len(A_samples[0:-2]))
#         w.fit(T_samples[0:-2], ["T"]*len(T_samples[0:-2]))

        
#         A_test = w.predict(A_samples[-1])  # first 9 samples
#         T_test = w.predict(T_samples[-1])  # first 9 samples
       	

#         self.assertTrue(A_test['A'] > A_test['T'])
#         self.assertTrue(T_test['T'] > T_test['A'])

#     def test_T_versus_A_with_bleaching(self):

#         retina_length = 64
#         num_bits_addr = 2
#         bleaching = True
        
#         w = WiSARD(retina_length, num_bits_addr, bleaching)

#         w.create_discriminator("A")
#         w.create_discriminator("T")

#         # training discriminators
#         w.fit(A_samples[0:-2], ["A"]*len(A_samples[0:-2]))
#         w.fit(T_samples[0:-2], ["T"]*len(T_samples[0:-2]))

#         A_test = w.predict(A_samples[-1])  # first 9 samples
#         T_test = w.predict(T_samples[-1])  # first 9 samples
       	
#         self.assertTrue(A_test['A'] > A_test['T'])
#         self.assertTrue(T_test['T'] > T_test['A'])


# if __name__ == "__main__":
#     unittest.main()