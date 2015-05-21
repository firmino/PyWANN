import unittest

from PyWANN import Retina, Wisard
from samples import *


class TestWisard(unittest.TestCase):

    def test_T_versus_A(self):

        num_bits = 4
        w = Wisard(num_bits)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])
