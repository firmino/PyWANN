import unittest

from PyWANN import Retina, Wisard
from samples import *


class TestWisard(unittest.TestCase):

    def test_T_versus_A_without_bleaching(self):

        num_bits = 2
        w = Wisard(num_bits)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])

    def test_T_versus_A_with_bleaching(self):

        num_bits = 2
        use_vacuum = False
        use_bleaching = True,
        confidence_threshold = 0.6
        randomize_positions = False

        w = Wisard(num_bits, use_vacuum, use_bleaching, confidence_threshold,
                   randomize_positions)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))

    def test_T_versus_A_with_vacuum(self):
        num_bits = 4
        use_vacuum = True

        w = Wisard(num_bits, use_vacuum)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))
