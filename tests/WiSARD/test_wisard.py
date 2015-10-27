import unittest

from PyWANN.WiSARD  import Retina, WiSARD
from samples import *
import random
import numpy as np


class TestWiSARD(unittest.TestCase):

    def test_T_versus_A_without_bleaching(self):

        num_bits = 2
        retina_size = 64
        w = WiSARD(retina_size, num_bits)

        w.create_discriminator("A")
        w.create_discriminator("T")

        # training discriminators
        for sample in A_samples[0:-2]:
            w.add_training("A", sample)

        for sample in T_samples[0:-2]:
            w.add_training("T", sample)

        # classifying
        A_test = w.classify(A_samples[-1])  # first 9 samples
        T_test = w.classify(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])

    def test_T_versus_A_without_bleaching_training_unique(self):

        num_bits = 2
        retina_size = 64
        w = WiSARD(retina_size, num_bits)

        w.create_discriminator("A")
        w.create_discriminator("T")

        # training discriminators
        for ex in A_samples[0:-2]:
            w.add_training("A", ex)

        for ex in T_samples[0:-2]:
            w.add_training("T", ex)

        # classifying
        A_test = w.classify(A_samples[-1])  # first 9 samples
        T_test = w.classify(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])

    def test_T_versus_A_with_bleaching(self):

        retina_size = 64
        num_bits = 2
        use_vacuum = False
        use_bleaching = True,
        confidence_threshold = 0.6
        randomize_positions = False

        w = WiSARD(retina_size, num_bits,
                   use_vacuum, use_bleaching,
                   confidence_threshold, randomize_positions)

        w.create_discriminator("A")
        w.create_discriminator("T")

        # training discriminators
        for ex in A_samples[0:-2]:
            w.add_training("A", ex)

        for ex in T_samples[0:-2]:
            w.add_training("T", ex)

        # classifying
        A_test = w.classify(A_samples[-1])  # first 9 samples
        T_test = w.classify(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))

    def test_T_versus_A_with_vacuum(self):

        retina_size = 64
        num_bits = 4
        use_vacuum = True

        w = WiSARD(retina_size, num_bits, use_vacuum)

        w.create_discriminator("A")
        w.create_discriminator("T")

        # training discriminators
        for ex in A_samples[0:-2]:
            w.add_training("A", ex)

        for ex in T_samples[0:-2]:
            w.add_training("T", ex)

        # classifying
        A_test = w.classify(A_samples[-1])  # first 9 samples
        T_test = w.classify(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))
