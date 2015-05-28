import unittest

from PyWANN import Retina, Wisard, Bleaching
from samples import *


class TestWisard(unittest.TestCase):

    '''
    def test_T_versus_A_without_bleaching(self):

        num_bits = 4
        w = Wisard(num_bits)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        print "here"
        print A_test
        print T_test

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])

        print "#"*30
    '''

    def test_T_versus_A_with_bleaching(self):
        num_bits = 4

        confidence_threshold = 0.9
        is_cumulative = True
        bleaching = Bleaching(3)

        w = Wisard(num_bits, confidence_threshold, is_cumulative, bleaching)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        print "PRIMEIRO O A"
        A_test = w.classifier(A_samples[-1])  # first 9 samples
        print A_test
        print "-"*100

        print "SEGUNDO O T"
        T_test = w.classifier(T_samples[-1])  # first 9 samples
        print T_test

        #self.assertTrue(A_test['A'] > A_test['T'])
        #self.assertTrue(T_test['T'] > T_test['A'])

        print "#"*30
