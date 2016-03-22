import unittest

from PyWANN.WiSARD  import WiSARD
import numpy as np
from samples import *


class TestWiSARD(unittest.TestCase):

    def test_T_versus_A_without_bleaching(self):

        num_bits_addr = 2
        bleaching = False
        
        w = WiSARD(num_bits_addr, bleaching)

        X = A_samples[0:-2] + T_samples[0:-2]
        y = ["A"]*len(A_samples[0:-2]) + ["T"]*len(T_samples[0:-2])

        # training discriminators
        w.fit(X, y)

        A_test = w.predict(A_samples)
        T_test = w.predict(T_samples)

        tot_A = np.sum([1 for x in A_test if x == 'A'])
        tot_T = np.sum([1 for x in T_test if x == 'T'])

        self.assertEqual(tot_A, len(A_samples))
        self.assertEqual(tot_T, len(T_samples))

    def test_T_versus_A_with_bleaching(self):

        num_bits_addr = 2
        bleaching = True
        
        w = WiSARD(num_bits_addr, bleaching)

        X = A_samples[0:-2] + T_samples[0:-2]
        y = ["A"]*len(A_samples[0:-2]) + ["T"]*len(T_samples[0:-2])

        # training discriminators
        w.fit(X, y)


        A_test = w.predict(A_samples)  # first 9 samples
        T_test = w.predict(T_samples)  # first 9 samples

        tot_A = np.sum([1 for x in A_test if x == 'A'])
        tot_T = np.sum([1 for x in T_test if x == 'T'])

        self.assertEqual(tot_A, len(A_samples))
        self.assertEqual(tot_T, len(T_samples))

    def test_T_versus_A_with_bleaching_and_softmax(self):

        num_bits_addr = 2
        bleaching = True
        softmax = True
        
        w = WiSARD(num_bits_addr,
                   bleaching,
                   use_softmax=softmax)

        X = A_samples[0:-2] + T_samples[0:-2]
        y = ["A"]*len(A_samples[0:-2]) + ["T"]*len(T_samples[0:-2])

        # training discriminators
        w.fit(X, y)


        A_test = w.predict(A_samples)  # first 9 samples
        T_test = w.predict(T_samples)  # first 9 samples

        tot_A = np.sum([1 for x in A_test if x == 'A'])
        tot_T = np.sum([1 for x in T_test if x == 'T'])

        self.assertEqual(tot_A, len(A_samples))
        self.assertEqual(tot_T, len(T_samples))

    def test_T_versus_A_with_bleaching_and_softmax_predict_proba(self):

        num_bits_addr = 2
        bleaching = True
        softmax = True
        
        w = WiSARD(num_bits_addr,
                   bleaching,
                   use_softmax=softmax)

        X = A_samples[0:-2] + T_samples[0:-2]
        y = ["A"]*len(A_samples[0:-2]) + ["T"]*len(T_samples[0:-2])

        # training discriminators
        w.fit(X, y)

        res = w.predict_proba(A_samples)
        prob_a = np.mean(res[:,0])
        prob_t = np.mean(res[:,1])
        self.assertTrue(prob_a > prob_t)

        res = w.predict_proba(T_samples)
        prob_a = np.mean(res[:,0])
        prob_t = np.mean(res[:,1])
        self.assertTrue(prob_t > prob_a)

if __name__ == "__main__":
    unittest.main()