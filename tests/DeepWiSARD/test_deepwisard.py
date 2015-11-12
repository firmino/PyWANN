import unittest

from PyWANN.DeepWiSARD import DeepWiSARD
from samples import *


class TestDeepWiSARD(unittest.TestCase):

    def test_create_tree(self):
 
        config_path = "tests/DeepWiSARD/config.yaml"
        num_bits_addr = 3
        retina_length = 64
        coverage_threshold = 0.8
        confidence_threshold = 0.2
        randomize_positions = True

        t = DeepWiSARD(config_path,
                       retina_length,
                       num_bits_addr,
                       coverage_threshold,
                       confidence_threshold,
                       randomize_positions)

        #  config.yaml has 5 nodes
        self.assertEqual( len(t._DeepWiSARD__nodes), 5)


    def test_fit_and_predict(self):

        config_path = "tests/DeepWiSARD/config_2.yaml"
        retina_length = 64
        num_bits_addr = 3
        coverage_threshold = 0.9
        confidence_threshold = 0.1
        randomize_positions = True

        y = ['A','T']
        X = {'A':A_samples, 'T':T_samples}


        t = DeepWiSARD(config_path,
                       retina_length,
                       num_bits_addr,
                       coverage_threshold,
                       confidence_threshold,
                       randomize_positions)

        #  training the deepWiSARD
        t.fit(X,y)
        self.assertTrue(len(t._DeepWiSARD__nodes) > 0)


        #  predicting values
        count = 0
        for x in X["A"]:
            result = t.predict(x)
            if result == "A":
                count += 1

        self.assertEqual(count, len(X["A"]))


        count = 0
        for x in X["T"]:
            result = t.predict(x)
            if result == "T":
                count += 1

        self.assertEqual(count, len(X["T"]))
        


if __name__ == "__main__":
    unittest.main()    
