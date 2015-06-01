import unittest

from PyWANN import Retina, Wisard
from samples import *
import random
import numpy as np


class TestWisard(unittest.TestCase):



    def test_T_versus_A_without_bleaching(self):

        num_bits = 2
        retina_size = 64
        w = Wisard(retina_size, num_bits)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])


    def test_T_versus_A_without_bleaching_training_unique(self):

        num_bits = 2
        retina_size = 64
        w = Wisard(retina_size, num_bits)

        w.add_discriminator("A")
        w.add_discriminator("T")

        for ex in A_samples[0:-2]:
            w.add_training("A",ex)

        for ex in T_samples[0:-2]:
            w.add_training("T",ex)


        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue(A_test['A'] > A_test['T'])
        self.assertTrue(T_test['T'] > T_test['A'])


    def test_T_versus_A_with_bleaching(self):

        retina_size = 64
        num_bits = 2
        use_vacuum = False
        use_bleaching = True,
        confidence_threshold = 0.6
        randomize_positions = False

        w = Wisard(retina_size, num_bits, use_vacuum, use_bleaching,
                   confidence_threshold,randomize_positions)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))

    def test_T_versus_A_with_vacuum(self):

        retina_size = 64
        num_bits = 4
        use_vacuum = True

        w = Wisard(retina_size, num_bits, use_vacuum)

        w.add_discriminator("A", A_samples[0:-2])
        w.add_discriminator("T", T_samples[0:-2])

        A_test = w.classifier(A_samples[-1])  # first 9 samples
        T_test = w.classifier(T_samples[-1])  # first 9 samples

        self.assertTrue((A_test['A'] > A_test['T']) and
                        (T_test['T'] > T_test['A']))

    
    def test_k_fold_using_bleaching(self):

        # wisard parameters
        num_bits = 7
        retina_size = 18
        use_vacuum = True
        use_bleaching = False,
        confidence_threshold = 0.6

        # k-fold parameters
        k = 10
        annotation = Y
        base = X
        num_samples = len(annotation)
        fold_size = num_samples/k
        clazzs = set(annotation)

        # sorting positions
        aux_vec = range(0, num_samples )
        random.shuffle(aux_vec)

        # creating folds
        folds = {}
        for i in xrange(k):
            folds[i] = aux_vec[i*fold_size: (i+1)*fold_size]


        vec_result = []
        # testing data
        for i in folds:

            # training fold
            positions = []
            candidate_list = [folds[key] for key in folds if key != i]
            for candidates in candidate_list:
                for item in candidates:
                    positions.append(item)

        
            w = Wisard(retina_size,num_bits,
                       use_vacuum, use_bleaching, confidence_threshold)

            for clazz in clazzs:
                w.add_discriminator(clazz)


            for position in positions:
                class_name = annotation[position]
                value = base[position]
                w.add_training(class_name, value)


            #classifing
            acertos = 0
            for position in folds[i]:                 
                result = w.classifier(base[position])
                #print "RESULTADO: "+str(result)


                if result['0'] > result['1'] and annotation[position] == '0':
                    acertos += 1

                if result['1'] > result['0'] and annotation[position] == '1':
                    acertos += 1


            vec_result.append(acertos/float(len(folds[i])))

        

        mean = np.mean(vec_result)
        std = np.std(vec_result) 
        self.assertTrue(mean > 0.5 and std < 0.20)
        