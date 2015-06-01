import unittest

from PyWANN import Retina, Wisard
from samples import *
import random


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


    def test_k_fold(self):

        # wisard parameters
        num_bits = 1
        retina_size = 3
        use_vacuum = False
        use_bleaching = False,
        confidence_threshold = 0.6
        randomize_positions = True

        # k-fold parameters
        k = 3
        annotation =["A","B","A","B","A","A"]
        base = [[0,0,0],[0,0,1],[0,0,0],[0,0,1],[0,0,0],[0,0,0]]
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

        # testing data
        for i in folds:

            # training fold
            positions = []
            candidate_list = [folds[key] for key in folds if key != i]
            for candidates in candidate_list:
                for item in candidates:
                    positions.append(item)

            w = Wisard(retina_size,num_bits,
                       use_vacuum, use_bleaching)

            for clazz in clazzs:
                w.add_discriminator(clazz)

            for position in positions:
                class_name = annotation[position]
                value = base[position]
                w.add_training(class_name, value)

            #classifing
            for position in folds[i]:
                print "O QUE ERA PRA SER: "+annotation[position]
                print "O QUE FOI: "+str(w.classifier(base[position]))
