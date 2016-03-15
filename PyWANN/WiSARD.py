# -*- coding: utf-8 -*-

import numpy as np
from PyWANN import Discriminator

class WiSARD:

    def __init__(self,
                 retina_length,
                 num_bits_addr,
                 bleaching=True,
                 memory_is_cumulative=True,
                 defaul_b_bleaching=1,
                 confidence_threshold=0.1,
                 ignore_zero_addr=False, 
                 randomize_positions=True,
                 seed=424242,
                 output_use_softmax=False):

        if (not isinstance(retina_length, int)):
            raise Exception('retina_length must be a integer')

        if (not isinstance(num_bits_addr, int)):
            raise Exception('num_bits must be a integer')

        if (not isinstance(bleaching, bool)):
            raise Exception('bleaching must be a boolean')

        if (not isinstance(memory_is_cumulative, bool)):
            raise Exception('memory_is_cumulative must be a boolean')

        if (not isinstance(defaul_b_bleaching, int))
            raise Exception('defaul_b_bleaching must be a integer ')

        if (not isinstance(confidence_threshold, float)):
            raise Exception('confidence_threshold must be a float')

        if (not isinstance(ignore_zero_addr, bool)):
            raise Exception('ignore_zero_addr must be a boolean')

        if (not isinstance(randomize_positions, bool)):
            raise Exception('randomize_positions must be a boolean')

        if (not isinstance(seed, int)):
            raise Exception('seed must be a boolean')

         if (not isinstance(output_use_softmax, bool)):
            raise Exception('output_use_softmax must be a boolean')

        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr
        self.__bleaching = bleaching
        self.__memory_is_cumulative = memory_is_cumulative
        self.__defaul_b_bleaching = defaul_b_bleaching
        self.__confidence_threshold = confidence_threshold
        self.__ignore_zero_addr = ignore_zero_addr
        self.__randomize_positions = randomize_positions
        self.__seed = seed
        self.__output_use_softmax = output_use_softmax

        self.__discriminators = {}


    #  X is a matrix of retinas (each line will be a retina)
    #  y is a list of label (each line defina a retina in the same position in Y)
    def fit(self, X, y):
        # creating discriminators
        clazz = set(y)
        for clazz_name in clazz:
            disc = Discriminator(retina_length= self.__retina_length,
                                 num_bits_addr=self.__num_bits_addr,
                                 memory_is_cumulative=self.__memory_is_cumulative,
                                 ignore_zero_addr=self.__ignore_zero_addr,
                                 random_positions=self.__randomize_positions,
                                 seed=self.__seed)

            self.__discriminators[clazz_name] = disc

        # add training
        num_samples =  len(y)
        for i in xrange(num_samples):
            retina = X[i]
            label = y[i]
            self.__discriminators[label].add_training(retina)
        

    # 
    def predict(self, X):
        result = []
        discriminator_names = self.__discriminators.keys()

        for x in X:
            result_value = np.array([self.__discriminators[class_name].predict(x) \
                                     for class_name in discriminator_names] )

            result_sum = np.sum(result_value[:]>=1, axis=1) 
            
            if self.__bleaching:
                b = self.__defaul_b_bleaching
                confidence = self.__calc_confidence(result_sum)

                while confidence < self.__confidence_threshold:
                    result_sum = np.sum(result_value[:]>=b, axis=1)
                    confidence = self.__calc_confidence(result_sum)
                    b += 1

            result = { discriminator_names[i] : result_sum[i] for i in xrange(len(result_sum) )}

        return result
        
    def __calc_confidence(self,results):
            
        # getting max value
        max_value = results.max()
        if(max_value == 0):
            return 0

        # getting second max value
        second_max = max_value
        if results[results < max_value].size > 0:
            second_max = results[results < max_value].max()
        
        # calculating confidence value
        c = 1 - float(second_max) / float(max_value)

        return c
