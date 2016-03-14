# -*- coding: utf-8 -*-

import numpy as np


from PyWANN import Discriminator



class WiSARD:

    def __init__(self,
                 retina_length,
                 num_bits_addr,
                 bleaching=True,
                 confidence_threshold=0.1,
                 ignore_zero_addr=False, 
                 defaul_b_bleaching=1,
                 randomize_positions=True,
                 memory_is_cumulative=True):

        self.__retina_length = retina_length
        self.__bleaching = bleaching
        self.__defaul_b_bleaching = defaul_b_bleaching
        self.__discriminators = {}
        self.__confidence_threshold = confidence_threshold
        
        positions = np.arange(retina_length)
        if randomize_positions:
            np.random.shuffle(positions)

        #  spliting positions for each memory
        self.__mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                                     for i in xrange(0, retina_length, num_bits_addr) }

        #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
        self.__memories_template = { i/num_bits_addr:  Memory( num_bits_addr = len(self.__mapping_positions[i/num_bits_addr] ), 
                                                               is_cummulative = memory_is_cumulative,
                                                               ignore_zero_addr = ignore_zero_addr)  \
                                   for i in xrange(0,retina_length, num_bits_addr)}



    def create_discriminator(self, name):
        #  have to copy() memories or all discriminator will have the same set of memories
        new_memories = copy.deepcopy(self.__memories_template)
        self.__discriminators[name] = Discriminator(retina_length = self.__retina_length,
                                                    mapping_positions = self.__mapping_positions,
                                                    memories = new_memories) 
        


    #  X is a matrix of retinas (each line will be a retina)
    #  y is a list of label (each line defina a retina in the same position in Y)
    def fit(self, X, y):
        num_samples =  len(y)
        for i in xrange(num_samples):
            retina = X[i]
            label = y[i]
            self.__discriminators[label].add_training (retina)
        
    def predict(self, x):
        
        discriminator_names = [class_name for class_name in self.__discriminators]

        result_value = np.array( [ self.__discriminators[class_name].classify(x) \
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
