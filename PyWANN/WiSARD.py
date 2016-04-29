# -*- coding: utf-8 -*-

import numpy as np
from Discriminator import Discriminator


class WiSARD:

    def __init__(self,
                 num_bits_addr,
                 bleaching=True,
                 memory_is_cumulative=True,
                 defaul_b_bleaching=1,
                 confidence_threshold=0.1,
                 ignore_zero_addr=False,
                 randomize_positions=True,
                 seed=424242):

        if (not isinstance(num_bits_addr, int)):
            raise Exception('num_bits must be a integer')

        if (not isinstance(bleaching, bool)):
            raise Exception('bleaching must be a boolean')

        if (not isinstance(memory_is_cumulative, bool)):
            raise Exception('memory_is_cumulative must be a boolean')

        if (not isinstance(defaul_b_bleaching, int)):
            raise Exception('defaul_b_bleaching must be a integer ')

        if (not isinstance(confidence_threshold, float)):
            raise Exception('confidence_threshold must be a float')

        if (not isinstance(ignore_zero_addr, bool)):
            raise Exception('ignore_zero_addr must be a boolean')

        if (not isinstance(randomize_positions, bool)):
            raise Exception('randomize_positions must be a boolean')

        if (not isinstance(seed, int)):
            raise Exception('seed must be a boolean')

        self.__num_bits_addr = num_bits_addr
        self.__bleaching = bleaching
        self.__memory_is_cumulative = memory_is_cumulative
        self.__defaul_b_bleaching = defaul_b_bleaching
        self.__confidence_threshold = confidence_threshold
        self.__ignore_zero_addr = ignore_zero_addr
        self.__randomize_positions = randomize_positions
        self.__seed = seed

        self.__discriminators = {}
        self.classes_ = []

    def get_num_bits(self):
        return self.__num_bits_addr

    # X is a matrix of retinas (each line will be a retina)
    # y is a list of label (each line defina a retina in the
    # same position in Y)
    def fit(self, X, y):
        # creating discriminators
        self.__retina_length = len(X[0])
        clazz = set(y)

        for clazz_name in clazz:
            d = Discriminator(retina_length=self.__retina_length,
                              num_bits_addr=self.__num_bits_addr,
                              memory_is_cumulative=self.__memory_is_cumulative,
                              ignore_zero_addr=self.__ignore_zero_addr,
                              random_positions=self.__randomize_positions,
                              seed=self.__seed)

            self.__discriminators[clazz_name] = d
        
        self.classes_ = self.__discriminators.keys()

        # add training
        num_samples = len(y)
        for i in xrange(num_samples):
            retina = X[i]
            label = y[i]

            if type(retina) is list:
                retina = np.array(retina)

            self.__discriminators[label].add_training(retina)

    #  X is a matrix of retinas (each line will be a retina)
    def predict(self, X):
        final_result = []

        results = self.predict_proba(X)
        for res in results:
            index = np.argmax(res)
            final_result.append(self.classes_[index])

        return final_result

    def predict_proba(self, X):
        result = []

        X = np.array(X)
        for x in X:

            if self.__bleaching:
                result_x = self.__predict_with_bleaching(x)
                result.append(result_x)
            else:
                result_x = self.__predict_without_bleaching(x)
                result.append(result_x)

        return np.array(result)

    def __predict_with_bleaching(self, x):
        b = self.__defaul_b_bleaching
        confidence = 0.0
        result_partial = None

        res_disc = np.array([self.__discriminators[class_name].predict(x)
                             for class_name in self.classes_])

        while confidence < self.__confidence_threshold:
            result_partial = np.sum(res_disc >= b, axis=1)

            confidence = self.__calc_confidence(result_partial)
            b += 1

            if(np.sum(result_partial) == 0):
                result_partial = np.sum(res_disc >= 1, axis=1)
                break
        result_sum = np.sum(result_partial, dtype=np.float32)
        result = np.array(result_partial)/result_sum

        return result

    def __predict_without_bleaching(self, x):
        res_disc = np.array([self.__discriminators[class_name].predict(x)
                             for class_name in self.classes_])
        result_partial = np.sum(res_disc, axis=1)
        result_sum = np.sum(result_partial, dtype=np.float32)
        result = np.array(result_partial)/result_sum
        return result

    def __calc_confidence(self, results):
        # getting max value
        max_value = results.max()
        if(max_value == 0):
            return 0

        # if there are two positions with same value
        position = np.where(results == max_value)
        if position[0].shape[0]>1:
            return 0

        # getting second max value
        second_max = results[results < max_value].max()
        if results[results < max_value].size > 0:
            second_max = results[results < max_value].max()

        # calculating confidence value
        c = 1 - float(second_max) / float(max_value)

        return c
