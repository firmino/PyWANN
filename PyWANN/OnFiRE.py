# -*- coding: utf-8 -*-
import numpy as np


class OnFiRE:

    def __init__(self,
                 retina_length,
                 class_names,
                 regularization_lambda=13,
                 bagging_percentage=63.7,
                 min_num_bits=8,
                 max_num_bits=32,
                 ignore_zero=False):

        self.__retina_length = retina_length
        self.__class_names = class_names
        self.__regularization_lambda = regularization_lambda
        self.__bagging_percentage = bagging_percentage
        self.__min_num_bits = min_num_bits
        self.__max_num_bits = max_num_bits
        self.__ignore_zero = ignore_zero
        self.__epsilon = 0.02  # used to compare objective values
        self.__wisards = []

    def fit(self, X, y):

        test_size = X.shape[0]
        num_samples = int(test_size * self.bagging_percentage)

        indexes = np.arange(test_size)  # generating all posible positions

        loss_value = 1  # max possible errors (100%)
        regularization_value = 0  # there is no WiSARDs

        objective_value = self.__calc_objective_value(loss_value, regularization_value)
        previos_objective_value = objective_value + 1  # is bigger to running the first interation

        # minimizing objective value
        while (previos_objective_value - objective_value) > self.__epsilon:

            # random number of bits for addr in wisard
            num_bits_addr = np.random.randint(self.__min_num_bits, self.__max_num_bits)

            # creating a WiSARD
            w = Wisard(retina_length=self.__retina_length,
                       num_bits_addr=num_bits_addr,
                       ignore_zero_addr=self.__ignore_zero,
                       bleaching=False,
                       memory_is_cumulative=False)

            # creating discriminators
            for class_name in self.__class_names:
                w.create_discriminator(class_name)

            # sampling positions
            np.random.shuffle(indexes)  # shuffle indexes to get diferent positions
            selected_positions = indexes[0:num_samples]
            X_test = [X[i] for i in selected_positions]
            y_test = [y[i] for i in selected_positions]

            # training
            training_size = int(len(X_test) * 0.8)
            test_size = int(len(X_test) * 0.2)
            w.fit(X_test[0: training_size], y_test[0: training_size])
            self.__wisards.append(w)

            # testing
            count_hits = 0
            for i in xrange(training_size, training_size + test_size):
                X_i = X_test[i]
                y_i = y_test[i]

                selected_class = self.predict(X_i)
                if selected_class == y_i:
                    count_hits += 1

            # calculating objective value
            loss_value = 1 - (count_hits / float(test_size))  # loss value is not square error, check after if results won't be good
            regularization_value += num_bits_addr

            previos_objective_value = objective_value
            objective_value = self.__calc_objective_value(loss_value, regularization_value)

    def predict(self, x):
        results = {}

        for w in self.__wisards:

            partial_result = w.predict(x)
            clazz, confidence = self.__calc_result_with_confidence(partial_result)

            if clazz in results:
                results[clazz] += confidence
            else:
                results[clazz] = confidence

        selected_class = max(results, key=results.get)
        return selected_class

    def __calc_result_with_confidence(self, results):

        values = np.array(results.values())

        # getting max value
        max_value = values.max()
        if(max_value == 0):
            return 0

        # getting second max value
        second_max = max_value
        if values[values < max_value].size > 0:
            second_max = values[values < max_value].max()

        # calculating confidence value
        c = 1 - float(second_max) / float(max_value)

        # getting the max element
        selected_class = max(result, key=result.get)

        return selected_class, c

    def __calc_objective_value(self, loss_value, regularization_value):
        return loss_value + self.__regularization_lambda * regularization_value
