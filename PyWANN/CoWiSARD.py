# -*- coding: utf-8 -*-
import random as rand
import math

import numpy as np
import scipy.signal as signal


class Memory:

    def __init__(self, num_bits=2, is_cummulative=False, ignore_zero_addr=False):
        self.__data = {}
        self.__num_bits = num_bits
        self.__is_cummulative = is_cummulative
        self.__ignore_zero_addr = ignore_zero_addr

    def get_memory_size(self):  
        return 2**self.__num_bits

    def add_value(self, addr, value=1):

        if type(addr) != list:
            raise Exception("address' type is not a list")

        if len(addr) > self.__num_bits:
            raise Exception('number of the bits of address are bigger \
                             than max size')

        # bit list to int
        int_position = self.__list_to_int(addr)

        if self.__is_cummulative:
            if int_position in self.__data:
                self.__data[int_position] += value
            else:
                self.__data[int_position] = value
        else:
            self.__data[int_position] = value

    def get_value(self, addr):
        int_position = self.__list_to_int(addr)
        
        # ignore zero is for cases where 0 addr are not important 
        # (is a parameter in the WiSARD)
        if self.__ignore_zero_addr and int_position == 0:
            return 0

        if int_position not in self.__data:
            return 0
        else:
            return self.__data[int_position]

    def __list_to_int(self, addr_list):
        reverse_list = addr_list[-1::-1]
        return sum([(2**i*reverse_list[i]) for i in range(len(reverse_list))])


class Discriminator:

    def __init__(self,
                 retina_width,
                 retina_height,
                 num_bits, 
                 list_conv_matrix,
                 memories_values_cummulative=True,
                 ignore_zero_addr=False):

        self.__num_bits = num_bits
        self.__list_conv_matrix = np.array(list_conv_matrix)
        self.__num_conv_matrix = self.__list_conv_matrix.shape[0]

        self.__retina_width  = retina_width
        self.__retina_height = retina_height

        self.__retina_size = self.__retina_width * self.__retina_height
        self.__num_mem =  self.__retina_size // self.__num_bits

        # if retina size is not a multiple of number of bits
        self.__has_rest_memory = False
        self.__number_of_rest_bits = self.__retina_size % self.__num_bits
        if self.__number_of_rest_bits != 0:
            self.__has_rest_memory = True

        self.__mapping  = []
        self.__memories = []

        #---------------------------------------------------------------------------------#
        # generating aleatory mapping 
        mapping = range(0,  self.__retina_size)
        np.random.shuffle(mapping)
        self.__mapping = mapping

        # creating memories
        # generating void memories
        self.__memories = [Memory(self.__num_bits, memories_values_cummulative, ignore_zero_addr) 
                           for x in range (0, self.__num_mem)]

        # create a small memory to get all positions
        if self.__has_rest_memory:
            rest_memory = Memory(self.__number_of_rest_bits, memories_values_cummulative, ignore_zero_addr) 
            self.__memories.append(rest_memory)

        #---------------------------------------------------------------------------------#

    def add_trainning(self, retina):

        #apply convolutions over images
        linearized_retina = self.__process_retina(retina)
        
        # for each memory
        for memory_position in range(0, self.__num_mem):
            mapping_position = memory_position * self.__num_bits
            positions =  self.__mapping[mapping_position : mapping_position + self.__num_bits] 
            
            addr = []
            for posi in positions:
                addr.append(linearized_retina[posi])

            self.__memories[memory_position].add_value(addr)

        if self.__has_rest_memory:
            positions =  self.__mapping[ (-1* self.__number_of_rest_bits) : ] 
            addr = []

            for posi in positions:
                addr.append(linearized_retina[posi])

            self.__memories[memory_position].add_value(addr)        

    def classify(self, retina, bleaching):

        result = 0
        #apply convolutions over images
        linearized_retina = self.__process_retina(retina)
        
        # for each memory
        for memory_position in range(0, self.__num_mem):
            mapping_position = memory_position * self.__num_bits
            positions =  self.__mapping[mapping_position : mapping_position + self.__num_bits] 
            
            addr = []
            for posi in positions:
                addr.append(linearized_retina[posi])
            
            value = self.__memories[memory_position].get_value(addr)
            if value >= bleaching:
                result += 1

        if self.__has_rest_memory:
            positions =  self.__mapping[ (-1* self.__number_of_rest_bits) : ] 
            
            addr = []
            for posi in positions:
                addr.append(linearized_retina[posi])
            
            value = self.__memories[memory_position].get_value(addr)
            if value >= bleaching:
                result += 1

        return result

    def __process_retina(self, retina):
        
        processed_image = np.zeros((self.__retina_width , self.__retina_height))

        for conv_matrix_index  in range(len(self.__list_conv_matrix)):
            conv_matrix = self.__list_conv_matrix[conv_matrix_index]
            filtered_retina = np.array(self.__conv_img(retina, conv_matrix))  # applying convolution
            processed_image += filtered_retina
        
        # uniform convoluted layer
        np.place(processed_image, processed_image <= 1, 0)
        np.place(processed_image, processed_image > 1, 1)

        # retina is a list (a matrix is tranformed into a list)
        linearized_retina = processed_image.reshape(1, self.__retina_size)[0].tolist()

        return linearized_retina

    def __conv_img(self, img, conv):
        
        origin = np.array(img)
        img_filter = np.array(conv)

        result = signal.convolve2d(origin, img_filter, boundary='symm', mode='same')
        np.place(result, result < 0, 0)
        np.place(result, result > 1, 1)
    
        return result.tolist()
        

class CoWiSARD:

    def __init__(self, 
                 retina_width,
                 retina_height,
                 num_bits, 
                 list_conv_matrix,
                 confidence_threshold = 0.1,
                 default_bleaching_b_value = 3,
                 memories_values_cummulative=True,
                 ignore_zero_addr=False):

        self.__retina_width = retina_width
        self.__retina_height = retina_height
        self.__num_bits = num_bits
        self.__list_conv_matrix = list_conv_matrix

        self.__confidence_threshold = confidence_threshold
        self.__bleaching_b_value = default_bleaching_b_value
    
        self.__discriminators = {}
        self.__confidence_threshold = confidence_threshold

        self.__ignore_zero_addr = ignore_zero_addr    
        
        
    def create_discriminator(self, name):

        # creating discriminator
        self.__discriminators[name] =  Discriminator(retina_width=self.__retina_width,
                                                     retina_height=self.__retina_height,
                                                     num_bits = self.__num_bits,
                                                     list_conv_matrix = self.__list_conv_matrix)

    # add a example to training in an especific discriminator
    # def add_training(self, disc_name, training_example):
    def add_trainning(self, disc_name, training_example):
        self.__discriminators[disc_name].add_trainning(training_example)

    def classify(self, example):
        result = {}  # classes and values

        confidence = 0
        increment_b = 0

        while confidence < self.__confidence_threshold:

            for class_name in self.__discriminators:
                b = self.__bleaching_b_value + increment_b
                res = self.__discriminators[class_name].classify(example, b)
                result[class_name] = res

            confidence = Util.calc_confidence(result)
            increment_b += 1

        return result


class Util:
    @staticmethod
    def calc_confidence(list_of_results):

        try:
            values = list_of_results.values()

            # getting max value
            max_value = max(values)

            # removing max from the list
            values.remove(max_value)

            # getting second max value
            second_max = max(values)

            # calculating confidence value
            c = 1 - float(second_max)**4 / float(max_value)**4

            return c

        except Exception, Error:
            return -1
