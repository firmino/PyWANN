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
        
        # ignore zero is for cases where 0 addr are not important (is a parameter in the WiSARD)
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
                 num_memo_to_combine,
                 list_conv_matrix,
                 memories_values_cummulative=True,
                 ignore_zero_addr=False):

        self.__num_bits = num_bits
        self.__list_conv_matrix = np.array(list_conv_matrix)
        self.__num_conv_matrix = self.__list_conv_matrix.shape[0]

        self.__conv_matrix_width  = len(self.__list_conv_matrix[0])
        self.__conv_matrix_height = len(self.__list_conv_matrix[0][0])

        # filtered retina will have adicional lines and columns due to convolutional operations
        self.__retina_width_filtered  = 28#retina_width + self.__conv_matrix_width - 1
        self.__retina_height_filtered = 28#retina_height + self.__conv_matrix_height -1
        self.__retina_size = self.__retina_width_filtered * self.__retina_height_filtered


        self.__num_mem =  self.__retina_size // self.__num_bits

        # if retina size is not a multiple of number of bits
        has_rest_memory = False
        number_of_rest_bits = self.__retina_size % self.__num_bits
        if number_of_rest_bits != 0:
            has_rest_memory = True

        self.__conv_mapping  = []
        self.__conv_memories = []

        #---------------------------------------------------------------------------------#
        # generating aleatory mapping 
        mapping = range(0,  self.__retina_size)
        np.random.shuffle(mapping)
        self.__conv_mapping = mapping

        # for each convoluted retina, get self.__num_bits.
        # combine the self.__num_bits of each retina to compose the address in a memory
        total_num_bits_memories = self.__num_bits * len(self.__list_conv_matrix)

        # creating memories
        # generating void memories for each combination
        self.__conv_memories = [Memory(total_num_bits_memories, memories_values_cummulative, ignore_zero_addr) 
                                for x in range (0, self.__retina_size, self.__num_bits)]

        # create a small memory to get all positions
        if has_rest_memory:
            address_lenght = number_of_rest_bits * len(self.__list_conv_matrix)
            rest_memory = Memory( address_lenght, memories_values_cummulative, ignore_zero_addr) 
            self.__conv_memories.append(rest_memory)
        #---------------------------------------------------------------------------------#

    def add_trainning(self, retina):

        # trainning convoluted layer
        linearized_retinas = []
        for conv_matrix_index  in range(len(self.__list_conv_matrix)):

            conv_matrix = self.__list_conv_matrix[conv_matrix_index]
            filtered_retina = np.array(self.__conv_img(retina, conv_matrix))  # applying convolution
            linearized_retina = filtered_retina.reshape(1, self.__retina_size)[0].tolist()

            linearized_retinas.append(linearized_retina)

        # for each memory, get the mapped positions, the values in the diferent retinas and store
        memory_position = 0
        for posi_index in range(0, self.__retina_size, self.__num_bits):

            # the number of  positions is related with the num_bits used.
            # num_bits is related for each conv, so the final addrs results is eq num_bits * num_memory
            positions =  self.__conv_mapping[posi_index:  posi_index + self.__num_bits] 

            addr = []
            for lin_retina in linearized_retinas:

                for posi in positions:
                    addr.append(lin_retina[posi])

            self.__conv_memories[memory_position].add_value(addr)
            memory_position += 1 # go to next memory

        
        number_of_rest_bits = self.__retina_size % self.__num_bits
        if number_of_rest_bits != 0:
            # the rest of the positions
            positions =  self.__conv_mapping[-1 * number_of_rest_bits : ] 
            addr = []
            for lin_retina in linearized_retinas:
                for position in positions:
                    addr.append(lin_retina[position])

            self.__conv_memories[memory_position].add_value(addr)
        


    def classify(self, retina, bleaching):

        result = 0

        # trainning convoluted layer
        linearized_retinas = []
        for conv_matrix_index  in range(len(self.__list_conv_matrix)):

            conv_matrix = self.__list_conv_matrix[conv_matrix_index]
            filtered_retina = np.array(self.__conv_img(retina, conv_matrix))  # applying convolution
            linearized_retina = filtered_retina.reshape(1, self.__retina_size)[0].tolist()

            linearized_retinas.append(linearized_retina)

        # for each memory, get the mapped positions, the values in the diferent retinas and store
        memory_position = 0
        for posi_index in range(0, self.__retina_size, self.__num_bits):

            # the number of  positions is related with the num_bits used.
            # num_bits is related for each conv, so the final addrs results is eq num_bits * num_memory
            positions =  self.__conv_mapping[posi_index:  posi_index + self.__num_bits] 
            addr = []
            for lin_retina in linearized_retinas:
                for position in positions:
                    addr.append(lin_retina[position])

            if self.__conv_memories[memory_position].get_value(addr) > bleaching:
                result += 1
            memory_position += 1 # go to next memory


        
        # if there is rest of positions
        number_of_rest_bits = self.__retina_size % self.__num_bits
        if number_of_rest_bits != 0:
            # the rest of the positions
            positions =  self.__conv_mapping[-1 * number_of_rest_bits : ] 
            addr = []
            for lin_retina in linearized_retinas:
                for position in positions:
                    addr.append(lin_retina[position])

            if self.__conv_memories[memory_position].get_value(addr) > bleaching:
                result += 1

        return result

    def __conv_img(self, img, conv):#, img, conv)
        
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
                 num_memo_to_combine,
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
                                                     list_conv_matrix = self.__list_conv_matrix,
                                                     num_memo_to_combine = 2,)



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
