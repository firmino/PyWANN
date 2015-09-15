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
                 num_bits_first_layer, 
                 num_memo_to_combine,
                 list_conv_matrix,
                 memories_values_cummulative=True,
                 ignore_zero_addr=False):

        self.__num_bits_first_layer = num_bits_first_layer
        self.__list_conv_matrix = np.array(list_conv_matrix)
        self.__num_conv_matrix = self.__list_conv_matrix.shape[0]

        self.__conv_matrix_width  = len(self.__list_conv_matrix[0])
        self.__conv_matrix_height = len(self.__list_conv_matrix[0][0])

        # filtered retina will have adicional lines and columns due to convolutional operations
        self.__retina_width_filtered  = retina_width + self.__conv_matrix_width - 1
        self.__retina_height_filtered = retina_height + self.__conv_matrix_height -1
        self.__retina_size = self.__retina_width_filtered * self.__retina_height_filtered


        self.__num_mem =  self.__retina_size // self.__num_bits_first_layer

        # if retina size is not a multiple of number of bits
        has_rest_memory = False
        number_of_rest_bits = self.__retina_size % self.__num_bits_first_layer
        if number_of_rest_bits != 0:
            has_rest_memory = True

        self.__conv_mapping  = []
        self.__conv_memories = []

        #---------------------------------------------------------------------------------#
        for i in xrange( len(self.__list_conv_matrix) ):
            # generating aleatory mapping for each filtered image
            mapping = range(0,  self.__retina_size)
            np.random.shuffle(mapping)
            self.__conv_mapping.append(mapping)
            
            # generating void memories for each filtered image
            memories = [Memory(num_bits_first_layer, memories_values_cummulative, ignore_zero_addr) 
                        for x in range(self.__num_mem)]

            # if the retine size is not a multiple of number of addrs bits
            # create a small memory to get all positions
            if has_rest_memory:
                rest_memory = Memory(number_of_rest_bits, memories_values_cummulative, ignore_zero_addr) 
                memories.append(rest_memory)

            self.__conv_memories.append(memories)

        #---------------------------------------------------------------------------------#
        # generating memories to combining each filtered image's memory
        self.__combined_memories_mapping = [(conv,mem_posi) for conv in range(self.__num_conv_matrix) for mem_posi in range (self.__num_mem)]
        np.random.shuffle(self.__combined_memories_mapping)

        # for the numb of memories in second layer has the same number of bit addrs.
        # it is not a feature, just i don't have enough time to work with this
        if (len(self.__combined_memories_mapping) % num_memo_to_combine) != 0:
            raise Exception('num_memo_to_combine cannot divide len(self.__combined_memories_mapping)' )


        num_memories_combined = len(self.__combined_memories_mapping) / num_memo_to_combine

        self.__combined_memories = [ Memory(num_memories_combined, memories_values_cummulative,
                                     ignore_zero_addr) for x in  range(num_memo_to_combine) ]

        #---------------------------------------------------------------------------------#

    def add_trainning(self, retina):

        # trainning convoluted layer
        for conv_matrix_index  in range(len(self.__list_conv_matrix)):

            conv_matrix = self.__list_conv_matrix[conv_matrix_index]
            filtered_retina = np.array(self.__conv_img(retina, conv_matrix))  # applying convolution
            linearized_retina = filtered_retina.reshape(1, self.__retina_size)[0].tolist()

            # getting the positions for each memory
            # each memory has (self.__num_bits_first_layer) positions
            memories = self.__conv_memories[conv_matrix_index] # getting the memories related with conv_matrix
            conv_mapping = self.__conv_mapping[conv_matrix_index]


            # getting the shuffled positions stored in self.__conv_mapping
            for i in range(0, self.__retina_size, self.__num_bits_first_layer): 
                positions = conv_mapping[i : i + self.__num_bits_first_layer]
                
                # to each related position in the retine, getting if the value (one or zero)
                binary_addres = []
                for posi in positions:
                    binary_addres.append(linearized_retina[posi])

                # the number of memories is equal to (self.__retina_size/self.__num_bits_first_layer)
                # I'm getting the correspondent memory
                memory_position = i / self.__num_bits_first_layer
                memories[memory_position].add_value(binary_addres)


        # trainning combined memories (second layer)
        #for comb_memo_index in range(len(self.__combined_memories)):

    def classify(self, retina, bleaching):

        result = 0
         # trainning convoluted layer
        for conv_matrix_index  in range(len(self.__list_conv_matrix)):

            conv_matrix = self.__list_conv_matrix[conv_matrix_index]
            filtered_retina = np.array(self.__conv_img(retina, conv_matrix))  # applying convolution
            linearized_retina = filtered_retina.reshape(1, self.__retina_size)[0].tolist()

            # getting the positions for each memory
            # each memory has (self.__num_bits_first_layer) positions
            memories = self.__conv_memories[conv_matrix_index] # getting the memories related with conv_matrix
            conv_mapping = self.__conv_mapping[conv_matrix_index]

            # getting the shuffled positions stored in self.__conv_mapping
            for i in range(0, self.__retina_size, self.__num_bits_first_layer): 
                positions = conv_mapping[i : i + self.__num_bits_first_layer]
                
                # to each related position in the retine, getting if the value (one or zero)
                binary_addres = []
                for posi in positions:
                    binary_addres.append(linearized_retina[posi])

                # the number of memories is equal to (self.__retina_size/self.__num_bits_first_layer)
                # I'm getting the correspondent memory
                memory_position = i / self.__num_bits_first_layer

                if  memories[memory_position].get_value(binary_addres) > bleaching:
                    result += 1

        return result

    def __conv_img(self, img, conv):#, img, conv)
        
        origin = np.array(img)
        img_filter = np.array(conv)

        result = signal.convolve2d(origin, img_filter, boundary='symm', mode='full')
        np.place(result, result < 0, 0)
        np.place(result, result > 1, 1)
    
        return result.tolist()
        

class CoWiSARD:

    def __init__(self, 
                 retina_width,
                 retina_height,
                 num_bits_first_layer, 
                 list_conv_matrix,
                 num_memo_to_combine,
                 confidence_threshold = 0.1,
                 default_bleaching_b_value = 3,
                 memories_values_cummulative=True,
                 ignore_zero_addr=False):

        self.__retina_width = retina_width
        self.__retina_height = retina_height
        self.__num_bits_first_layer = num_bits_first_layer
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
                                                     num_bits_first_layer = self.__num_bits_first_layer,
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
