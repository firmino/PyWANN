# -*- coding: utf-8 -*-
import random as rand
import math

import numpy as np
import scipy.signal as signal


class Discriminator:

    def __init__(self,
                 retina_width,
                 retina_height,
                 num_bits_first_layer, 
                 num_memo_to_combine,
                 list_conv_matrix,
                 memories_values_cummulative=True):


        self.__num_bits_first_layer = num_bits_first_layer
        self.__list_conv_matrix = np.array(list_conv_matrix)

        self.__conv_matrix_width  = self.__list_conv_matrix.shape[0]
        self.__conv_matrix_height = self.__list_conv_matrix.shape[0]

        # filtered retina will have adicional lines and columns due to convolutional operations
        self.__retina_width_filtered = retina_width + self.__conv_matrix_width - 1
        self.__retina_height_filtered = retina_height + self.__conv_matrix_height -1

        retina_size = self.__conv_matrix_width * self.__conv_matrix_height        
        num_mem = retina_size // self.__num_bits_first_layer

        # if retina size is not a multiple of number of bits
        has_rest_memory = False
        number_of_rest_bits = retina_size % self.__num_bits_first_layer
        if number_of_rest_bits != 0:
            has_rest_memory = True

        self.__conv_mapping  = []
        self.__conv_memories = []
        for i in xrange( len(self.__list_conv_matrix) ):
            # generating aleatory mapping for each filtered image
            mapping = range(0, retina_size)
            np.random.shuffle(mapping)
            self.__conv_mapping.append(mapping)
            
            # generating void memories for each filtered image
            memories = np.zeros((num_mem, 2**num_bits_first_layer))

            # if the retine size is not a multiple of number of addrs bits
            # create a small memory to get all positions
            if has_rest_memory:
                rest_memory = np.zeros(2**number_of_rest_bits)
                memories =  np.vstack ((memories, rest_memory))

            self.__conv_memories.append(memories)

        # generating a mapping combination of memories
        self.__combined_memories_mapping = np.array(range(len(self.__conv_memories)))
        np.random.shuffle(self.__combined_memories_mapping)

        # generating memories to combining each filtered image's memory
        if self.__combined_memories_mapping.shape[0] % num_memo_to_combine != 0:
            raise Exception("list_conv_matrix must be multiple of numb_mem_comb ")

        num_memories_combined = self.__combined_memories_mapping.shape[0] / num_memo_to_combine
        self.__combined_memories = np.zeros( (num_memories_combined, 2**num_memories_combined))

    '''
    def add_train(self, retina):
        return 1
        #for conv_matrix in self.__list_conv_matrix:
        #    filtered_retina = self.__conv_img(retina, conv_matrix)
        
    def classify(self, retina):
        return 1
    '''

    def __list_to_int(self, addr_list):
        reverse_list = addr_list[-1::-1]
        return sum([(2**i*reverse_list[i]) for i in range(len(reverse_list))])

    def __conv_img(self, img, conv):#, img, conv)
        
        origin = np.array(img)
        img_filter = np.array(conv)

        result = signal.convolve2d(origin, img_filter, boundary='symm', mode='full')
        np.place(result, result < 0, 0)
        np.place(result, result > 1, 1)
    
        return result.tolist()
        
'''
class CoWiSARD:

    def __init__(self, 
                 retina_width,
                 retina_height,
                 list_conv_matrix,
                 conv_box):
'''