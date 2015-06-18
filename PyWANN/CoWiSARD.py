# -*- coding: utf-8 -*-
import random as rand
import math


class Memory:

    def __init__(self):
        self.__memory = {}

    def add_value(self, addr):
        if addr not in self.__memory:
            self.__memory[addr] = 1
        else:
            self.__memory[addr] += 1

    def get_value(self, addr):

        if addr not in self.__memory:
            return 0

        return self.__memory[addr]

    def __str__(self):
        return str(self.__memory)


class Discriminator:

    def __init__(self,
                 retina_width, 
                 retina_height, 
                 list_conv_matrix):

        self.__threshold = 0.9

        self.__memories = {}
        self.__conv_matrix = list_conv_matrix

        self.__retina_width = retina_width
        self.__retina_height = retina_height 

        self.__convolutional_matrix_width = len(list_conv_matrix[0])
        self.__convolutional_matrix_height = len(list_conv_matrix[0][0])
        self.__convolution_matrix_size = self.__convolutional_matrix_width * self.__convolutional_matrix_height

        for memo_index in range(len(self.__conv_matrix)):
            self.__memories[memo_index] = Memory()

    def add_train(self, retina):        

        for conv_index in range(len(self.__conv_matrix)):
            mem_addr = 0    
            matrix_conv = self.__conv_matrix[conv_index]

            for i in range(0, self.__retina_width - self.__convolutional_matrix_width  ):
                for j in range(0,  self.__retina_height -  self.__convolutional_matrix_height):
                    overlap = self.__calculate_superposition(conv_index, retina, i, j)
                    rating = float(overlap) / self.__convolution_matrix_size

                    if rating >= self.__threshold:
                        mem_addr += 1

            self.__memories[conv_index].add_value(mem_addr)


    def classify(self, retina):
        result = 0
        
        for conv_index in range(len(self.__conv_matrix)):
            mem_addr = 0    
            matrix_conv = self.__conv_matrix[conv_index]

            for i in range(0, self.__retina_width - self.__convolutional_matrix_width  ):
                for j in range(0,  self.__retina_height -  self.__convolutional_matrix_height):
                    overlap = self.__calculate_superposition(conv_index, retina, i, j)
                    rating = float(overlap) / self.__convolution_matrix_size

                    if rating >= self.__threshold:
                        mem_addr += 1

            result += self.__memories[conv_index].get_value(mem_addr)

        return result

    def get_memories(self):
        return self.__memories

    def __calculate_superposition(self, conv_index, retina, pos_x, pos_y):
        result = 0

        if len(self.__conv_matrix) == 0:
            raise Exception('Convolutional Matrix is not defined')

        matrix = self.__conv_matrix[conv_index]

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == retina[pos_x + i ][pos_y + j]:
                    result += 1

        return result


class CoWiSARD:

    def __init__(self, 
                 retina_width,
                 retina_height,
                 conv_width,
                 conv_height):

        self.__retina_width  = retina_width
        self.__retina_height = retina_height
        self.__convolutional_matrix_width = conv_width
        self.__convolutional_matrix_height = conv_height
        self.__convolution_matrix = []
        self.__discriminators = {}

    def add_conv_matrix(self, matrix):
        self.__convolution_matrix.append(matrix)
    
    def create_discriminator(self, name):

        self.__discriminators[name] = Discriminator(self.__retina_width,
                                                    self.__retina_height,                                                
                                                    self.__convolution_matrix)

        
    def train_discriminator(self, name, retina):
        return self.__discriminators[name].add_train(retina)

    def classify(self, retina):
        result = {}

        for disc_name in self.__discriminators:
            result[disc_name] = self.__discriminators[disc_name].classify(retina)

        return result