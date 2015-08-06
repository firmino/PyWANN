# -*- coding: utf-8 -*-
import random as rand
import math


class Discriminator:

    def __init__(self,
                 retina_width, 
                 retina_height, 
                 list_conv_matrix,
                 conv_box,
                 memories_values_cummulative = True):

        self.__threshold_overlap = 0.9
        self.__conv_matrices = list_conv_matrix

        self.__retina_width = retina_width
        self.__retina_height = retina_height 

        self.__conv_matrix_width = len(list_conv_matrix[0])
        self.__conv_matrix_height = len(list_conv_matrix[0][0])

        self.__quadrant_width = conv_box[0]
        self.__quadrant_height = conv_box[1]

        self.__num_quadrant_x = self.__retina_width / conv_box[0]
        self.__num_quadrant_y = self.__retina_height / conv_box[1]
        self.__num_tot_quadrat = self.__num_quadrant_y * self.__num_quadrant_y

        self.__memories_values_cummulative = memories_values_cummulative

        
        # for each invariant a line in the memory table with the presence 
        # or absence (0 or 1) in each quadrant 
        self.__conv_memory = []
        for i in range( self.__num_tot_quadrat ):
            # each quadrant has a memory addressed by a vector represented by 
            # the set of convolutional matrices.
            # ex: supose there are 5 conv matrices, so, each quadrant has a memory
            #     with 32 position (2** number of conv matrix). The position 0 
            #     ([0,0,0,0,0])represents the absence of all matrices, 
            #     position 31 [1,1,1,1,1] represents de presence of all matrices
            linha = [0] * (2**len (self.__conv_matrices))
            self.__conv_memory.append(linha)

    def add_train(self, retina):

        # each quadrand (line to column)
        quadrant = 0
        for num_column in range(0, self.__num_quadrant_x): # represents the line of a retina
            for num_line in range(0, self.__num_quadrant_y): # represents the column of a retina

                x_begin = num_column * self.__quadrant_width 
                x_end = x_begin + self.__quadrant_width

                y_begin = num_line * self.__quadrant_height
                y_end = y_begin + self.__quadrant_height

                addr = []
                for conv_matrix in self.__conv_matrices:

                    if self.__has_overlap(conv_matrix, retina, x_begin, x_end, y_begin, y_end):
                        addr.append(1)
                    else:
                        addr.append(0)

                memory_position = self.__list_to_int(addr)

                if self.__memories_values_cummulative:
                    self.__conv_memory[quadrant][memory_position] += 1
                else:
                    self.__conv_memory[quadrant][memory_position]  = 1

                quadrant += 1 

    def classify(self, retina):
        pass

    def __matching_memories(self, table_result):
        pass

    def __list_to_int(self, addr_list):
        reverse_list = addr_list[-1::-1]
        return sum([(2**i*reverse_list[i]) for i in range(len(reverse_list))])

    # lookup into a box for the presence of a conv_matrix, if the box presents the conv_matrix with a high similarity 
    # (percentage_overlap >= self.__threshold_overlap), the function returns True
    def __has_overlap(self, conv_matrix, retina, x_begin, x_end, y_begin, y_end):
        
        size_conv = len(conv_matrix) * len(conv_matrix[0])

        for x in range(x_begin, x_end - self.__conv_matrix_width+1):  # for each possible position in the quadrant
            for y in range(y_begin, y_end - self.__conv_matrix_height+1):  # for each possible position in the quadrant
                if self.__percent_overlap(conv_matrix, retina, x, y) >= self.__threshold_overlap:
                    return True

        return False

    # return the percentage of overlap between conv_matrix and a specific position in a bon
    def __percent_overlap(self, conv_matrix, retina, posi_ini_x, posi_ini_y):
        conv_size = len(conv_matrix) * len(conv_matrix[0]) 
        cont = self.__count_overlap(conv_matrix, retina, posi_ini_x, posi_ini_y)
        return float(cont) / conv_size

    # count the number of overlaps between conv_matrix and a specific position in a box
    def __count_overlap (self, conv_matrix, retina, posi_ini_x, posi_ini_y):  
        cont = 0
        for x in range(0, self.__conv_matrix_width):
            for y in range(0, self.__conv_matrix_height):
                if conv_matrix[x][y] == retina[x + posi_ini_x][y + posi_ini_y]:
                    cont += 1
        return cont


class CoWiSARD:

    def __init__(self, 
                 retina_width,
                 retina_height,
                 list_conv_matrix,
                 conv_box):

        self.__retina_width  = retina_width
        self.__retina_height = retina_height
        self.__list_conv_matrix = list_conv_matrix
        self.__conv_box = conv_box

        self.__discriminators = {}


    def create_discriminator(self, name):

        self.__discriminators[name] = Discriminator(self.__retina_width,
                                                    self.__retina_height,                                                
                                                    self.__list_conv_matrix,
                                                    self.__conv_box)
        
    def train_discriminator(self, name, retina):
        return self.__discriminators[name].add_train(retina)

    def classify(self, retina):
        result = {}

        for disc_name in self.__discriminators:
            result[disc_name] = self.__discriminators[disc_name].classify(retina)
            

        return result