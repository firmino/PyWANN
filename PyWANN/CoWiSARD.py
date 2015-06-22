# -*- coding: utf-8 -*-
import random as rand
import math


class Discriminator:

    def __init__(self,
                 retina_width, 
                 retina_height, 
                 list_conv_matrix,
                 conv_box):


        self.__threshold_overlap = 0.9
        self.__mapping_table = []
        self.__conv_matrix = list_conv_matrix

        self.__retina_width = retina_width
        self.__retina_height = retina_height 

        self.__conv_matrix_width = len(list_conv_matrix[0])
        self.__conv_matrix_height = len(list_conv_matrix[0][0])


        self.__quadrant_width = conv_box[0]
        self.__quadrant_height = conv_box[1]

        self.__num_quadrant_x = self.__retina_width / conv_box[0]
        self.__num_quadrant_y = self.__retina_height / conv_box[1]

        self.__memory_table = []

        # for each invariant a line in the memory table with the presence 
        # or absence (0 or 1) in each quadrant 
        for i in range(len(list_conv_matrix)):
            linha = [0] * (self.__num_quadrant_x * self.__num_quadrant_y)  
            self.__memory_table.append(linha)

    def add_train(self, retina):        

        conv_addr = 0  # each invariant has a conv_addr (in the sequence of creation)
        for conv_matrix in self.__conv_matrix: 
            cont_column = 0  # each column represents a quadrant

            for num_column in range(0, self.__num_quadrant_x): # represents the line of a retina
                for num_line in range(0, self.__num_quadrant_y): # represents the column of a retina

                    x_begin = num_column * self.__quadrant_height 
                    x_end = x_begin + self.__quadrant_height

                    y_begin = num_line *  self.__quadrant_width
                    y_end = y_begin +  self.__quadrant_width

                    # check if there is a invariant (identified by conv_addr) in the especific
                    # region of the retina
                    if self.__has_overlap(conv_matrix, retina, x_begin, x_end, y_begin, y_end):
                        self.__memory_table[conv_addr][cont_column] += 1
                    
                    cont_column+= 1

            conv_addr += 1

    def classify(self, retina):

        matrix_result = []
        for i in range(len(self.__conv_matrix )):  # for each line
            linha = [0] * (self.__num_quadrant_x * self.__num_quadrant_y)
            matrix_result.append(linha)

        conv_addr = 0
        for conv_matrix in self.__conv_matrix:
            cont_column = 0
            for num_column in range(0, self.__num_quadrant_x):
                for num_line in range(0, self.__num_quadrant_y):

                    x_begin = num_column * self.__quadrant_height
                    x_end = x_begin + self.__quadrant_height

                    y_begin = num_line *  self.__quadrant_width
                    y_end = y_begin +  self.__quadrant_width

                    if self.__has_overlap(conv_matrix, retina, x_begin, x_end, y_begin, y_end):
                        matrix_result[conv_addr][cont_column] += 1
                    
                    cont_column += 1

            conv_addr += 1

        return self.__matching_memories(matrix_result)


    def __matching_memories(self, table_result):
        b = 5
        count = 0
        for i in range(len(table_result)):
            for j in range(len(table_result[0])):
                pass
                if self.__memory_table[i][j] > b and table_result[i][j] == 1:
                    count += 1

        return count
            
    def __has_overlap(self, conv_matrix, retina, x_begin, x_end, y_begin, y_end):
        
        size_conv = len(conv_matrix) * len(conv_matrix[0])

        for x in range(x_begin, x_end - self.__conv_matrix_width+1):  # for each possible position in the quadrant
            for y in range(y_begin, y_end - self.__conv_matrix_height+1):  # for each possible position in the quadrant
                if self.__percent_overlap(conv_matrix, retina, x, y) > self.__threshold_overlap:
                    return True

        return False

    def __percent_overlap(self, conv_matrix, retina, posi_x, posi_y):
        conv_size = len(conv_matrix) * len(conv_matrix[0]) 
        cont = self.__calculate_overlap(conv_matrix, retina, posi_x, posi_y)
        return float(cont) / conv_size

    def __calculate_overlap (self, conv_matrix, retina, posi_x, posi_y):
        cont = 0
        for x in range(0, self.__conv_matrix_width):
            for y in range(0, self.__conv_matrix_height):
                if conv_matrix[x][y] == retina[x + posi_x][y + posi_y]:
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