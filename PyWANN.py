# -*- coding: utf-8 -*-
import numpy as np
import random as rand


class Retina:

    def __init__(self, data):

        if type(data) is not list:
            raise Exception('data should be a multidimensional list')
        if len(data) == 0:
            raise Exception('data in the list can not be void')

        # converting matrix to a list of elements
        raw_data = np.matrix(data)
        number_of_elements = 1
        for i in range(len(raw_data.shape)):
            number_of_elements *= raw_data.shape[i]  # multiply each dimension

        # retina's data is a list of elements
        self.__data = raw_data.reshape(1, number_of_elements).tolist()[0]

        # original dimensions of retina
        self.__shape = raw_data.shape

    def get_data(self):
        return self.__data

    def get_original_retina(self):
        data = np.matrix(self.__data)
        return data.reshape(self.__shape).tolist()


class Memory:

    def __init__(self, num_bits=2, is_cummulative=False):
        self.__data = {}
        self.__is_cummulative = is_cummulative

        # number of address positions 2 bits
        for i in range(2**num_bits):
            self.__data[i] = 0

    def get_memory_data(self):
        return self.__data

    def add_value(self, addr, value=1):

        if type(addr) != list:
            raise Exception("address' type is not a list")

        if len(addr) > len(self.__data):
            raise Exception('number of the bits of address are bigger \
                             than max size')

        int_position = self.__list_to_int(addr)

        if self.__is_cummulative:
            self.__data[int_position] += value
        else:
            self.__data[int_position] = value

    def get_value(self, addr):
        int_position = self.__list_to_int(addr)
        return self.__data[int_position]

    def __list_to_int(self, addr_list):
        reverse_list = addr_list[-1::-1]
        return sum([(2**i*reverse_list[i]) for i in range(len(reverse_list))])


class Discriminator:

    def __init__(self, retina_length, num_bits_addr=2,
                 memories_values_cummulative=False,
                 randomize_positions=True):

        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr

        num_memories = retina_length // num_bits_addr

        # creating list of memories
        self.__memories = {}
        for i in range(num_memories):
            self.__memories[i] = Memory(self.__num_bits_addr,
                                        memories_values_cummulative)


        # mapping random positions (if randomize_positions is True)
        self.__memories_mapping = {}

        # generating all possible positions
        position_list = range(retina_length)

        if randomize_positions: # random positions to mapping aleatory
            rand.shuffle(position_list)

        for i in range(num_memories):
            init = i * num_bits_addr
            end = init + num_bits_addr
            self.__memories_mapping[i] = position_list[init:end]


        #if the retina's length is not a multiple of number of bits of
        #addressing, is necessary create a smaller memory to map all positions.
        #this memory will have the number of address bits equal to module of
        #number of memories and number of bits of addressing
        self.__num_bits_addr_final = retina_length % num_bits_addr
        if self.__num_bits_addr_final > 0:

            #adding in the last position of the list (position == num_memories)
            self.__memories[num_memories] = Memory(self.__num_bits_addr_final,
                                                   memories_values_cummulative)

            #getting the last positions to mapping, how they are in the end of
            #the randomized position list, we are using negative index of python
            position_map = -1 * self.__num_bits_addr_final
            self.__memories_mapping[num_memories] = position_list[position_map:]

    def get_memories(self):
        return self.__memories

    def get_memories_mapping(self):
        return self.__memories_mapping

    def get_memory(self, index):
        return self.__memories[index]

    def get_mapping(self, index):
        return self.__memories_mapping[index]

    def training(self, list_positive_retina):
        for retina in list_positive_retina:  # for each element of training

            # for each mapping position in retina, each position of n bits
            # correspond an only one memory
            for memory_key in self.__memories_mapping:

                addr_list = []

                # get the mapping positions (size is equal of number of address
                # in the memory)
                position_list = self.__memories_mapping[memory_key]

                # for each position mapped get binary value (1 if position has
                # value positive and 0 otherwise)
                for position in position_list:
                    if retina.get_data()[position] > 0:
                        addr_list.append(1)
                    else:
                        addr_list.append(0)

                # add value 1 into the positon (defined by addr_list)
                self.__memories[memory_key].add_value(addr_list, 1)

    def classifier(self, retina):
        result = 0

        # for each mapping position in retina, each position of n bits
        # correspond an only one memory
        for memory_key in self.__memories_mapping:
            addr_list = []

            # get the mapping positions (size is equal of number of address
            # in the memory)
            position_list = self.__memories_mapping[memory_key]

            for position in position_list:
                    addr_list.append(1)
                else:
                    addr_list.append(0)
                if retina.get_data()[position] > 0:

            result += self.__memories[memory_key].get_value(addr_list)

        return result

    def get_drasiw(self):
        pass


class Wisard:

    def __init__(self, num_bits_addr=2, bleaching=None, is_cumulative=False ):
        self.__num_bits_addr = num_bits_addr
        self.__is_cumulative = is_cumulative
        self.__bleaching_method = bleaching
        self.__discriminators = {}


    def add_discrimator(self, name, training_set):

        #transform training_set(multidimensional matrix) to type Retina
        retina_training_set = []
        for element in training_set:
            retina_training_set.append(Retina(element))

        #getting the first retina to know the retina's size
        retina_size = len(retina_training_set[0].get_data())

        #creating discriminator
        self.__discriminators[name] = Discriminator(retina_size,
                                                    self.__num_bits_addr,
                                                    self.__is_cumulative)

        #training discriminator
        self.__discriminators[name].training(retina_training_set)


    def classifier(example):
        result = {}

        #become example a retina
        r = Retina(example)

        for class_name in self.__discriminators:
            result[class_name] = self.__discriminators[class_name].classifier(r)

        if self.__bleaching_method is not None:
            result = self.__bleaching_method.apply(result)

        return result


class Bleaching:

    def __init__(self):
        pass
