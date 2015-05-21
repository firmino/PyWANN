# -*- coding: utf-8 -*-
import random as rand


class Retina:

    def __init__(self, data):

        self.__data = None

        if type(data) is not list:
            raise Exception('data must be a multidimensional list')

        if len(data) == 0:
            raise Exception('data in the list can not be void')

        if isinstance(data[0][0], list):  # check if data is bigger than 2
            raise Exception('data must be 1-dimensional or 2-dimensional')

        # check if data is 2-dimensional
        if isinstance(data[0], list):
            aux = []
            # converting matrix to a list of elements
            for i in range(len(data)):  # for each line
                for j in range(len(data[0])):  # for each column
                    aux.append(data[i][j])
            self.__data = aux

        # if is data is 1-dimensional
        else:
            self.__data = data

    def get_data(self):
        return self.__data


class Memory:

    def __init__(self, num_bits=2, is_cummulative=False):
        self.__data = {}
        self.__is_cummulative = is_cummulative

        # number of address positions 2 bits
        for i in xrange(2**num_bits):
            self.__data[i] = 0

    def get_memory_data(self):
        return self.__data

    def add_value(self, addr, value=1):

        if type(addr) != list:
            raise Exception("address' type is not a list")

        if len(addr) > len(self.__data):
            raise Exception('number of the bits of address are bigger \
                             than max size')

        # bit list to int
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

    def __init__(self,
                 retina_length,
                 num_bits_addr,
                 position_list,
                 memories_values_cummulative=False):

        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr
        self.__memories = {}
        self.__memories_mapping = {}

        # calculating the number of memories
        num_mem = retina_length // num_bits_addr

        # creating list of memories
        for i in xrange(num_mem):
            self.__memories[i] = Memory(self.__num_bits_addr,
                                        memories_values_cummulative)

        # mapping positions for each memory
        for i in xrange(num_mem):
            init = i * num_bits_addr
            end = init + num_bits_addr
            self.__memories_mapping[i] = position_list[init:end]

        # if the retina's length is not a multiple of number of bits of
        # addressing, it is necessary create a smaller memory to map all
        # positions. This memory will have the number of address bits equal to
        # module of number of memories for number of bits of addressing
        self.__num_bits_addr_final = retina_length % num_bits_addr
        if self.__num_bits_addr_final > 0:
            # adding in the last position of the list (position equal
            # num_mem)
            self.__memories[num_mem] = Memory(self.__num_bits_addr_final,
                                              memories_values_cummulative)

            # getting the last positions to mapping, how they are in the end of
            # the randomized position list, we are using negative
            # index of python
            position_map = -1 * self.__num_bits_addr_final
            self.__memories_mapping[num_mem] = position_list[position_map:]

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

            # for each mapping position in retina, each position has n bits
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
        result = []

        # for each mapping position in retina, each position of n bits
        # correspond an only one memory
        for memory_key in self.__memories_mapping:
            addr_list = []

            # get the mapping positions (size is equal of number of address
            # in the memory)
            position_list = self.__memories_mapping[memory_key]

            for position in position_list:
                if retina.get_data()[position] > 0:
                    addr_list.append(1)
                else:
                    addr_list.append(0)

            result.append(self.__memories[memory_key].get_value(addr_list))

        return result

    def get_drasiw(self):
        pass


class Wisard:

    def __init__(self, num_bits_addr=2,
                 bleaching=None,
                 confidence_threshold=0.3,
                 is_cumulative=False,
                 randomize_positions=True):

        self.__num_bits_addr = num_bits_addr
        self.__confidence_threshold = confidence_threshold
        self.__is_cumulative = is_cumulative
        self.__bleaching = bleaching
        self.__discriminators = {}
        self.__position_list = None
        self.__randomize_positions = randomize_positions

    def add_discriminator(self, name, training_set):

        # transform training_set(multidimensional matrix) to type Retina
        retina_training_set = []
        for element in training_set:
            retina_training_set.append(Retina(element))

        # getting the first retina to know the retina's size
        retina_size = len(retina_training_set[0].get_data())

        # if there is not a mapping position defined
        if self.__position_list is None:
            self.__generate_position_list(retina_size)

        # creating discriminator
        self.__discriminators[name] = Discriminator(retina_size,
                                                    self.__num_bits_addr,
                                                    self.__position_list,
                                                    self.__is_cumulative)

        # training discriminator
        self.__discriminators[name].training(retina_training_set)

    def classifier(self, example):
        result = {}  # classes and values
        memory_result = {}  # for each class the memories values obtained

        # transform example into a retina
        r = Retina(example)

        for class_name in self.__discriminators:

            # for each class the memorie values obtained
            memory_result[class_name] = self.__discriminators[class_name] \
                                            .classifier(r)

            # for each class, store the value
            result[class_name] = sum(memory_result[class_name])

        # applying bleaching method if exist
        if self.__bleaching is not None:
            # calculate the confidence for  results
            cfd = self.__calc_result_confidence(result)

            # apply bleaching method for all memories values
            if cdf < self.__confidence_threshold:
                self.__bleaching.run(result, memory_result)

        return result

    def __generate_position_list(self, retina_length):

        # generating all possible positions
        position_list = range(retina_length)

        # mapping random positions (if randomize_positions is True)
        if self.__randomize_positions:  # random positions to mapping aleatory
            rand.shuffle(position_list)

        self.__position_list = position_list

    def __calc_result_confidence(self, list_of_results):

        # getting max value
        max_value = max(v)

        # getting second max value
        second_max = max(n for n in v if n != max_value)

        # calculating confidence value
        c = (max_value - second_max) / float(max_value)

        return c


class Bleaching:

    def __init__(self):
        pass
