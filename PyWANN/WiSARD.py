# -*- coding: utf-8 -*-
import random as rand
import math

class Retina:

    def __init__(self, data):

        self.__data = None

        if type(data) is not list:
            raise Exception('data must be a multidimensional list')

        if len(data) == 0:
            raise Exception('data in the list can not be void')

        if isinstance(data[0], list) and \
           isinstance(data[0][0], list):  # check if data is bigger than 2

            raise Exception('data must be 1-dimensional or 2-dimensional')

        # check if data is 2-dimensional
        if isinstance(data[0], list):
            aux = []
            # converting matrix to a list of elements
            for i in range(len(data)):  # for each line
                for j in range(len(data[0])):  # for each column
                    value = 1 if data[i][j] > 0 else 0
                    aux.append(value)
            self.__data = aux

        # if is data is 1-dimensional
        else:
            self.__data = data

    def get_data(self):
        return self.__data


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
                 retina_length,
                 num_bits_addr,
                 position_list,
                 memories_values_cummulative=False,
                 ignore_zero_addr=False):

        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr
        self.__memories = {}
        self.__memories_mapping = {}

        # calculating the number of memories
        num_mem = retina_length // num_bits_addr

        # creating list of memories
        for i in xrange(num_mem):
            self.__memories[i] = Memory(self.__num_bits_addr,
                                        memories_values_cummulative,
                                        ignore_zero_addr)

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
                                              memories_values_cummulative,
                                              ignore_zero_addr)

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

    def train(self, positive_retina):

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
                if positive_retina.get_data()[position] > 0:
                    addr_list.append(1)
                else:
                    addr_list.append(0)

            # add value 1 into the positon (defined by addr_list)
            self.__memories[memory_key].add_value(addr_list, 1)

    def classify(self, retina):
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


class WiSARD:

    def __init__(self,
                 retina_size,
                 num_bits_addr=2,
                 vacuum=False,
                 bleaching=False,
                 confidence_threshold=0.6,
                 randomize_positions=True,
                 default_bleaching_b_value=3,
                 ignore_zero_addr=False):

        self.__num_bits_addr = num_bits_addr
        self.__retina_size = retina_size
        self.__confidence_threshold = confidence_threshold
        self.__is_cumulative = False

        self.__discriminators = {}
        self.__position_list = None
        self.__ignore_zero_addr = ignore_zero_addr
        self.__randomize_positions = randomize_positions
        self.__confidence_threshold = confidence_threshold

        self.__bleaching = None
        self.__vacuum = None

        if bleaching:
            self.__is_cumulative = True
            self.__bleaching = Bleaching(default_bleaching_b_value)

        if vacuum:
            self.__is_cumulative = True
            self.__vacuum = Vacuum()

    def create_discriminator(self, name):

        # if there is not a mapping position defined
        if self.__position_list is None:
            self.__generate_position_list(self.__retina_size)

        # creating discriminator
        self.__discriminators[name] = Discriminator(self.__retina_size,
                                                    self.__num_bits_addr,
                                                    self.__position_list,
                                                    self.__is_cumulative,
                                                    self.__ignore_zero_addr)

    # add a example to training in an especific discriminator
    # def add_training(self, disc_name, training_example):
    def train(self, disc_name, training_example):

        if disc_name not in self.__discriminators:
            raise Exception('the discriminator does not exist')

        r = Retina(training_example)
        self.__discriminators[disc_name].train(r)

    def classify(self, example):
        result = {}  # classes and values
        memory_result = {}  # for each class the memories values obtained

        # transform example into a retina
        r = Retina(example)

        for class_name in self.__discriminators:

            # for each class the memorie values obtained
            memory_result[class_name] = self.__discriminators[class_name] \
                                            .classify(r)

            # for each class, store the value
            result[class_name] = sum(memory_result[class_name])

        # applying bleaching method if it is selected
        if self.__bleaching is not None:
            result = self.__bleaching.run(memory_result,
                                          self.__confidence_threshold)

        # applying vacuum method if it is selected
        if self.__vacuum is not None:
            result = self.__vacuum.run(memory_result)

        return result

    def __generate_position_list(self, retina_length):

        # generating all possible positions
        position_list = range(retina_length)

        # mapping random positions (if randomize_positions is True)
        if self.__randomize_positions:  # random positions to mapping aleatory
            rand.shuffle(position_list)

        self.__position_list = position_list


class Bleaching:

    def __init__(self, ini_b):
        self.__initial_b = ini_b

    def run(self, memory_result, confidence_threshold):
        b = self.__initial_b
        previous_result = {}  # if it is not possible continue the method
        result = {}

        for class_name in memory_result:
            valid_values = [1 for x in memory_result[class_name] if x >= b]
            result[class_name] = sum(valid_values)

        previous_result = result.copy()
        confidence = Util.calc_confidence(result)

        while confidence < confidence_threshold:

            previous_result = result
            result = {}
            # generating a new result list using bleaching
            for class_name in memory_result:
                valid_values = [1 for x in memory_result[class_name] if x >= b]
                result[class_name] = sum(valid_values)

            # recalculating confidence
            confidence = Util.calc_confidence(result)
            if confidence == -1:
                return previous_result

            b += 1  # next value of b

        return result


class Vacuum:

    def run(self, list_of_memories):
        result = {}
        num_columns = len(list_of_memories.values()[0])
        sum_list = [0]*num_columns

        
        for class_name in list_of_memories:
            result[class_name] = sum(list_of_memories[class_name])

        confidence = Util.calc_confidence(result)
        

        # sum all memories positions
        for class_name in list_of_memories:
            for column in xrange(num_columns):
                sum_list[column] += list_of_memories[class_name][column]

        # calculating the average for each position
        avg = [0]*num_columns
        for column in xrange(num_columns):
            avg[column] = float(sum_list[column])/len(list_of_memories)

        # applying the vacuumn in each memory
        for class_name in list_of_memories:
            sum_mem = 0
            for column in xrange(num_columns):
                if list_of_memories[class_name][column] > avg[column]*(1-confidence):
                    sum_mem += 1
            result[class_name] = sum_mem
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
