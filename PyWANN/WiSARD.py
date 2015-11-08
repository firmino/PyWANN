# -*- coding: utf-8 -*-

import numpy as np
import itertools
import math


class Memory:

    def __init__(self, num_bits_addr=2, is_cummulative=True, ignore_zero_addr=False):
        
        self.__data = {}
        self.__num_bits_addr = num_bits_addr
        self.__is_cummulative = is_cummulative
        self.__ignore_zero_addr = ignore_zero_addr
    
    def get_memory_size(self):  
        return 2**self.__num_bits_addr
    
    def add_value(self, addr, value=1):
        
        if (not isinstance( addr, int )):
            raise Exception('addr must be a integer')

        if self.__is_cummulative:
            if addr in self.__data:
                self.__data[addr] += value
            else:
                self.__data[addr] = value
        else:
            self.__data[addr] = 1

    def get_value(self, addr):

        # ignore zero is for cases where 0 addr are not important (is a parameter in the WiSARD)
        if self.__ignore_zero_addr and addr == 0:
            return 0

        if addr not in self.__data:
            return 0
        else:
            return self.__data[addr]

    def __int_to_binary(self, addr):

        bin_addr = np.zeros(self.__num_bits_addr)

        quoc = addr
        for i in xrange(self.__num_bits_addr):
            rest = quoc % 2
            quoc = quoc / 2

            bin_addr [self.__num_bits_addr - 1 - i] = rest

        return bin_addr


    def get_part_DRASiW(self):

        part_DRASiW =  np.zeros(self.__num_bits_addr)
        
        for addr_key in  self.__data:
            value = self.__data[addr_key]
            bin_addr = self.__int_to_binary(addr_key)

            for bit_posi in xrange(len(bin_addr)):
                bit = bin_addr[bit_posi]
                
                if bit == 1 and value > 0:

                    if self.__is_cummulative:
                        part_DRASiW[bit_posi] += value
                    else:
                        part_DRASiW[bit_posi] = 1

        return part_DRASiW


class Discriminator:

    def __init__(self,
                 retina_length,
                 mapping_positions,
                 memories):

        self.__retina_length = retina_length        
        self.__mapping_positions = mapping_positions
        self.__memories = memories

    def get_memories(self):
        
        return self.__memories

    def get_memories_mapping(self):
        
        return self.__mapping_positions

    def get_memory(self, index):
        
        return self.__memories[index]

    def add_training(self, retina):

        for (mem_index, mapping) in self.__mapping_positions.iteritems():
            
            #  calculating the position (addr) that will be insert a value into the memory
            #  indexed by mem_index
            addr = 0
            for i in xrange(len(mapping)):
                addr += 2 ** i * retina[ mapping[i] ] 

            self.__memories[mem_index].add_value(addr)

    def classify(self, retina):
        result = np.zeros(len(self.__memories))
        for (mem_index, mapping) in self.__mapping_positions.iteritems():
            
            #  calculating the position (addr) that will be insert a value into the memory
            #  indexed by mem_index
            addr = 0
            for i in xrange(len(mapping)):
                addr += 2 ** i * retina[ mapping[i] ] 

            result[mem_index] = self.__memories[mem_index].get_value(addr)

        return result

    def get_DRASiW(self):

        DRASiW = np.zeros(self.__retina_length)  # DRASiW is like a retina of stored positions
        
        for (mem_index, mapping) in self.__mapping_positions.iteritems():

            DRASiW_part = self.__memories[mem_index].get_part_DRASiW()

            for i in xrange(len(mapping)):
                DRASiW[ mapping[i] ] = DRASiW_part[i]

        return DRASiW



class WiSARD:

    def __init__(self,
                 retina_size,
                 num_bits_addr=2,
                 bleaching=True,
                 confidence_threshold=0.1,
                 randomize_positions=True,
                 default_bleaching_b_value=3,
                 ignore_zero_addr=False):

        self.__num_bits_addr = num_bits_addr
        self.__retina_size = retina_size
        self.__confidence_threshold = confidence_threshold
        self.__is_cumulative = False

        self.__discriminators = {}
        self.__mapping_positions = None
        self.__ignore_zero_addr = ignore_zero_addr
        self.__randomize_positions = randomize_positions
        self.__confidence_threshold = confidence_threshold

        self.__bleaching = None
        if bleaching:
            self.__is_cumulative = True
            self.__bleaching = Bleaching(default_bleaching_b_value)




        # ###############################PASSAR COMO PARÂMETRO PARA O DISCRIMINADOR  ############################
        # # mapping positions for each memory
        # # the mapping will be like {0:[0,2,3], 1:[3,8,7], 2: [1,4,5]}
        # self.__mapping_positions = { i/self.__num_bits_addr : a[i:i+self.__num_bits_addr] \
        #                              for i in xrange(0,self.__retina_length, self.__num_bits_addr)}

        # # creating list of memories
        # self.__memories = { i/self.__num_bits_addr:  Memory( len(self.__mapping_positions[i/self.__num_bits_addr] ), 
        #                                                      memories_are_cumulative,
        #                                                      ignore_zero_addr)  \
        #                     for i in xrange(0,self.__retina_length, self.__num_bits_addr)}  }



    def create_discriminator(self, name):

        # if there is not a mapping position defined
        if self.__mapping_positions is None:
            self.__generate_mapping_positions(self.__retina_size)

        # creating discriminator
        self.__discriminators[name] = Discriminator(self.__retina_size,
                                                    self.__num_bits_addr,
                                                    self.__mapping_positions,
                                                    self.__is_cumulative,
                                                    self.__ignore_zero_addr)

    # add a example to training in an especific discriminator
    # def add_training(self, disc_name, training_example):
    def add_training(self, disc_name, training_example):

        if disc_name not in self.__discriminators:
            raise Exception('the discriminator does not exist')

        r = Retina(training_example)
        self.__discriminators[disc_name].add_training(r)

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
            result = self.__bleaching.run(memory_result, self.__confidence_threshold)

        return result

    def __generate_mapping_positions(self, retina_length):

        # generating all possible positions
        mapping_positions = range(retina_length)

        # mapping random positions (if randomize_positions is True)
        if self.__randomize_positions:  # random positions to mapping aleatory
            rand.shuffle(mapping_positions)

        self.__mapping_positions = mapping_positions


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
            c = 1 - float(second_max) / float(max_value)

            return c

        except Exception, Error:
            return -1