# -*- coding: utf-8 -*-

import numpy as np
import copy


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
                 retina_length,
                 num_bits_addr,
                 bleaching=True,
                 confidence_threshold=0.1,
                 ignore_zero_addr=False, 
                 defaul_b_bleaching=1,
                 randomize_positions=True,
                 memory_is_cumulative=True):

        self.__retina_length = retina_length
        self.__bleaching = bleaching
        self.__defaul_b_bleaching = defaul_b_bleaching
        self.__discriminators = {}
        self.__confidence_threshold = confidence_threshold
        
        positions = np.arange(retina_length)
        if randomize_positions:
            np.random.shuffle(positions)

        #  spliting positions for each memory
        self.__mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                                     for i in xrange(0, retina_length, num_bits_addr) }

        #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
        self.__memories_template = { i/num_bits_addr:  Memory( num_bits_addr = len(self.__mapping_positions[i/num_bits_addr] ), 
                                                               is_cummulative = memory_is_cumulative,
                                                               ignore_zero_addr = ignore_zero_addr)  \
                                   for i in xrange(0,retina_length, num_bits_addr)}



    def create_discriminator(self, name):
        #  have to copy() memories or all discriminator will have the same set of memories
        new_memories = copy.deepcopy(self.__memories_template)
        self.__discriminators[name] = Discriminator(retina_length = self.__retina_length,
                                                    mapping_positions = self.__mapping_positions,
                                                    memories = new_memories) 
        


    #  X is a matrix of retinas (each line will be a retina)
    #  y is a list of label (each line defina a retina in the same position in Y)
    def fit(self, X, y):
        num_samples =  len(y)
        for i in xrange(num_samples):
            retina = X[i]
            label = y[i]
            self.__discriminators[label].add_training (retina)
        
    def predict(self, x):
        
        discriminator_names = [class_name for class_name in self.__discriminators]

        result_value = np.array( [ self.__discriminators[class_name].classify(x) \
                                    for class_name in discriminator_names] )

        result_sum = np.sum(result_value[:]>=1, axis=1) 
        
        if self.__bleaching:
            b = self.__defaul_b_bleaching
            confidence = self.__calc_confidence(result_sum)

            while confidence < self.__confidence_threshold:
                result_sum = np.sum(result_value[:]>=1, axis=1)
                confidence = self.__calc_confidence(result_sum)
                b += 1

        result = { discriminator_names[i] : result_sum[i] for i in xrange(len(result_sum) )}

        return result
        
    def __calc_confidence(self,results):
            
        # getting max value
        max_value = results.max()

        # getting second max value
        second_max = results[results < max_value].max()

        # calculating confidence value
        c = 1 - float(second_max) / float(max_value)

        return c

        