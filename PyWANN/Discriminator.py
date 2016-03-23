# -*- coding: utf-8 -*-

import numpy as np
from Memory import Memory


class Discriminator:

    def __init__(self,
                 retina_length,
                 num_bits_addr=2,
                 memory_is_cumulative=True,
                 ignore_zero_addr=False,
                 random_positions=True,
                 seed=424242):

        if (not isinstance(retina_length, int)):
            raise Exception('retina_length must be a integer')

        if (not isinstance(num_bits_addr, int)):
            raise Exception('num_bits_addr must be a integer')

        if (not isinstance(memory_is_cumulative, bool)):
            raise Exception('memory_is_cumulative must be a boolean')

        if (not isinstance(ignore_zero_addr, bool)):
            raise Exception('ignore_zero_addr must be a boolean')

        if (not isinstance(random_positions, bool)):
            raise Exception('random_positions must be a boolean')

        if (not isinstance(seed, int)):
            raise Exception('seed must be a boolean')

        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr

        self.__mapping_positions = np.arange(retina_length)
        if random_positions:
            np.random.seed(seed)
            np.random.shuffle(self.__mapping_positions)

        num_memories = self.__retina_length/self.__num_bits_addr
        self.__memories = []
        for i in xrange(0, num_memories):
            m = Memory(num_bits_addr=self.__num_bits_addr,
                       is_cummulative=memory_is_cumulative,
                       ignore_zero_addr=ignore_zero_addr)
            self.__memories.append(m)

        # if there is rest positions
        if retina_length % num_bits_addr > 0:
            num_rest_positions = retina_length % num_bits_addr
            m = Memory(num_bits_addr=num_rest_positions,
                       is_cummulative=memory_is_cumulative,
                       ignore_zero_addr=ignore_zero_addr)
            self.__memories.append(m)

    def add_training(self, retina):
        if (not isinstance(retina, np.ndarray)):
            raise Exception('retina must be a np.ndarray')

        if (len(retina.shape) > 1):
            raise Exception('retina must be a 1D np.ndarray')

        num_cicles = self.__retina_length - self.__num_bits_addr + 1

        for i in xrange(0, num_cicles, self.__num_bits_addr):
            bin_addr = self.__mapping_positions[i:i+self.__num_bits_addr]
            int_addr = np.sum([2**i*retina[bin_addr[posi]]
                               for posi in xrange(self.__num_bits_addr)])
            mem_index = i/self.__num_bits_addr
            self.__memories[mem_index].add_value(addr=int_addr, value=1)

        # if there is rest positions
        if self.__retina_length % self.__num_bits_addr > 0:
            num_rest_positions = self.__retina_length % self.__num_bits_addr
            bin_addr = self.__mapping_positions[-num_rest_positions:]
            int_addr = np.sum([2**i*retina[bin_addr[posi]]
                               for posi in xrange(num_rest_positions)])
            self.__memories[-1].add_value(addr=int_addr, value=1)

    def predict(self, retina):

        if (not isinstance(retina, np.ndarray)):
            raise Exception('retina must be a np.ndarray')

        if (len(retina.shape) > 1):
            raise Exception('retina must be a 1D np.ndarray')

        result = []

        num_cicles = self.__retina_length - self.__num_bits_addr + 1
        for i in xrange(0, num_cicles, self.__num_bits_addr):
            bin_addr = self.__mapping_positions[i:i+self.__num_bits_addr]
            int_addr = np.sum([2**i*retina[bin_addr[posi]]
                               for posi in xrange(self.__num_bits_addr)])
            mem_index = i/self.__num_bits_addr
            value = self.__memories[mem_index].get_value(addr=int_addr)
            result.append(value)

        # if there is rest positions
        if self.__retina_length % self.__num_bits_addr > 0:
            num_rest_positions = self.__retina_length % self.__num_bits_addr
            bin_addr = self.__mapping_positions[-num_rest_positions:]
            int_addr = np.sum([2**i*retina[bin_addr[posi]]
                               for posi in xrange(num_rest_positions)])
            value = self.__memories[-1].get_value(addr=int_addr)
            result.append(value)

        return result

    def get_DRASiW(self):
        # DRASiW is like a retina of stored positions
        DRASiW = np.zeros(self.__retina_length)

        for i in xrange(0, self.__retina_length, self.__num_bits_addr):
            mem_index = i/self.__num_bits_addr
            DRASiW_part = self.__memories[mem_index].get_part_DRASiW()
            positions = self.__mapping_positions[i:i+self.__num_bits_addr]
            for j in xrange(self.__num_bits_addr):
                posi = positions[j]
                DRASiW[posi] = DRASiW_part[j]

        # if there is rest positions
        if retina_length % num_bits_addr_addr > 0:
            DRASiW_part = self.__memories[-1].get_part_DRASiW()
            positions = self.__mapping_positions[-self.__num_bits_addr:]
            for j in xrange(self.__num_bits_addr):
                posi = positions[j]
                DRASiW[posi] = DRASiW_part[j]

        return DRASiW
