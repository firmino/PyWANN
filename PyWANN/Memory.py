# -*- coding: utf-8 -*-
import numpy as np


class Memory:

    def __init__(self,
                 num_bits_addr=2,
                 is_cummulative=True,
                 ignore_zero_addr=False):

        self.__data = {}
        self.__num_bits_addr = num_bits_addr
        self.__is_cummulative = is_cummulative
        self.__ignore_zero_addr = ignore_zero_addr

    def get_memory_size(self):
        return 2**self.__num_bits_addr

    def add_value(self, addr, value=1):
        if (not isinstance(addr, int)):
            raise Exception('addr must be a integer')

        if self.__is_cummulative:
            if addr in self.__data:
                self.__data[addr] += value
            else:
                self.__data[addr] = value
        else:
            self.__data[addr] = 1

    def get_value(self, addr):
        # ignore zero is for cases where 0 addr are
        # not important (is a parameter in the WiSARD)
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

            bin_addr[self.__num_bits_addr - 1 - i] = rest

        return bin_addr

    def get_part_DRASiW(self):
        part_DRASiW = np.zeros(self.__num_bits_addr)

        for addr_key in self.__data:
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
