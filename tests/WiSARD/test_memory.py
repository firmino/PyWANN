
import unittest

from PyWANN.WiSARD  import Memory
import numpy as np


class TestMemory(unittest.TestCase):

    def test_memory_size(self):
        m = Memory(3)
        self.assertEquals(2**3, m.get_memory_size())

    def test_bit_conversion(self):
        m = Memory(4)
        bin_0 = m._Memory__int_to_binary(0)
        bin_1 = m._Memory__int_to_binary(1)
        bin_2 = m._Memory__int_to_binary(2)
        bin_3 = m._Memory__int_to_binary(3)
        bin_4 = m._Memory__int_to_binary(4)
        bin_5 = m._Memory__int_to_binary(5)
        bin_6 = m._Memory__int_to_binary(6)
        bin_7 = m._Memory__int_to_binary(7)
        bin_8 = m._Memory__int_to_binary(8)
        bin_9 = m._Memory__int_to_binary(9)
        bin_10 = m._Memory__int_to_binary(10)

        self.assertTrue(np.array_equal(bin_0,  [0,0,0,0]))
        self.assertTrue(np.array_equal(bin_1,  [0,0,0,1]))
        self.assertTrue(np.array_equal(bin_2,  [0,0,1,0]))
        self.assertTrue(np.array_equal(bin_3,  [0,0,1,1]))
        self.assertTrue(np.array_equal(bin_4,  [0,1,0,0]))
        self.assertTrue(np.array_equal(bin_5,  [0,1,0,1]))
        self.assertTrue(np.array_equal(bin_6,  [0,1,1,0]))
        self.assertTrue(np.array_equal(bin_7,  [0,1,1,1]))
        self.assertTrue(np.array_equal(bin_8,  [1,0,0,0]))
        self.assertTrue(np.array_equal(bin_9,  [1,0,0,1]))
        self.assertTrue(np.array_equal(bin_10, [1,0,1,0]))


    def test_add_and_get_value_cumulative(self):
        m = Memory(4)

        m.add_value(addr=1)
        m.add_value(addr=1)
        m.add_value(addr=1)
        m.add_value(addr=1)

        self.assertEquals(m.get_value(1), 4)

    def test_add_and_get_value_non_cumulative(self):
        m = Memory(4, is_cummulative=False)

        m.add_value(addr=1)
        m.add_value(addr=1)
        m.add_value(addr=1)
        m.add_value(addr=1)

        self.assertEquals(m.get_value(1), 1)


    def test_DRASiW_cumulative(self):
        
        m = Memory(4)

        m.add_value(addr=10)
        m.add_value(addr=10)
        m.add_value(addr=10)
        m.add_value(addr=10)

        
        self.assertTrue(np.array_equal(m.get_part_DRASiW(), [4,0,4,0]))

    def test_DRASiW_non_cumulative(self):
        
        m = Memory(4, is_cummulative=False)

        m.add_value(addr=10)
        m.add_value(addr=10)
        m.add_value(addr=10)
        m.add_value(addr=10)

        self.assertTrue(np.array_equal(m.get_part_DRASiW(), [1,0,1,0]))

    
    def test_invalid_type_address(self):
        
        addr = [1.0,1]
        m = Memory(3)
        self.assertRaises(Exception, m.add_value, addr)

    
    def test_ignore_zero_addr(self):
         m = Memory(3, ignore_zero_addr=True)
         addr = 0

         m.add_value(addr)
         m.add_value(addr)
         m.add_value(addr)
         m.add_value(addr)
         m.add_value(addr)

         self.assertEquals(m.get_value(addr), 0)

    def test_not_ignore_zero_addr(self):
        m = Memory(3, ignore_zero_addr=False)  #  False is default value
        addr = 0

        m.add_value(addr)
        m.add_value(addr)
        m.add_value(addr)
        m.add_value(addr)

        self.assertEquals(m.get_value(addr), 4)

if __name__ == "__main__":
    unittest.main()    
