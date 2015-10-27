
import unittest

from PyWANN.WiSARD  import Memory


class TestMemory(unittest.TestCase):

    def test_memory_size(self):
        m = Memory(3)
        self.assertEquals(2**3, m.get_memory_size())

    def test_bit_conversion(self):
        m = Memory(2)
        bits = [1, 0, 1, 1]
        self.assertEquals(m._Memory__list_to_int(bits), 11)

    def test_add_value_address_no_cummulative(self):
        m = Memory(3)
        addr = [1, 0, 1]
        m.add_value(addr, 1)
        self.assertEquals(m.get_value(addr), 1)

    def test_add_value_address_cummulative(self):
        m = Memory(3, is_cummulative=True)
        addr = [1, 0, 1]
        m.add_value(addr, 1)
        m.add_value(addr, 2)
        self.assertEquals(m.get_value(addr), 3)

    def test_invalid_type_address(self):
        m = Memory(3)
        addr = "0,1,0"
        self.assertRaises(Exception, m.add_value, addr, 1)

    def test_get_value(self):
        m = Memory(3)
        addr = [1, 1, 1]
        m.add_value(addr, 1)
        self.assertEquals(m.get_value(addr), 1)

    def test_ignore_zero_addr(self):
        m = Memory(3,False,True)
        addr = [0,0,0]
        m.add_value(addr, 1)
        self.assertEquals(m.get_value(addr), 0)


if __name__ == "__main__":

    unittest.main()
