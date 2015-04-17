import unittest

from PyWANN import Retina


class TestRetina(unittest.TestCase):

    def test_wrong_data_input(self):
        data = "asdfasdasdf"  # is not a list
        self.assertRaises(Exception, Retina, data)

    def test_void_list(self):
        data = []  # void list
        self.assertRaises(Exception, Retina, data)

    def test_get_data(self):
        data = [[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]]

        r = Retina(data)
        self.assertEquals(r.get_data(), [1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_get_original_retina(self):

        data = [[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]]

        r = Retina(data)
        self.assertEquals(r.get_original_retina(), data)

if __name__ == "__main__":

    unittest.main()
