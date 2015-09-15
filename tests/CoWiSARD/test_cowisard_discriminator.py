import unittest

from PyWANN.CoWiSARD import Discriminator
from samples import *


class TestDiscriminator(unittest.TestCase):


    def test_list_to_int(self):

        list_0  = [0,0,0,0,0]
        list_32 = [1,1,1,1,1]
        list_17 = [1,0,0,0,1]

        retina_width = 0
        retina_height = 0

        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]
        conv_box = (len(conv_1), len(conv_1[0]))

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits_first_layer = 2,
                          num_memo_to_combine = 2,
                          list_conv_matrix = list_conv)

        self.assertEquals(0, d._Discriminator__list_to_int(list_0))
        self.assertEquals(31, d._Discriminator__list_to_int(list_32))
        self.assertEquals(17, d._Discriminator__list_to_int(list_17))


    def test_conv_matrix(self):

        list_conv_matrix = [  [[ -1,  0,  1],
                               [ -1,  0,  1],
                               [ -1,  0,  1]],

                              [[ -1, -1, -1],
                               [  0,  0,  0],
                               [  1,  1,  1]],

                              [[  0,  1,  1],
                               [ -1,  0,  1],
                               [ -1, -1,  0]],

                               [[ 1,  1,   0],
                                [ 1,  0,  -1],
                                [ 0, -1,  -1]] ]

        cross = samples["cross"]

        retina_width  = len(cross)
        retina_height = len(cross[0])

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits_first_layer = 2,
                          num_memo_to_combine = 2,
                          list_conv_matrix = list_conv_matrix)

        result_vert = d._Discriminator__conv_img(cross, list_conv_matrix[0])
        result_hori = d._Discriminator__conv_img(cross, list_conv_matrix[1])
        result_left_diag = d._Discriminator__conv_img(cross, list_conv_matrix[2])
        result_right_diag = d._Discriminator__conv_img(cross, list_conv_matrix[3])
        

        is_vertical   = True
        is_horizontal = True
        is_left_diag  = True
        is_right_diag = True
        for i in range( len(result_vert) ):
            for j in range( len(result_vert[0]) ):
                
                if result_vert[i][j] != expected_cross["vertical"][i][j]:
                  is_vertical = False
                  break

                if result_hori[i][j] != expected_cross["horizontal"][i][j]:
                  is_horizontal = False
                  break

                if result_left_diag[i][j] != expected_cross["left_diagonal"][i][j]:
                  is_left_diag = False
                  break

                if result_right_diag[i][j] != expected_cross["right_diagonal"][i][j]:
                  is_right_diag = False
                  break

        self.assertTrue(is_vertical)
        self.assertTrue(is_horizontal)
        self.assertTrue(is_left_diag)
        self.assertTrue(is_right_diag)


    
    def test_retina_width_and_height(self):
        
        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        list_conv = [conv_1, conv_2]

        cross= samples["cross"]
        retina_width  = len(cross)
        retina_height = len(cross[0])

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits_first_layer = 2,
                          num_memo_to_combine = 2,
                          list_conv_matrix = list_conv)

        retina_filtered_width  = len(expected_cross["vertical"])
        retina_filtered_height = len(expected_cross["vertical"][0])
        
        self.assertEquals(d._Discriminator__retina_width_filtered, retina_filtered_width)
        
        self.assertEquals(d._Discriminator__retina_height_filtered, retina_filtered_height)
    

    def test_num_memory(self):
      pass

    def test_mapping_each_filtered_image(self):
      pass

    def test_mapping_combined_memories(self):
      pass

    def test_number_combined_memories(self):
      pass







       
if __name__ == "__main__":

    unittest.main()
