import unittest

from PyWANN.CoWiSARD import Discriminator
from samples import *


class TestDiscriminator(unittest.TestCase):


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
    
    def test_num_memory_conv_layer(self):

        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        list_conv = [conv_1, conv_2]

        cross= samples["cross"]
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits_layer = 2


        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits_first_layer = num_bits_layer,
                          num_memo_to_combine = 2,
                          list_conv_matrix = list_conv)


        samples_size = len(expected_cross["vertical"]) * len(expected_cross["vertical"][0])
        expected_number_of_memories = samples_size / num_bits_layer


        self.assertEquals( len(d._Discriminator__conv_memories[0]), expected_number_of_memories)
    

    def test_invalid_number_combined_layers(self):
        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        list_conv = [conv_1, conv_2]

        cross= samples["cross"]
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits_layer = 2
        num_memo_combined_layer = 17


        raise_a_exception = False
        try: 
            d = Discriminator(retina_width=retina_width,
                              retina_height=retina_height,
                              num_bits_first_layer = num_bits_layer,
                              num_memo_to_combine = num_memo_combined_layer,
                              list_conv_matrix = list_conv)
        except:
            raise_a_exception = True
        

        self.assertTrue(raise_a_exception)
    

       
if __name__ == "__main__":

    unittest.main()
