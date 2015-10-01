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
                          num_bits=2,
                          list_conv_matrix=list_conv_matrix)

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
                    print i, j
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

    
    def test_num_memory_retina_proportional_number_of_bits(self):

        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        
        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so is a multiple of 2 (num_bits)
        
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits = 2 


        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)

        num_memory_expected = (retina_width * retina_height) / num_bits
        num_memory_obtained = len(d._Discriminator__memories)

        self.assertEquals( num_memory_expected, num_memory_obtained )
    


    def test_num_memory_retina_not_proportional_number_of_bits(self):

        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        
        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so it is not a multiple of 5 (num_bits)
        
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits = 5 

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)

        num_memory_expected = ((retina_width * retina_height) / num_bits) + 1
        num_memory_obtained = len(d._Discriminator__memories)

        self.assertEquals( num_memory_expected, num_memory_obtained )

    def test_mapping(self):
        
        conv_1 = [[1,1,1],[1,1,1],[1,1,1]]
        conv_2 = [[1,1,1],[1,1,1],[1,1,1]]
        
        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so it is not a multiple of 5 (num_bits)
        
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits = 5 

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)
        
        mapping_size = len(d._Discriminator__mapping)
        mapping_min = min(d._Discriminator__mapping)
        mapping_max = max(d._Discriminator__mapping)
        
        self.assertEquals(mapping_size, (retina_width * retina_height) )
        self.assertEquals( 0, mapping_min)
        self.assertEquals((retina_width * retina_height)-1, mapping_max)


    def test_process_image(self):

        conv_1 = [[-1,0,1],
                  [-1,0,1],
                  [-1,0,1]]

        conv_2 = [[-1,-1,-1],
                  [ 0, 0, 0],
                  [ 1, 1, 1]]
        
        cross = samples["cross"]

        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so it is not a multiple of 5 (num_bits)
        
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits = 5 

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)

        processed_retina = d._Discriminator__process_retina(cross)

        processed_right = True
        for i in range(len(processed_retina)):

            if i == 36: #  be 1 only in the midle
                continue

            if processed_retina[i] != 0:
                processed_right = False
                break

        self.assertTrue(processed_right)
        

    def test_add_training(self):

        conv_1 = [[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]]

        conv_2 = [[-1,-1,-1],
                  [ 0, 0, 0],
                  [ 1, 1, 1]]
        
        cross = samples["cross"]

        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so it is not a multiple of 5 (num_bits)
        
        retina_width  = len(cross)
        retina_height = len(cross[0])
        num_bits = 3 

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)


        num_memory_obtained = len(d._Discriminator__memories)

        # generating the all possible address
        address = [[x,y,z] for x in range(0,2) for y in range(0,2) for z in range(0,2)]


        # testing if all positions in all memories are zero
        result = 0
        for mem in d._Discriminator__memories:
            for addr in address:
                result += mem.get_value(addr)

        self.assertEquals(result, 0)


        # testing for one trainning case
        result = 0
        d.add_trainning(cross)
        for mem in d._Discriminator__memories:
            for addr in address:
                result += mem.get_value(addr)

        # must be one (cause one example was presented) to each memory
        # so, the sum is equal to num of memories
        self.assertEquals(result, num_memory_obtained)


    def test_classify(self):
        
        conv_1 = [[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]]

        conv_2 = [[-1, 0, 1],
                  [-1, 0, 0],
                  [-1, 0, 1]]

        vertical_example = [[0,0,1,0],
                            [0,0,1,0],
                            [0,0,1,0],
                            [0,0,1,0]]

        horizontal_example = [[0,0,0,0],
                              [1,1,1,1],
                              [0,0,0,0],
                              [0,0,0,0]]

        
        list_conv = [conv_1, conv_2]

        cross= samples["cross"] # cross has 8*8 pixels, so it is not a multiple of 5 (num_bits)
        
        retina_width  = len(vertical_example)
        retina_height = len(vertical_example[0])
        num_bits = 3 

        d = Discriminator(retina_width=retina_width,
                          retina_height=retina_height,
                          num_bits=num_bits,
                          list_conv_matrix=list_conv)

        d.add_trainning(vertical_example)

        result_vertical   = d.classify(vertical_example, 1)
        result_horizontal = d.classify(horizontal_example, 1)

        self.assertTrue(result_vertical > result_horizontal)


if __name__ == "__main__":

    unittest.main()
