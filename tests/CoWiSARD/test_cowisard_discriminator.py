import unittest

from PyWANN.CoWiSARD  import Discriminator


class TestDiscriminator(unittest.TestCase):





    def test_number_of_memories(self):
        retina_width = 6
        retina_height = 6
        

        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]


        d = Discriminator(retina_width,
                          retina_height,
                          list_conv)


        num_memories = len(list_conv) * (retina_width/2) * (retina_height/2)

        self.assertEquals(len(d.get_memories()) * len(d.get_memories()[0]), num_memories)

    
    def test_calculate_superposition(self):

        retina_width = 6
        retina_height = 6
    
        retina = [[1,1,0,0,0,0],
                  [1,1,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]

        
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv)

        # for the first conv_matrix
        conv_index = 0 

        result = d._Discriminator__calculate_superposition(conv_index, retina, 0, 0)
        self.assertEquals(result, 4)

        result = d._Discriminator__calculate_superposition(conv_index, retina, 2, 2)
        self.assertEquals(result, 0)        

        # for the second conv_matrix
        conv_index = 1 

        result = d._Discriminator__calculate_superposition(conv_index, retina, 0, 0)
        self.assertEquals(result, 2)

        result = d._Discriminator__calculate_superposition(conv_index, retina, 2, 2)
        self.assertEquals(result, 2)        

        result = d._Discriminator__calculate_superposition(conv_index, retina, 4, 4)
        self.assertEquals(result, 2)


    def test_train(self):        

        retina_width = 6
        retina_height = 6
        
        retina = [[1,1,0,0,0,0],
                  [1,1,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]

        
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv)

        d.add_train(retina)

        self.assertEquals( d.get_memories()[0][0].get_value(4), 1)
        self.assertEquals( d.get_memories()[1][0].get_value(2), 1)


    def test_classify(self):
        
        retina_width = 6
        retina_height = 6
        
        retina = [[1,1,0,0,0,0],
                  [1,1,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]

        
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]

        list_conv = [conv_1, conv_2]

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv)

        d.add_train(retina)

        example_to_classify = [[1,1,0,0,0,0],
                               [1,1,0,0,0,0],
                               [1,1,0,0,0,0],
                               [1,1,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]]
      
        self.assertEquals( d.classify(example_to_classify), 17)
        

if __name__ == "__main__":

    unittest.main()
