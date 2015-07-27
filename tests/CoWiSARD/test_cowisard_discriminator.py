import unittest

from PyWANN.CoWiSARD  import Discriminator


class TestDiscriminator(unittest.TestCase):



    def test_list_to_int(self):
        list_0 = [0,0,0,0,0]
        list_32 = [1,1,1,1,1]
        list_17 = [1,0,0,0,1]

        retina_width = 0
        retina_height = 0
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]
        conv_box = (len(conv_1), len(conv_1[0]))

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv,
                          (3,3))

        self.assertEquals(0, d._Discriminator__list_to_int(list_0))
        self.assertEquals(31, d._Discriminator__list_to_int(list_32))
        self.assertEquals(17, d._Discriminator__list_to_int(list_17))

    def test_percent_overlap(self):
      
        retina =  [[1,1,0,0,0,0],
                   [1,1,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]]

        retina_width = len(retina)
        retina_height = len(retina[0])  
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]
        conv_box = (len(conv_1), len(conv_1[0]))

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv,
                          (3,3))

        perc_overlap = d._Discriminator__percent_overlap(conv_1, retina, 0, 0 )
        self.assertEquals (perc_overlap, 1.0)

        perc_overlap = d._Discriminator__percent_overlap(conv_1, retina, 0, 1 )
        self.assertEquals (perc_overlap, 0.5)

        perc_overlap = d._Discriminator__percent_overlap(conv_1, retina, 0, 2 )
        self.assertEquals (perc_overlap, 0.0)

    def test_has_overlap(self):
        
        retina =  [[1,1,0,0,0,0],
                   [1,1,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]]

        retina_width = len(retina)
        retina_height = len(retina[0])  
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]
        conv_box = (len(conv_1), len(conv_1[0]))

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv,
                          (3,3))        

        self.assertTrue(d._Discriminator__has_overlap(conv_1, retina, 0, 3, 0, 3))
        self.assertFalse(d._Discriminator__has_overlap(conv_2, retina, 0, 3, 0, 3) )


    def  test_train(self):

        retina =  [[1,1,0,0,0,0],
                   [1,1,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]]

        retina_width = len(retina)
        retina_height = len(retina[0])  
        conv_1 = [[1,1],[1,1]]
        conv_2 = [[0,1],[1,0]]
        list_conv = [conv_1, conv_2]
        conv_box = (3, 3)

        d = Discriminator(retina_width,
                          retina_height,
                          list_conv,
                          conv_box)        

        print d._Discriminator__conv_memory




if __name__ == "__main__":

    unittest.main()
