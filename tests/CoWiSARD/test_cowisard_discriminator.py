import unittest

from PyWANN.CoWiSARD  import Discriminator


class TestDiscriminator(unittest.TestCase):


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
    

    def test_add_train(self):

        retina =    [[1,1,0,0,0,0],
                     [1,1,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,1,0]]

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

        d.add_train(retina)
        d.add_train(retina)
        d.add_train(retina)   

        self.assertEquals(d._Discriminator__memory_table[0][0],3)


    def test_classify(self):

        retina =    [[1,1,0,0,0,0],
                     [1,1,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,1,0]]

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

        d.add_train(retina)
        d.add_train(retina)
        d.add_train(retina)   

        self.assertEquals(d.classify(retina), 8)



if __name__ == "__main__":

    unittest.main()
