import gzip
import pickle
from struct import unpack



class MNISTItem:

    def __init__(self):
        self.__label = "not defined"
        self.__data  = []
        self.__threshold = 15


    def get_labe(self):
        return self.__label

    def get_data(self):
        return self.__data

    def set_data(self, label, raw_img):
        self.__label = label

        for line in raw_img:
            aux_line = []

            for item in line:
                if item >= self.__threshold:
                    aux_line.append(1)
                else:
                    aux_line.append(0)

            self.__data.append(aux_line)


        self.print_data()

    def print_data(self):

        print "LABEL: " + str(self.__label)
        print "DATA:\n"
        for line in self.__data:
            for item in line:
                print "%d"%(item),
            print "\n"
        print "#"*20



        
class MNISTDB:


    def __init__(self):

        self.__images_name = "raw_data/t10k-images-idx3-ubyte.gz"
        self.__labels_name = "raw_data/t10k-labels-idx1-ubyte.gz"
        self.__database_file = "processed_data/MNIST"


    def processes_data(self):

        images = gzip.open(self.__images_name, 'rb')
        labels = gzip.open(self.__labels_name, 'rb')


        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]


        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]


        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = labels.read(4)
        N = unpack('>I', N)[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')


        out_img = []
        for num_img in xrange(number_of_images):

            if num_img % 1000 == 0:
                print num_img

            image = []
            for row in xrange(rows):
                linha = []
                for col in xrange(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    linha.append(tmp_pixel)
                image.append(linha)

            # reading label
            tmp_label = labels.read(1)
            tmp_label = unpack('>B', tmp_label)[0]

            # serializing object
            mnist_item = MNISTItem()
            mnist_item.set_data(tmp_label, image)
            
            #storing image with label
            out_img.append(mnist_item)


        file_output = self.__database_file
        output = open(file_output,'wb') 
        pickle.dump(mnist_item,output)   
        output.close()


    def read_data(self):
        # we open the file for reading
        db_mnist = open(self.__database_file,'r')  
        
        # load the object from the file into var b
        b = pickle.load(fileObject)  

        return b


if __name__ == "__main__":
    m = MNISTDB()
    m.processes_data()