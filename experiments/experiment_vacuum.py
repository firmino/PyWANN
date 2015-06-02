
import sys
sys.path.append("..")
from PyWANN import Wisard
from databases.MNIST.images_10k import *
import time
import multiprocessing
import random
import numpy as np

def classification(retina_size, num_bits, 
                   clazzs, fold,
                   use_vacuum, use_bleaching, confidence_threshold, 
                   positions, annotation, base, 
                   output):

            w = Wisard(retina_size,num_bits,
                       use_vacuum, use_bleaching, confidence_threshold)

            for clazz in clazzs:
                w.add_discriminator(clazz)

            for position in positions:
                class_name = annotation[position]
                value = base[position]
                w.add_training(class_name, value)

            #classifing
            acertos = 0
            for position in fold:                 
                result = w.classifier(base[position])

                selected_class =  max(result, key=result.get)

                if selected_class == annotation[position]:
                    acertos += 1

            output.put(acertos/float(len(fold)) )


def experiment():

        # wisard parameters
        num_bits = 4
        retina_size = (28*28)
        use_vacuum = False
        use_bleaching = True,
        confidence_threshold = 0.6

        # k-fold parameters
        k = 10
        annotation = labels  # from MNIST.images_10k
        base = images  # from MNIST.images_10k
        num_samples = len(annotation)
        fold_size = num_samples/k
        clazzs = set(annotation)


        # random positions
        aux_vec = range(0, num_samples )
        random.shuffle(aux_vec)

        # creating folds
        folds = {}
        for i in xrange(k):
            folds[i] = aux_vec[i*fold_size: (i+1)*fold_size]


        # testing data
        output = multiprocessing.Queue()
        processes = []  # a process per fold
        
        for i in folds:
            # training fold
            positions = []
            candidate_list = [folds[key] for key in folds if key != i]
            for candidates in candidate_list:
                for item in candidates:
                    positions.append(item)


            parameters = (retina_size, num_bits, clazzs, folds[i], 
                          use_vacuum, use_bleaching, confidence_threshold, 
                          positions, annotation, base, output) 

            p = multiprocessing.Process(target=classification, args=parameters)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        vec_result = [output.get() for p in processes]  # getting value for each process
        mean = np.mean(vec_result)
        std = np.std(vec_result) 
        


        print "\n"
        print "-" * 30
        print "Classification Results"

        if use_bleaching:
            print "Method: Bleaching" 
            print "confidence Threshold: " + str(confidence_threshold) 
        elif use_vacuum:
            print "Method: Vacuum"
        else:
            print "Method: None" 

        print "Num Folds: " + str(k)

        print "Results: " + str(vec_result)
        print "AVG: " + str(mean)
        print "STD: " + str(std)

        print "-" * 30    

if __name__ == "__main__":
    experiment()
