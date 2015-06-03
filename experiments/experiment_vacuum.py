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
                w.create_discriminator(clazz)

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

def experiment(num_bits, use_vacuum, use_bleaching, sample_size, confidence_threshold):

        retina_size = 28*28

        # k-fold parameters
        k = 10
        annotation = labels[0:1000]  # from MNIST.images_10k
        base = images[0:1000]  # from MNIST.images_10k
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
        

        ini_time = time.time()
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
        
        total_time = time.time() - ini_time
        return vec_result, mean, std, total_time

        
if __name__ == "__main__":

    output_file = open("~/Desktop/saida_experimento_vacuum",'wr')

    output_file.write("method;num_bits;num_inputs;avg;std;time\n")

    num_bits = [2, 4, 6, 8, 10, 12, 14, 16]
    size_samples = [100, 1000, 10000]
    confidence_threshold = 0.1
    
    # wisard
    method = "wisard"
    use_vacuum = False
    use_bleaching = False
    print("Start: "+ method)
    for num_bit in num_bits:
        for size_sample in size_samples:
            res = experiment(num_bit, use_vacuum, use_bleaching, size_sample, confidence_threshold)
            line = ("%s;%d;%d;%f;%f;%f\n") % (method,num_bit,size_sample,res[1],res[2],res[3])
            output_file.write(line)
            print("\tFINISHED STEP: %d number of bits and %d samples") %(num_bit, size_sample)
    print "-"*30

    # bleaching
    method = "wisard + bleaching"
    use_vacuum = False
    use_bleaching = True
    print("Start: "+ method)
    for num_bit in num_bits:
        for size_sample in size_samples:
            res = experiment(num_bit, use_vacuum, use_bleaching, size_sample, confidence_threshold)
            line = ("%s;%d;%d;%f;%f;%f\n") % (method,num_bit,size_sample,res[1],res[2],res[3])
            output_file.write(line)
            print("\tFINISHED STEP: %d number of bits and %d samples") %(num_bit, size_sample)
    print "-"*30

    # vacuum
    method = "wisard + vacuum"
    use_vacuum = True
    use_bleaching = False
    print("Start: "+ method)
    for num_bit in num_bits:
        for size_sample in size_samples:
            res = experiment(num_bit, use_vacuum, use_bleaching, size_sample, confidence_threshold)
            line = ("%s;%d;%d;%f;%f;%f\n") % (method,num_bit,size_sample,res[1],res[2],res[3])
            output_file.write(line)
            print("\tFINISHED STEP: %d number of bits and %d samples") %(num_bit, size_sample)
    print "-"*30
    output_file.close()