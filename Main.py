import random
import numpy as np
from PyWANN import Wisard
from databases.MNIST.images_10k import *




def run():
 		# wisard parameters
        num_bits = 4
        retina_size = (28*28)
        use_vacuum = False
        use_bleaching = True,
        confidence_threshold = 0.6

        # k-fold parameters
        k = 10
        annotation = labels
        base = images
        num_samples = len(annotation)
        fold_size = num_samples/k
        clazzs = set(annotation)

        # sorting positions
        aux_vec = range(0, num_samples )
        random.shuffle(aux_vec)

        # creating folds
        folds = {}
        for i in xrange(k):
            folds[i] = aux_vec[i*fold_size: (i+1)*fold_size]


        vec_result = []
        # testing data
        for i in folds:

            # training fold
            positions = []
            candidate_list = [folds[key] for key in folds if key != i]
            for candidates in candidate_list:
                for item in candidates:
                    positions.append(item)

        
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
            for position in folds[i]:                 
                result = w.classifier(base[position])

                selected_class =  max(result, key=result.get)

                if selected_class == annotation[position]:
                	acertos += 1

            vec_result.append(acertos/float(len(folds[i])))

        

        mean = np.mean(vec_result)
        std = np.std(vec_result) 
	
	




run()