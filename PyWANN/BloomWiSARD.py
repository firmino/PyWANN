# -*- coding: utf-8 -*-
import numpy as np
import copy
import mmh3

class BloomWiSARD:
    def __init__(self,
                 num_bits_addr,
                 memory_size,
                 num_hashs_function,
                 hash_treshold,
                 randomize_positions=True):
        self.__num_bits_addr = num_bits_addr
        self.__memory_size = memory_size
        self.__num_hashs_function = num_hashs_function
        self.__hash_treshold = hash_treshold
        self.__randomize_positions = randomize_positions

        self.__discriminators = {}

        if(randomize_positions == True):
            self.__seed = np.random.randint(0, 100000)

        self.__hash_functions_seed = []

        for i in xrange(num_hashs_function):
            random_seed = np.random.randint(0, 100000)
            self.__hash_functions_seed.append(random_seed)

    def fit(self, X, y):
        classes = set(y)
        self.__retina_length = len(X[0])
        memories_template = {i/self.__num_bits_addr: BloomMemory() \
                            for i in xrange(0,self.__retina_length, self.__num_bits_addr)}
        for cls in classes:
            self.__discriminators[cls] = BloomDiscriminator(self.__retina_length,
                                                            self.__seed,
                                                            self.__num_bits_addr,
                                                            self.__memory_size,
                                                            self.__hash_functions_seed,
                                                            self.__hash_treshold,
                                                            copy.deepcopy(memories_template))
        for i in xrange(len(X)):
            self.__discriminators[y[i]].add_training(X[i])

    def predict(self, X):
        result = []
        for i in xrange(len(X)):
            result.append(self.__predict_one(X[i]))
        return result

    def __predict_one(self, x):
        ranks = {}
        for d in self.__discriminators:
            ranks[d] = self.__discriminators[d].recognize_pattern(x)
        return max(ranks, key=ranks.get)

class BloomMemory:
    def __init__(self):
        self.__data = {}

    def add_value(self, addr):
        self.__data[addr] = 1

    def get_value(self, addr):
        if addr not in self.__data:
            return 0
        else:
            return self.__data[addr]

class BloomDiscriminator:
    def __init__(self,
                 retina_length,
                 seed,
                 num_bits_addr,
                 memory_size,
                 hash_functions_seed,
                 hash_treshold,
                 memories):

        self.__num_bits_addr = num_bits_addr
        self.__memory_size = memory_size
        self.__hash_functions_seed = hash_functions_seed
        self.__hash_treshold = hash_treshold
        self.__seed = seed
        self.__memories = memories

    def add_training(self, x):
        np.random.seed(self.__seed)
        x_shuffled = np.array(x)
        np.random.shuffle(x_shuffled)
        z = 0
        for i in xrange(0, x_shuffled.shape[0], self.__num_bits_addr):
            key = np.packbits(x_shuffled[i:i+self.__num_bits_addr])[0] #key is a integer that cames from a bitarray
            for function in self.__hash_functions_seed:
                h1, h2 = mmh3.hash64(key, seed=function)
                position = h1%self.__memory_size
                self.__memories[z].add_value(position)
            z += 1

    def recognize_pattern(self, x):
        np.random.seed(self.__seed)
        x_shuffled = np.array(x)
        np.random.shuffle(x_shuffled)
        z = 0
        discriminator_rank = []
        for i in xrange(0, x_shuffled.shape[0], self.__num_bits_addr):
            key = np.packbits(x_shuffled[i:i+self.__num_bits_addr])[0] #key is a integer that cames from a bitarray
            memory_rank = 0
            for function in self.__hash_functions_seed:
                h1, h2 = mmh3.hash64(key, seed=function)
                position = h1%self.__memory_size
                memory_rank += self.__memories[z].get_value(position)
            z += 1
            discriminator_rank.append(memory_rank)
        return sum([1 for rank in discriminator_rank if rank > self.__hash_treshold])

if __name__ == '__main__':
    from mnist import MNIST
    import time
    from sklearn import metrics
    time1 = time.time()
    mndata = MNIST('/home/rangel/Desktop/Dataset_MINST/')
    X_not_binary, y = mndata.load_training()
    X_test_not_binary, y_test = mndata.load_testing()

    X_not_binary = X_not_binary[:2000]
    y = y[:2000]
    X_test_not_binary = X_test_not_binary[:1000]
    y_test = y_test[:1000]

    X = []
    X_test = []
    tam = len(X_not_binary[0])

    y = list(y)
    y_test = list(y_test)

    for ex in X_not_binary:
        for i in xrange(tam):
            if ex[i] < 100:
                ex[i] = 0
            else:
                ex[i] = 1
        X.append(ex)

    for ex in X_test_not_binary:
        for i in xrange(tam):
            if ex[i] < 100:
                ex[i] = 0
            else:
                ex[i] = 1
        X_test.append(ex)

    print "Tempo para binarizar a base", time.time() - time1
    time1 = time.time()

    w = BloomWiSARD(num_bits_addr = 4,
                    memory_size = 185,
                    num_hashs_function = 2,
                    hash_treshold = 2)

    w.fit(X, y)

    print "Tempo para fit", time.time() - time1
    time1 = time.time()

    prediction = w.predict(X_test)

    print "Tempo para prediction", time.time() - time1
    print metrics.accuracy_score(y_test, prediction)