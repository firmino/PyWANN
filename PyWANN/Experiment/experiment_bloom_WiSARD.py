from mnist import MNIST
import time
from sklearn import metrics

if __name__ == '__main__':
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