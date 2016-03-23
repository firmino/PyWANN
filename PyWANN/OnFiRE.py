# -*- coding: utf-8 -*-
import numpy as np
from WiSARD import WiSARD

class OnFiRE:

    def __init__(self,
                 loss = "squared_error",
                 max_bits = 64,
                 min_bits = 2,
                 bleaching = True,
                 memory_is_cumulative=True,
                 defaul_b_bleaching=1,
                 max_w = 30,
                 nthread = 4,
                 seed = 0,
                 reg_lambda = 0.01,
                 subsample = 0.1,
                 verbose = True):

        self.__bleaching = bleaching
        self.__memory_is_cumulative = memory_is_cumulative
        self.__defaul_b_bleaching = defaul_b_bleaching
        self.__max_w = max_w
        self.__nthread = nthread
        self.__reg_lambda = reg_lambda
        self.__subsample = subsample
        self.__loss = loss
        self.__max_bits = max_bits
        self.__min_bits = min_bits
        self.__verbose = verbose

        np.random.seed(seed=seed)

        if(not loss in ["cross_entropy", "absolute_error", "squared_error", "accuracy"]):
            raise Exception("Loss Function not defined %s"%(self.__loss))

    def fit(self, X, y, early_stopping_rounds=10, eval_set=None):
        self.class_ = np.unique(y)
        self.__X = X
        self.__y = y
        self.__clfs = []
        n_round = 0
        w = 0
        objective_before = np.inf
        self.__eval_set = eval_set

        while(n_round < early_stopping_rounds and w < self.__max_w):
            w += 1
            self.__clfs.append(self.__get_clf())
            continue_ = True
            while(continue_):
                X_, y_, X_eval, y_eval = self.__random_subsample()
                if(len(self.class_) != len(np.unique(y_))):
                    if(self.__verbose):
                        print "Alert: subsample may be too small."
                else:
                    continue_ = False

            self.__clfs[-1].fit(X_, y_)
            if(eval_set != None):
                X_eval = eval_set[0]
                y_eval = eval_set[1]
                ypred = self.predict_proba(X_eval)
                obj_value = self.__objective_function(y_eval, ypred)

                if(self.__verbose):
                    print "Accuracy: ", 1 - self.__error(y_eval, ypred)
            else:
                ypred = self.predict_proba(X_eval)
                obj_value = self.__objective_function(y_eval, ypred)

            if(self.__verbose):
                print "Iteration: %d - Objective Function : %f"%(w, obj_value)

            if objective_before > obj_value:
                objective_before = obj_value
                n_round = 0

            else:
                n_round += 1
                del(self.__clfs[-1])

    def predict_proba(self, X):
        y_proba = np.zeros((len(X), len(self.class_)))
        for clf in self.__clfs:
            y_proba += clf.predict_proba(X)
        y_proba = y_proba/len(self.__clfs)
        return y_proba

    def __random_subsample(self):
        X_size = len(self.__X)
        fit_size = int(X_size*self.__subsample)
        aux = range(X_size)
        np.random.shuffle(aux)

        X_fit = []
        y_fit = []
        X_eval = []
        y_eval = []

        for val in aux[:fit_size]:
            X_fit += [self.__X[val]]
            y_fit += [self.__y[val]]

        if(self.__eval_set == None):
            for val in aux[fit_size:fit_size*2]:
                X_eval += [self.__X[val]]
                y_eval += [self.__y[val]]

        return X_fit, y_fit, X_eval, y_eval

    def __objective_function(self, y, ypred):
        l = self.__loss_function(y, ypred)
        r = self.__reg_function()
        return l + self.__reg_lambda*r

    def __squared_error(self, y, ypred):
        summing = 0
        yh = self.__hot_encoder(y)
        for i in xrange(len(y)):
            summing += np.sum((np.array(ypred[i]) - np.array(yh))**2)
        return summing

    def __loss_function(self, y, ypred):
        if(self.__loss == "cross_entropy"):
            return self.__cross_entropy(y, ypred)
        if(self.__loss == "absolute_error"):
            return self.__absolute_error(y, ypred)
        if(self.__loss == "squared_error"):
            return self.__squared_error(y, ypred)
        if(self.__loss == "error"):
            return self.__error(y, ypred)

    def __reg_function(self):
        summing = np.sum([clf.get_num_bits() for clf in self.__clfs])
        return float(summing)

    def __cross_entropy(self, y, ypred):
        yh = self.__hot_encoder(y)
        summing = 0
        for i in xrange(len(yh)):
            pos = np.argmax(ypred[i])
            summing += yh[i][pos] * np.log(ypred[i][pos])
        return -1*summing

    def __error(self, y, ypred):
        summing = 0
        for i in xrange(len(ypred)):
            pos = np.argmax(ypred[i])
            if(self.class_[pos] == y[i]):
                summing += 1
        return 1 - summing/float(len(ypred))

    def __get_clf(self):
        num_bits_addr = np.random.randint(self.__min_bits, self.__max_bits)
        seed = np.random.randint(0,10000)
        confidence_threshold = np.random.rand()
        return WiSARD(num_bits_addr,
                      bleaching = self.__bleaching,
                      memory_is_cumulative = self.__memory_is_cumulative,
                      defaul_b_bleaching = self.__defaul_b_bleaching,
                      confidence_threshold = confidence_threshold,
                      seed=seed)

    def __hot_encoder(self, y):
        yh = np.zeros((len(y), len(self.class_)))
        for i in xrange(yh.shape[0]):
            pos = np.where(self.class_ == y[i])[0][0]
            yh[i][pos] = 1
        return yh

if __name__ == '__main__':
    from mnist import MNIST
    import random
    import time
    from sklearn.metrics import accuracy_score
    mndata = MNIST("/home/rangel/Desktop/Dataset_MNIST")
    X, y = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    X = np.array(X)
    for x in X:
        x[x<128] = 0
        x[x>=128] = 1

    seed = int(time.time())
    random.seed(seed)
    random.shuffle(X)
    random.seed(seed)
    random.shuffle(y)

    X_test = X[1000:2000]
    y_test = y[1000:2000]

    X = X[:1000]
    y = y[:1000]   

    # clf = OnFiRE(bleaching = False, memory_is_cumulative = False, seed=int(time.time()))
    # clf.fit(X,y,eval_set=[X_test, y_test])

    w = WiSARD(8,
              bleaching = True,
              memory_is_cumulative = True,
              defaul_b_bleaching = 1,
              confidence_threshold = 0.1,
              seed=8383)
    w.fit(X, y)
    ypred = w.predict(X_test)
    print accuracy_score(list(y_test), ypred)