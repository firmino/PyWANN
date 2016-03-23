# -*- coding: utf-8 -*-
import numpy as np
from WiSARD import WiSARD

class OnFiRE:

    def __init__(self,
                 loss = "cross_entropy",
                 max_bits = 64,
                 min_bits = 2,
                 bleaching = True,
                 memory_is_cumulative=True,
                 defaul_b_bleaching=1,
                 max_w = 100,
                 nthread = 4,
                 seed = 0,
                 reg_lambda = 1,
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

        np.random.seed(seed=seed)

        if(not loss in ["cross_entropy", "absolute_error", "squared_error"]):
            raise Exception("Loss Function not defined %s"%(self.__loss))

    def fit(self, X, y, early_stopping_rounds, eval_set=None):
        self.class_ = np.unique(y)
        self.__X = X
        self.__y = y
        self.__eval_set = eval_set
        self.__clfs = []
        n_round = 0
        objective_before = np.inf

        while(n_round < early_stopping_rounds):
            self.__clfs.append(self.__get_clf())
            continue_ = True
            while(continue_):
                X_, y_, X_eval, y_eval = self.__subsample()
                if(len(self.class_) != len(np.unique(y_))):
                    if(verbose):
                        print "Alert: subsample may be too small."
                else:
                    continue_ = False

            self.__clfs[-1].fit(X_, y_)
            if(eval_set == None):
                ypred = self.predict_proba(X_eval)
                obj_value = self.__objective_function(y_eval, ypred)
            else:
                ypred = self.predict_proba(eval_set[0])
                obj_value = self.__objective_function(eval_set[1], ypred)

            if objective_before > obj_value:
                objective_before = obj_value
                n_round = 0

            else:
                n_round += 1
                del(self.__clfs[-1])
            
    def predict_proba(self, X):
        for clf in self.__clfs:
            y_proba += clf.predict_proba(X)
        y_proba/len(self.__clfs)
        return y_proba

    def __subsample(self):
        X_size = len(self.__X)
        fit_size = X_size*self.__subsample
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

    def __loss_function(self, y, ypred):
        if(loss == "cross_entropy"):
            return self.__cross_entropy(y, ypred)
        if(loss == "absolute_error"):
            return self.__absolute_error(y, ypred)
        if(loss == "squared_error"):
            return self.__squared_error(y, ypred)

    def __cross_entropy(self, y, ypred):
        yh = self.__hot_encoder(y)
        summing = 0
        for i in xrange(len(yh)):
            pos = np.argmax(ypred[i])
            summing += y[i][pos] * ypred[i][pos]
        return summing

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
            pos = self.class_.index(y[i])
            yh[i][pos] = 1
        return yh

if __name__ == '__main__':
    fire = OnFiRE()
    y = [1,2,1,1,1,1,2,1,1,1,3]
    fire.class_ = [2,1,3]
    print fire.hot_encoder(y)