from WiSARD import Memory, Discriminator
from copy import deepcopy

class Node:

    def __init__(self, label, is_leaf, discriminator_template):

        self.__label = label
        self.__is_leaf = is_leaf
        self.__discriminator_template = discriminator_template

        if self.__is_leaf:
            self.__discriminator = None
            self.__children = None
            self.__leaf_cluster = []
        else:
            self.__discriminator= deepcopy(discriminator_template)
            self.__children = []
            self.__leaf_cluster = None

        self.__children.append(child)

    def add_trainning(self, retina):

        if self.__is_leaf:

            max_value = 0

            for d in self.__leaf_cluster:


        else:

        

    def classify(self, retina):
        pass