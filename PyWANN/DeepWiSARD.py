import numpy as np


class Node:
    
    def __init__(self, name):

        self.__name     = name
        self.__children = []
        self.__vector_class =None
        
    def get_name(self):
        return self.__name

    def fit(self, X):
        X = np.array(X)
        result = np.sum(X, axis=0)
        
        mean = np.mean(result)
        std = np.std(result)

        result_binarized = np.where(result > (mean-2*std), 1, 0)
        self.__vector_class = result_binarized

    def predict(self, x):
        x = np.array(x)
        diff = x - self.__vector_class
        return np.count_nonzero(diff >= 0)
    
    def add_child(self, node_child):
        self.__children.append(node_child)

    def get_children(self):
        return self.__children


class DeepWiSARD:

    def __init__(self, config_tree, confidence_threshold=0.1):

        self.__confidence_threshold = confidence_threshold
        self.__nodes = self.__create_tree(config_tree)

    def fit(self, X, y):
        self.__nodes[y].fit(X)

    def predict(self, retina):
        
        node = self.__nodes['root']
        result = ['root']
        while True:
            
            best_node = None
            best_value =  0
            second_best_value = 0

            for child_node in node.get_children():

                value = child_node.predict(retina)

                if value > best_value:
                    second_best_value = best_value
                    best_value = value
                    best_node = child_node
                    
                # take care with ties, because i have to choose one node to explorer
                elif value == best_value:
                    best_value = value
                    second_best_value = value
                    best_node = child_node
                    
                elif value < best_value and value > second_best_value:
                    second_best_value = value

            confidence = 0
            if best_value != 0:
                confidence = 1.0 - second_best_value/float(best_value)
            
            #  if the confidence 
            if confidence < self.__confidence_threshold:
                return node.get_name()

            if len( best_node.get_children()) == 0:
                return best_node.get_name()

            node = best_node
            
    def __create_tree(self, config_tree):

        node_tree = {}

        #  creating the tree structure
        for node_conf in config_tree:

            name = node_conf['name']

            #  creating the node
            node = Node(name)

            node_tree[name] = node

            #  add node to parent
            if name != 'root':
                parent_name = node_conf['parent']
                node_tree[parent_name].add_child(node)

        return node_tree