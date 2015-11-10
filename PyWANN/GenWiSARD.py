from WiSARD import Memory, Discriminator

from copy import deepcopy
import yaml
import numpy as np


class Node:
    
    def __init__(self, name, parent):
        self.__name   = name
        self.__parent = parent
        self.__children = []
        self.__value = 0

    def get_parent(self):
        return self.__parent

    def add_child(self, child_node):
        self.__children.append(child_node)

    def get_children(self):
        return self.__children

    def set_value(self, value):
        self.__value = value

    def get_value(self):
        return self.__value

    def reset_value(self):
        self.__value = 0

class Tree:
    
    def __init__(self, tolerance_dif=0.05):

        self.__tolerance_dif = tolerance_dif
        self.__root = Node(name = "root", parent=None)
        self.__node_leaf_index = {}  # leafs have clusters of WiSARDS, so is necessary be indexed to receive classification values 

    def add_child(self, parent, child_name, is_leaf=False):

        node = Node(name=child_name, parent= parent)
        parent.add_child(node)
        if is_leaf:
            self.__node_leaf_index[child_name] = node

    def predict(self, results):

        #  each leaf will receive the data from clusters (results)
        #  result is a dictionary, using the node name as key and best acuracy for the classification. 
        #  ex: result = {"leaf_a":12, "leaf_b":5, "leaf_c":12, "leaf_d":3}
        for leaf_name in results:
            node = self.__node_leaf_index[leaf_name]
            value = results [leaf_name]
            node.add_value( value )

            # propagated values to parents
            self.__up_value(node.get_parent(), value)

        #  choosing the bast node to represent the class

    def __up_value(self, node, value):
        
        if node.get_value() < value:
            node.set_value(value)
            self.__up_data(node.get_parent(), value)

    def __down_classifying(self, node):

        best_child = None
        best_value = 0
        node_tied = False

        for child in node.get_children():

            #  if exist two son of a node that have similar values
            if best_child != None and np.isclose(best_child.get_value(), child.get_value(), rtol=self.__tolerance_dif):
                return node

            if child.get_value() > best_value:
                best_value = child.get_value()
                best_child = child

        #  if the best_child has not children, its a leaf
        if len(best_child.get_children()) == 0:
            return best_child

        self.__down_classifying(best_child)  # recursion


class Cluster:
    
    def __init__(self, discriminator_template, coverage_threshold=0.6):

        self.__discriminator_template = discriminator_template
        self.__b_value = b_value
        self.__coverage_threshold = coverage_threshold
        self.__cluster = []

    def add_training(self, retina):
        
        trained = False

        for d in self.__cluster:
            
            mem_result = d.classify(retina)            
            num_memories_accessed = mem_result.sum()
            num_memories = mem_result.size

            #  coverage is the percentual of memories that recognize the pattern
            coverage = num_memories_accessed/float(num_memories)

            if coverage >= coverage_threshold:
                d.add_training(retina)
                trained = True

        if not trained: 
            d = deepcopy(self.__discriminator_template)
            d.add_training(retina)
            self.__cluster.append(d)

        
    def classify(self, retina):

        max_coverage = 0.0
        result = []

        for d in self.__cluster:

            mem_result = d.classify(retina)

            num_memories_accessed = mem_result.sum()
            num_memories = mem_result.size

            #  coverage is the percentual of memories that recognize the pattern
            coverage = num_memories_accessed/float(num_memories)

            if coverage > max_coverage:
                max_coverage = coverage
                result = mem_result

        return result


class GenWiSARD:

    def __init__(self, 
                 treeConfigPath,
                 retina_length,
                 num_bits_addr,
                 coverage_threshold=0.6,
                 randomize_positions=True):
        
        self.__children_list = {}
        self.__tree = {}
        self.__coverage_threshold = 0.6


        ################################################CREATING TEMPLATE OF DISCRIMINATORS###################################
        #  generationg mapping positions
        positions = np.arange(retina_length)
        if randomize_positions:
            np.random.shuffle(positions)  # random positions 
         
         #  spliting positions for each memory
        mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
                                     for i in xrange(0, retina_length, num_bits_addr) }

        #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
        memories_template = { i/num_bits_addr:  Memory( num_bits_addr = len(mapping_positions[i/num_bits_addr] ), 
                                                        is_cummulative = False,
                                                        ignore_zero_addr = False)  \
                              for i in xrange(0,retina_length, num_bits_addr)}


        self.__discriminator_template = Discriminator(retina_length = retina_length,
                                                      mapping_positions = mapping_positions,
                                                      memories = memories_template) 
        ######################################################################################################################



        #################################################GENERATING THE TREEE ################################################

        config_file = open(treeConfigPath)
        config = yaml.load(config_file)
        node = config['config']
        tree = self.__generate_tree(node)

        ######################################################################################################################


    def __generate_tree(self, node_conf):

        label = node_conf['name']
        is_leaf = True
        if 'children' in node_conf: 
            is_leaf = False

        node = Node(label=label,
                    is_leaf=is_leaf,
                    discriminator_template=self.__discriminator_template,
                    b_value=self.__b_value,
                    coverage_threshold=self.__coverage_threshold)
                
        if 'children' not in node_conf:
            self.__children_list [label] = node
            return node

        for child in node_conf['children']:
            child_node = self.__generate_tree(child)
            node.add_child(child_node)


        return node


    def fit(self, X, y):
        num_samples =  len(y)
        for i in xrange(num_samples):
            retina = X[i]
            label = y[i]
            self.__children_list[label].add_training(retina)

