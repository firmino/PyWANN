from WiSARD import Memory, Discriminator

from copy import deepcopy
import yaml
import numpy as np


class Cluster:
    
    def __init__(self,discriminator_template, b_value=1, coverage_threshold=0.6):

        self.__discriminator_template = discriminator_template
        self.__b_value = b_value
        self.__coverage_threshold = coverage_threshold

        




class Node:
    pass

class Tree:
    pass











class Node:

    def __init__(self, label, is_leaf, discriminator_template, b_value=1, coverage_threshold=0.6):

        self.__label = label
        self.__is_leaf = is_leaf
        self.__parent = None
        self.__discriminator_template = discriminator_template

        if self.__is_leaf:
            self.__children = None
            self.__leaf_cluster = []
        else:
            self.__children = []
            self.__leaf_cluster = None


    def set_parent(self, parent):
        self.__parent = parent

    def get_parent(self, parent):
        return self.__parent

    def get_label(self):
        return self.__label

    def add_child(self, child):
        if not self.__is_leaf:
            self.__children.append(child)

    def get_children(self):
        return self.__children

    def add_training(self, retina):
        if self.__is_leaf:

            trained = False
            for d in self.__leaf_cluster:
                
                mem_result = d.classify(retina)
                
                num_memories_accessed = mem_result[mem_result >= b_value].size
                num_memories = mem_result.size

                #  coverage is the percentual of memories that recognize the pattern
                coverage = num_memories_accessed/float(num_memories)

                if coverage >= coverage_threshold:
                    d.add_training(retina)
                    trained = True

            if not trained: 
                d = deepcopy(self.__discriminator_template)
                d.add_training(retina)
                self.__leaf_cluster.append(d)

        
    def classify(self, retina):
        
        if self.__is_leaf:

            max_coverage = 0.0
            result = []

            for d in self.__leaf_cluster:

                mem_result = d.classify(retina)

                num_memories_accessed = mem_result[mem_result >= b_value].size
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
                 num_bits_addr,
                 retina_length,
                 b_value=1,
                 randomize_positions=True,
                 memory_is_cumulative=False,
                 ignore_zero_addr=False,
                 coverage_threshold=0.6):
        
        self.__children_list = {}
        self.__tree = {}
        self.__b_value = b_value
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
                                                        is_cummulative = memory_is_cumulative,
                                                        ignore_zero_addr = ignore_zero_addr)  \
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

