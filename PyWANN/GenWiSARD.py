from WiSARD import Memory, Discriminator

from copy import deepcopy
import yaml
import numpy as np


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


class Node:
    
    def __init__(self, name):
        self.__name   = name
        self.__parent = None
    
        self.__best_child = None
        self.__best_value = 0
        self.__second_best_value = 0

    def get_name(self):
        return self.__name

    def set_value(self, value):
        self.__best_value = value

    def get_value(self):
        return self.__best_value

    def set_parent(self, parent):
        self.__parent = parent

    def get_parent(self):
        return self.__parent

    def propagate_value(self, node):        
        updated = False
        value = node.get_value()
        if value > self.__best_value:
            self.__second_best_value = self.__best_value
            self.__best_value = value
            self.__best_child = node
            updated = True
        
        elif value == self.__best_value or value > self.__second_best_value:
            self.__second_best_value  = value

        return updated


    def get_best_child(self):
        return self.__best_child

    def get_confidence(self):        
        if self.__best_value == 0:
            return 0
        return 1.0 - float(self.__second_best_value)/float(self.__best_value)

    def reset(self):
        self.__best_child = None
        self.__best_value = 0
        self.__second_best_value = 0


class Tree:

    def __init__(self, config_path, confidence_threshold=0.2):

        self.__confidence_threshold = confidence_threshold
        
        self.__node_leaf_index = {}  # leafs have clusters of WiSARDS, so is necessary be indexed to receive classification values 
        self.__node_leaf_cluster = {}  # each leaf will have a cluster

        self.__root = Node(name = "root")

        #  creating the tree structure
        config_file = open(config_path)
        config = yaml.load(config_file)
        node = config['config']
        self.__create_tree(None, node)  

    def fit(self, X, y):
        for i in xrange(len(y)):
            retina = X[i]
            label = y[i]
            self.__node_leaf_cluster[label].add_training(retina)

    def predict(self, retina):
        for node_leaf_name in self.__leaf_index:

            value = self.__node_leaf_cluster[node_leaf_name].classify(retina)

            leaf_node = self.__node_leaf_index[node_leaf_name]
            leaf_node.set_value(value)
            self.__up_information(leaf_node)

        node_result, confidence = self.__down_selecting(self.__root)
        self.__reset_tree()

        return node_result, confidence

    def __up_information(self, node):
        parent = node.get_parent()
        if parente == None:
            return

        updated = parent.propagate_value(node)
        if updated:
            self.__up_information(parent)
    
    def __down_selecting(self, node):
        if node.get_confidence() < self.__confidence_threshold:  # if confidence is low
            return node.get_name(), node.get_confidence()

        else:
            best_child = node.get_best_child()
            if best_child == None:  # until a leaf
                return node.get_name(), node.get_confidence()
            else:
                self.__down_selecting(best_child)

    def  __reset_tree(self, node):
        node.reset()
        parent = node.get_parent()        
        if parente == None:
            return

        self.__up_information(parent)
        
    def __create_tree(self, parent_node, node_conf):
        
        name = node_conf['name']
        
        if name != 'root':  # if is not the root node, create a new node
            new_node = Node(name=name)
            new_node.set_parent(parent_node)

        else:  # if is the root node
            new_node = self.__root
        
        # if is a leaf
        if 'children' not in node_conf: 
            self.__node_leaf_index[name] = new_node
            return 
        
        for child in node_conf['children']:
            child_node = self.__create_tree(new_node, child)



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


        # ################################################CREATING TEMPLATE OF DISCRIMINATORS###################################
        # #  generationg mapping positions
        # positions = np.arange(retina_length)
        # if randomize_positions:
        #     np.random.shuffle(positions)  # random positions 
         
        #  #  spliting positions for each memory
        # mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
        #                              for i in xrange(0, retina_length, num_bits_addr) }

        # #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
        # memories_template = { i/num_bits_addr:  Memory( num_bits_addr = len(mapping_positions[i/num_bits_addr] ), 
        #                                                 is_cummulative = False,
        #                                                 ignore_zero_addr = False)  \
        #                       for i in xrange(0,retina_length, num_bits_addr)}


        # self.__discriminator_template = Discriminator(retina_length = retina_length,
        #                                               mapping_positions = mapping_positions,
        #                                               memories = memories_template) 
        # ######################################################################################################################



        # #################################################GENERATING THE TREEE ################################################

        # config_file = open(treeConfigPath)
        # config = yaml.load(config_file)
        # node = config['config']
        # tree = self.__generate_tree(node)

        ######################################################################################################################


    # def __generate_tree(self, node_conf):

    #     label = node_conf['name']
    #     is_leaf = True
    #     if 'children' in node_conf: 
    #         is_leaf = False

    #     node = Node(label=label,
    #                 is_leaf=is_leaf,
    #                 discriminator_template=self.__discriminator_template,
    #                 coverage_threshold=self.__coverage_threshold)
                
    #     if 'children' not in node_conf:
    #         self.__children_list [label] = node
    #         return node

    #     for child in node_conf['children']:
    #         child_node = self.__generate_tree(child)
    #         node.add_child(child_node)


    #     return node


    # def fit(self, X, y):
    #     num_samples =  len(y)
    #     for i in xrange(num_samples):
    #         retina = X[i]
    #         label = y[i]
    #         self.__children_list[label].add_training(retina)

