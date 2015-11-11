from WiSARD import Memory, Discriminator

from copy import deepcopy
import yaml
import numpy as np


class DiscriminatorFactory:

    def __init__(self, retina_length, num_bits_addr, randomize_positions=True):    
        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr
        self.__randomize_positions = randomize_positions


    def generate_discriminator(self):
        return self.__create_discriminator()


    def __create_discriminator(self):

            #  creating mapping positions
            positions = np.arange(self.__retina_length)
            if self.__randomize_positions:
                np.random.shuffle(positions)  # random positions 
         
             #  spliting positions for each memory
            mapping_positions = { i/self.__num_bits_addr : positions[i: i + self.__num_bits_addr] \
                                  for i in xrange(0, self.__retina_length, self.__num_bits_addr) }

            #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
            memories_template = { i/self.__num_bits_addr:  Memory( num_bits_addr = len(mapping_positions[i/self.__num_bits_addr] ), 
                                                                   is_cummulative = False,
                                                                   ignore_zero_addr = False)  \
                                  for i in xrange(0, self.__retina_length, self.__num_bits_addr)}

            d =  Discriminator(retina_length = self.__retina_length,
                               mapping_positions = mapping_positions,
                               memories = memories_template)             
            return d



class Cluster:
    
    def __init__(self, retina_length, num_bits_addr, randomize_positions=True, coverage_threshold=0.8):    
        
        self.__discriminator_factory = DiscriminatorFactory(retina_length, num_bits_addr, randomize_positions)
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

            if coverage >= self.__coverage_threshold:
                d.add_training(retina)
                trained = True

        if not trained: 
            d = self.__discriminator_factory.generate_discriminator()
            d.add_training(retina)
            self.__cluster.append(d)

        
    def classify(self, retina):

        result = 0

        #  looking for the best discriminator on cluster
        for d in self.__cluster:
            #  mem result is a list of binaries numbers
            mem_result = d.classify(retina) 
            new_result = mem_result.sum()

            if result < new_result:
                result = new_result
        return result



class Node:
    
    def __init__(self, name, retina_length, num_bits_addr, randomize_positions=True, coverage_threshold=0.8):
        self.__name     = name
        self.__children = []
        self.__cluster  = Cluster(retina_length, num_bits_addr, randomize_positions, coverage_threshold)

    def get_name(self):
        return self.__name

    def fit(self, X):
        for retina in X:
            self.__cluster.add_training(retina)

    def predict(self, x):
        self.__cluster.classify(x)
    
    def add_child(self, node_child):
        self.__children.append(node_child)

    def get_children(self):
        return self.__children

    

class DeepWiSARD:

    def __init__(self, config_path, num_bits, coverage_threshold, confidence_threshold=0.2):

        self.__config_path = config_path
        self.__confidence_threshold = confidence_threshold
        self.__nodes = {}
        

       

    def fit(self, X, y):
        pass
        # for i in xrange(len(y)):
        #     retina = X[i]
        #     label = y[i]
        #     self.__node_leaf_cluster[label].add_training(retina)

    def predict(self, retina):
        pass
        # for node_leaf_name in self.__leaf_index:

        #     value = self.__node_leaf_cluster[node_leaf_name].classify(retina)

        #     leaf_node = self.__node_leaf_index[node_leaf_name]
        #     leaf_node.set_value(value)
        #     self.__up_information(leaf_node)

        # node_result, confidence = self.__down_selecting(self.__root)
        # self.__reset_tree()

        # return node_result, confidence

   
        
    def __create_tree(self, config_path):

        #  creating the tree structure
        config_file = open(config_path)
        config = yaml.load(config_file)
        nodes_conf = config['config']
        base_path = nodes_conf['base_path']
        
        for node_conf in nodes_conf['tree']:
            print "NAME: ", node_conf['name']
            print "PATH: ", base_path+node_conf['path']
            print "PARENT: ", node_conf['parent']
            print "-"*20


# class DeepWiSARD:

#     def __init__(self, 
#                  treeConfigPath,
#                  retina_length,
#                  num_bits_addr,
#                  coverage_threshold=0.6,
#                  randomize_positions=True):
        
#         self.__children_list = {}
#         self.__tree = {}
#         self.__coverage_threshold = 0.6


#         # ################################################CREATING TEMPLATE OF DISCRIMINATORS###################################
#         # #  generationg mapping positions
#         # positions = np.arange(retina_length)
#         # if randomize_positions:
#         #     np.random.shuffle(positions)  # random positions 
         
#         #  #  spliting positions for each memory
#         # mapping_positions = { i/num_bits_addr : positions[i: i + num_bits_addr] \
#         #                              for i in xrange(0, retina_length, num_bits_addr) }

#         # #  num_bits_addr is calculate based that last memory will have a diferent number of bits (rest of positions)
#         # memories_template = { i/num_bits_addr:  Memory( num_bits_addr = len(mapping_positions[i/num_bits_addr] ), 
#         #                                                 is_cummulative = False,
#         #                                                 ignore_zero_addr = False)  \
#         #                       for i in xrange(0,retina_length, num_bits_addr)}


#         # self.__discriminator_template = Discriminator(retina_length = retina_length,
#         #                                               mapping_positions = mapping_positions,
#         #                                               memories = memories_template) 
#         # ######################################################################################################################

#         # def get_confidence(self):        
#         # if self.__best_value == 0:
#         #     return 0
#         # return 1.0 - float(self.__second_best_value)/float(self.__best_value)


#         # #################################################GENERATING THE TREEE ################################################

#         # config_file = open(treeConfigPath)
#         # config = yaml.load(config_file)
#         # node = config['config']
#         # tree = self.__generate_tree(node)

#         ######################################################################################################################


#     # def __generate_tree(self, node_conf):

#     #     label = node_conf['name']
#     #     is_leaf = True
#     #     if 'children' in node_conf: 
#     #         is_leaf = False

#     #     node = Node(label=label,
#     #                 is_leaf=is_leaf,
#     #                 discriminator_template=self.__discriminator_template,
#     #                 coverage_threshold=self.__coverage_threshold)
                
#     #     if 'children' not in node_conf:
#     #         self.__children_list [label] = node
#     #         return node

#     #     for child in node_conf['children']:
#     #         child_node = self.__generate_tree(child)
#     #         node.add_child(child_node)


#     #     return node


#     # def fit(self, X, y):
#     #     num_samples =  len(y)
#     #     for i in xrange(num_samples):
#     #         retina = X[i]
#     #         label = y[i]
#     #         self.__children_list[label].add_training(retina)

