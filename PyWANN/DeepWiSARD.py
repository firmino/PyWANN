from WiSARD import Memory, Discriminator

from copy import deepcopy
import yaml
import numpy as np


class DiscriminatorFactory:

    def __init__(self, 
                 retina_length, 
                 num_bits_addr, 
                 randomize_positions=True):    

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
            memories_template = { i/self.__num_bits_addr: Memory( num_bits_addr    = len(mapping_positions[i/self.__num_bits_addr] ), 
                                                                  is_cummulative   = False,
                                                                  ignore_zero_addr = True)  \
                                  for i in xrange(0, self.__retina_length, self.__num_bits_addr)}

            d =  Discriminator(retina_length = self.__retina_length,
                               mapping_positions = mapping_positions,
                               memories = memories_template)             
            return d


class Cluster:
    
    def __init__(self, 
                 retina_length, 
                 num_bits_addr, 
                 randomize_positions=True, 
                 coverage_threshold=0.8):    
        
        self.__discriminator_factory = DiscriminatorFactory(retina_length=retina_length, 
                                                            num_bits_addr=num_bits_addr, 
                                                            randomize_positions=randomize_positions)
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

        print result
        return result

    def get_num_discriminators(self):
        return len(self.__cluster)


class Node:
    
    def __init__(self, 
                 name, 
                 retina_length, 
                 num_bits_addr, 
                 coverage_threshold=0.8, 
                 randomize_positions=True):

        self.__name     = name
        self.__children = []
        self.__cluster  = Cluster(retina_length=retina_length, 
                                  num_bits_addr=num_bits_addr, 
                                  randomize_positions=randomize_positions, 
                                  coverage_threshold=coverage_threshold)

    def get_name(self):
        return self.__name

    def fit(self, X):
        count = 0
        for retina in X:
            count +=1
            self.__cluster.add_training(retina)

    def predict(self, x):
        return self.__cluster.classify(x)
    
    def add_child(self, node_child):
        self.__children.append(node_child)

    def get_children(self):
        return self.__children

    def get_num_discriminators(self):
        return self.__cluster.get_num_discriminators()


class DeepWiSARD:

    def __init__(self,
                 config_tree,
                 retina_length,
                 num_bits_addr,
                 coverage_threshold=0.8,
                 confidence_threshold=0.2,
                 randomize_positions=True):

        
        self.__retina_length = retina_length
        self.__num_bits_addr = num_bits_addr
        self.__coverage_threshold = coverage_threshold
        self.__confidence_threshold = confidence_threshold
        self.__randomize_positions = randomize_positions
                
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

            if best_value == 0:
                confidence = 0
            else:            
                confidence = 1.0 - second_best_value/float(best_value)
            
            #  if the confidence 
            #if confidence < self.__confidence_threshold:
            #       return result
                #return node.get_name()

            if len( best_node.get_children()) == 0:
                #return result
                return best_node.get_name()

            node = best_node
            

    
    def get_stats(self):
        stats = {}
        for node_name in self.__nodes:
            stats[node_name] = self.__nodes[node_name].get_num_discriminators()
        return stats

    def __create_tree(self, config_tree):

        node_tree = {}

        #  creating the tree structure
        for node_conf in config_tree:

            name = node_conf['name']

            #  creating the node
            node = Node(name,
                        self.__retina_length,
                        self.__num_bits_addr,
                        self.__coverage_threshold,
                        self.__randomize_positions)

            node_tree[name] = node

            #  add node to parent
            if name != 'root':
                parent_name = node_conf['parent']
                node_tree[parent_name].add_child(node)

        return node_tree