# PyWANN
Python Weightless Artificial Neural Network

## Instructions

## How to Install

pip install git+git://github.com/firmino/PyWANN.git

## Inputs

Python lists or Numpy array.

X is a list of retinas and y is a list of class related with each retina. 

ex: 
```python
 X = [ [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 1],
       [1, 1, 0, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 1]]

y = ['class_a','class_a','class_b','class_b','class_a','class_a','class_b','class_a',]

```

### WiSARD Parameters
1. retina_length: (INT) The length of the retina
2. num_bits_addr: (INT) Number of bits used to build the memories
3. bleaching : (BOOLEAN) If bleaching thechnique is active or not, default value is True
4. confidence_threshold: (FLOAT) Confidence used by bleaching technique to solve tie problems, default is 0.1
5. ignore_zero_addr : (BOOLEAN) The classification of sparse feature vectors can be improved excluding positions zeros. Default value is False
6. defaul_b_bleaching : (INT) The initial value for bleaching technique. Default value is 1
7. randomize_positions:  (BOOLEAN) If the pseudo-random-mapping will be used or not. Default value is True.
8. memory_is_cumulative: (BOOLEAN) If false memories store 0 or 1, if true memories count the number of occurrences of patterns. Default value is True


### Basic WiSARD (Without Bleaching)
1. Define the number of bits for each memory
2. Define the retina's length
3. Set Bleaching to FALSE 
4. Create a Wisard
5. Fit with examples (trainning)
6. Predict (classify)

```python

retina_length = 64
num_bits_addr = 2
bleaching = False

w = WiSARD(retina_length, num_bits_addr, bleaching)


# training discriminators
w.fit(X, y)


# predicting class
result = w.predict(X_test)  #  Result will be a dictionary using the classes as key and the WiSARD result as values




```

### Bleaching WiSARD
1. Define the number of bits for each memory
2. Define the retina's length
3. Set Bleaching to FALSE 
4. Create a Wisard
5. Fit with examples (trainning)
6. Predict (classify)

```python

retina_length = 64
num_bits_addr = 2

w = WiSARD(retina_length, num_bits_addr, bleaching)

# training discriminators
w.fit(X, y)

# predicting class
result = w.predict(X_test)  #  Result will be a dictionary using the classes as key and the WiSARD result as values







