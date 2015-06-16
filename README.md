# PyWANN
Python Weightless Artificial Neural Network

## Instructions

## Inputs

Python lists or Python Matrixs (list of list)

ex: 
```python
 [[0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 1, 0, 0, 0, 1, 0],
  [1, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 0, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 1]]
```

### Basic WiSARD
1. Define the number of bits for each memory
2. Define the size of retina
3. Create a Wisard
4. Create discriminators
5. Train with examples
6. Classify

```python
num_bits = 2
retina_size = 64

w = WiSARD(retina_size, num_bits)

w.create_discriminator("A")
w.create_discriminator("T")


# training discriminators
for ex in A_samples:
  w.train("A", ex)

for ex in T_samples:
  w.train("T", ex)

# classifying
A_test = w.classify(A_samples[-1])  
T_test = w.classify(T_samples[-1])  


```

### Bleaching WiSARD
1. Define the number of bits for each memory
2. Define the size of retina
3. Set Vacuum to False
3. Set Bleaching to True
4. Define Confidence Value
5. Set if positions are randomized (default is True)
6. define a default value of b to Bleaching (default b=3)
3. Create a Wisard
4. Create discriminators
5. Train with examples
6. Classify

```python
num_bits = 2
retina_size = 64
use_vacuum = False
use_bleaching = True
confidence_threshold = 0.1
rand_positions = True
default_b = 3

w = WiSARD(retina_size, num_bits, use_vacuum, use_bleaching, confidence_threshold, rand_positions, default_b)

w.create_discriminator("A")
w.create_discriminator("T")


# training discriminators
for ex in A_samples:
  w.train("A", ex)

for ex in T_samples:
  w.train("T", ex)

# classifying
A_test = w.classify(A_samples[-1])  
T_test = w.classify(T_samples[-1])  


```



### Vacuum WiSARD
1. Define the number of bits for each memory
2. Define the size of retina
3. Set Vacuum to True
3. Set Bleaching to False
3. Create a Wisard
4. Create discriminators
5. Train with examples
6. Classify

```python
num_bits = 2
retina_size = 64
use_vacuum = True

w = WiSARD(retina_size, num_bits, use_vacuum)

w.create_discriminator("A")
w.create_discriminator("T")


# training discriminators
for ex in A_samples:
  w.train("A", ex)

for ex in T_samples:
  w.train("T", ex)

# classifying
A_test = w.classify(A_samples[-1])  
T_test = w.classify(T_samples[-1])  


```





