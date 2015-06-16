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
