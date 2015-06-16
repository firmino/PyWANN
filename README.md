# PyWANN
Python Weightless Artificial Neural Network

## Instructions

### Basic WiSARD
* Define the number of bits for each memories
* Define the size of retina
* Create a Wisard
* Create discriminators
* Train with examples
* Classify

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







retina_size,
                 num_bits_addr=2,
                 vacuum=False,
                 bleaching=False,
                 confidence_threshold=0.6,
                 randomize_positions=True,
                 default_bleaching_b_value=3):
