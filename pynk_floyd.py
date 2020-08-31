"""Using a DNN to generate new Pink Floyd lyrics


"""

import tensorflow as tf

import numpy as np
import os
import time

# For now just using a modified version of darkside.txt from textfiles.com
# In the future this will be the lyrics of the band's full catalog.
training_file_path = "C:\\Users\\daeur\\PycharmProjects\\pynk_floyd\\Training Data\\darkside.txt"

training_lyrics = open(training_file_path).read() # .decode(encoding='utf-8')

# Number of unique characters in the training file
vocab = sorted(set(training_lyrics))

# Not needed once this gets worked out
#print('{} unique characters'.format(len(vocab)))
#for darkside.txt n = 53

# Create a mapping between the unique characters and their indices.
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in training_lyrics])

# Print the mapping that was generated
# Not needed, just helpful to see
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

