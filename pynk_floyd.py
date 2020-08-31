"""Using a DNN to generate new Pink Floyd lyrics


"""

import tensorflow as tf

import numpy as np
import os
import time

# For now just using a modified version of darkside.txt from textfiles.com
# In the future this will be the lyrics of the band's full catalog.
training_file_path = "C:\\Users\\daeur\\PycharmProjects\\pynk_floyd\\Training Data\\darkside.txt"

training_lyrics = open(training_file_path).read().decode(encoding='utf-8')

#number of unique characters in the training file
vocab = sorted(set(training_lyrics))
print('{} unique characters'.format(len(vocab)))
