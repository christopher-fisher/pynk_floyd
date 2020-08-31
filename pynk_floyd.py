"""Using a RNN to generate new Pink Floyd lyrics


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
# for darkside.txt n = 53

# Create a mapping between the unique characters and their indices.
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in training_lyrics])

# Maximum length for single input (in characters)
seq_length = 100
examples_per_epoch = len(training_lyrics)//(seq_length+1)

# Create training examples and targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#for i in char_dataset.take(5):
#    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#for item in sequences.take(5):
#    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))
