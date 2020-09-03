"""Using a RNN to generate new Pink Floyd lyrics
"""

import tensorflow as tf

import numpy as np
import os

from pf_constants import *

# For now just using a modified version of darkside.txt from textfiles.com
# In the future this will be the lyrics of the band's full catalog.
training_lyrics = open(TRAINING_DATA_PATH).read() # .decode(encoding='utf-8')

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

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("   input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("   expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 12

# Buffer size used to shuffle dataset
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)

# Length of vocabulary in characters
vocab_size = len(vocab)

# Embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
#
# model.summary()
#
# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
#
# print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# Location in which to save checkpoints
checkpoint_dir = './training_checkpoints'

# Name used for checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# model.summary()

def generate_text(model, start_string):
    # Number of characters to generate
    num_generate = 1000

    # Vectorize start string
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []



    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions - predictions / TEMPERATURE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return  start_string + ''.join(text_generated)


print(generate_text(model, start_string=u"BRAIN"))
