"""Using a RNN to generate new Pink Floyd lyrics trained on their discography
through The Division Bell.

Adapted from the TensorFlow Shakespeare text generation tutorial:
https://www.tensorflow.org/tutorials/text/text_generation

Training data adopted from textfiles.com, AZlyrics.com and
pink-floyd-lyrics.com
"""

# TODO Clean up extraneous code from tutorial

# TODO General PEP8 cleanup and tweaks for readibility

import tensorflow as tf

import numpy as np
import os
import time

from pf_constants import *

# Set in pf_constants.py
# Using the lyrics of Dark Side of the Moon for now to get everything tuned
# This will be replaced with a much larger training file later
training_lyrics = open(TRAINING_DATA_PATH).read()

# Number of unique characters in the training file
vocab = sorted(set(training_lyrics))
# for darkside.txt n = 53

start_time = time.strftime("%m/%d/%Y, %H:%M:%S")

# Create a mapping between the unique characters and their indices.
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in training_lyrics])

# Maximum length for single input (in characters)
seq_length = 100
examples_per_epoch = len(training_lyrics)//(seq_length+1)

# Create training examples and targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

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
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# Location in which to save checkpoints
checkpoint_dir = './training_checkpoints'

# Name used for checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


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

        predictions = predictions / TEMPERATURE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# Use for generation of a single "song"
#print(generate_text(model, start_string=u"BRAIN"))


# Used to help avoid collision
def get_timestamp():
    return str(time.time())


timestamp = get_timestamp()

# Create output directory name
outdir_name =  PROJECT_NAME + '-' + timestamp

outdir_path = './Intermediary results/' + outdir_name

# make the directory
os.mkdir(outdir_path)

log_name = outdir_path + '/log-' + timestamp + '.txt'

for i in range(len(OUTPUT_SEEDS)):
    # Get a new timestamp for each file to avoid collision
    # in the case of repeated seeds
    outfile_name = OUTPUT_SEEDS[i] + '-' + get_timestamp()
    file_name = outdir_path + '/' + outfile_name + ".txt"
    with open(file_name, "w") as f:
        f.write(generate_text(model, start_string=OUTPUT_SEEDS[i]))

end_time = time.strftime("%m/%d/%Y, %H:%M:%S")

with open(log_name, "w") as l:
    l.write("Project: " + PROJECT_NAME + '-' + timestamp + '\n')
    l.write("Seeds: " + str(OUTPUT_SEEDS) + '\n')
    l.write("Training data: " + TRAINING_DATA_PATH + '\n')
    l.write("Epochs: " + str(EPOCHS) + '\n')
    l.write("Temperature: " + str(TEMPERATURE) + '\n')
    l.write("Start time: " + start_time + '\n')
    l.write("Finish time: " + end_time + '\n')
# Not currently implemented:
#    l.write("Model information: \n")






