"""Get size of vocab for a specific set of training data

Broken out into a separate file to help keep pynk_floyd.py clean
and for easy access from the CLI without having to go through
training.
"""

# TODO resolve issue where this runs pynk

from pynk_floyd import vocab

print('{} unique characters'.format(len(vocab)))
