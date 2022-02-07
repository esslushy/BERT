import tensorflow as tf
import os

def get_dataset():
    """
      Downloads the dataset if it doesn't already exist on the machine.
      Gives back the directory of where the data is stored
    """
    dataset = tf.keras.utils.get_file(
        os.path.abspath('./dataset/aclImdb_v1.tar.gz'),
        'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        extract=True, cache_dir='', cache_subdir=''
    )
    print(dataset)
    # return dataset_dir, train_dir

get_dataset()