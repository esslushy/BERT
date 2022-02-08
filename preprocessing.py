import tensorflow as tf
import shutil
import os

def get_dataset():
    """
      Downloads the dataset if it doesn't already exist on the machine.
      Gives back the directory of where the data is stored
    """
    # Download dataset
    dataset = tf.keras.utils.get_file(
        os.path.abspath('./dataset/aclImdb_v1.tar.gz'),
        'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        untar=True, cache_dir='', cache_subdir=''
    )
    # Remove folders we will not be using
    shutil.rmtree(os.path.join(dataset, 'aclImdb', 'train', 'unsup'))
    # Return root dataset location
    return dataset