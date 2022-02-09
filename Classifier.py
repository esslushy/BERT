import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from settings import BATCH_SIZE, EPOCHS, SEED

# Setup random seed and AUTOTUNER
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)

# Dataset loading
print('Loading Data')
# Loading train dataset from its directory batched to batch size, with 20% saved for validation
print('Loading training portion of the training data')
raw_train_dataset = tf.keras.utils.text_dataset_from_directory(
    './dataset/aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=SEED
)
# Class names for our data points
class_names = raw_train_dataset.class_names
# Set it up so that it runs faster while we need to access it
train_dataset = raw_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print('Finished loading training portion of the training data')

print('Loading validation protion of the training data')
# Loading validation dataset from its directory batched to batch size using the 20% saved for validation
raw_validation_dataset = tf.keras.utils.text_dataset_from_directory(
    './dataset/aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED
)
# Set it up so that it runs faster while we need to access it
validation_dataset = raw_validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print('Finished loading validation portion of the training data')

print('Loading testing data')
# Loading the testing dataset
raw_test_dataset = tf.keras.utils.text_dataset_from_directory(
    './dataset/aclImdb/test',
    batch_size=BATCH_SIZE
)
# Set it up so that it runs faster while we need to access it
test_dataset = raw_test_dataset.cache().prefetch(buffer_size=AUTOTUNE)