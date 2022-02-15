import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from settings import BATCH_SIZE, EPOCHS, SEED, MODE
from model import build_model

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

if (MODE == 'view'):
    print('Viewing some data and how it is preprocessed.')
    # Take one batch of the training data
    for text_batch, label_batch in train_dataset.take(1):
        # Run through all 32 data points in the batch
        for i in range(len(text_batch)):
            # Print data
            print(f'Review: {text_batch.numpy()[i]}')
            label = label_batch.numpy()[i]
            print(f'Label: {label} ({class_names[label]})')
    # Load BERT modeland its preprocessing model
    bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    # Experimenting with preprocessed data
    text_test = ['this is such an amazing movie!']
    text_preprocessed = bert_preprocess_model(text_test)
    # Show preprocessing results
    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

if (MODE == 'train'):
    print('Building model.')
    classifier_model = build_model()
    print(classifier_model.summary())