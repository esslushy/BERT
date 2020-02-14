import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from settings import *
from preprocessing import build_dataset

# Collect pretrained BERT layer. Note BERT takes 3 inputs: tokens, masks, segments
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1', trainable=False)
# Pull out data for BERT and build tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# Build datasets
train_dataset = build_dataset('dataset/training.1600000.processed.noemoticon.csv', tokenizer, MAX_LENGTH, header=None, encoding='latin1')
test_dataset = build_dataset('dataset/testdata.manual.2009.06.14.csv', tokenizer, MAX_LENGTH, header=None, encoding='latin1')

# Shuffle and batch datasets
train_dataset = train_dataset.shuffle(1000000).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(1000000).batch(BATCH_SIZE)

# Construct model
input_token_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='token_ids')
input_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='mask')
segment_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='segment_ids')
# BERT layer
pooled_representation, sequence_representation = bert_layer([input_token_ids, input_mask, segment_ids])
# Output dense layer. Uses pooled_output as that is for the entire sentence
output = Dense(1, name='sentiment_predictions')(pooled_representation)
# Construct Model
model = Model(inputs=[input_token_ids, input_mask, segment_ids], outputs=output)

# Compile the model. Adam optimizer and Kullback-Leibler divergence loss which is good for probabilities like probability of postive or negative
model.compile('adam', loss=tf.keras.losses.KLDivergence(), metrics=[tf.keras.metrics.Accuracy()])

# Train the model
model.fit(train_dataset, epochs=EPOCH, validation_data=test_dataset)
