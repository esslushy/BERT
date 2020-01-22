import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from settings import *
from preprocessing import process_input

# Collect pretrained BERT layer. Note BERT takes 3 inputs: tokens, masks, segments
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1', trainable=False)
# Pull out data for BERT and build tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# Construct model
input_token_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='token_ids')
input_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='mask')
segment_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='segment_ids')
# BERT layer
pooled_output, sequence_output = bert_layer([input_token_ids, input_mask, segment_ids])
# Construct Model
model = Model(inputs=[input_token_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
