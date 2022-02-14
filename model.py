import tensorflow as tf
import tensorflow_hub as hub
from settings import BERT_URL, PREPROCESSING_URL

def build_model():
    """
      Creates the model for sentiment analysis
    """
    # Input into the model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    # Load BERT preprocessing layer from tensorflow hub
    preprocessing_layer = hub.KerasLayer(PREPROCESSING_URL, name='preprocessing')
    # Preprocesss the input
    encoder_inputs = preprocessing_layer(text_input)
    # Load the BERT model. Let it be fine tuned
    encoder = hub.KerasLayer(BERT_URL, trainable=True, name='BERT_encoder')
    # Run it through BERT
    encoder_output = encoder(encoder_inputs)
    # Pull out only the pooled results for the sentence level representation rather than a word by word representation.
    net = encoder_output['pooled_output']
    # Add a dropout layer
    net = tf.keras.layers.Dropout(0.1)(net)
    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(net)
    return tf.keras.Model(text_input, output)