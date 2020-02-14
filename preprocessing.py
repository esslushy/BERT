import pandas as pd
import tensorflow as tf
import numpy as np

def build_dataset(file_name, tokenizer, max_segment_length, header='infer', encoding='utf-8'):
    """
      Constructs a tensorflow dataset for all the sentence inputs in a classifier problem.

      Args:
        file_name: location of the csv file of data
        tokenizer: the built tokenizer for sentence tokenization
        max_segment_length: maximum length that can be processed
        header: header control for csv file
        encoding: encoding to read the csv file
    """
    # Read data
    df = pd.read_csv(file_name, header=header, encoding=encoding)
    # Extract data and label columns
    labels = df[0].values
    data = df[5].values
    # Normalize labels between -1 (negative) and 1 (positive)
    labels = (labels - (np.amax(labels)/2))/(np.amax(labels)/2)
    data = process_inputs(data, tokenizer, max_segment_length)
    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    return dataset

def process_inputs(sentences, tokenizer, max_segment_length):
    """
      Processes a batch of sentences into correct tokens, masking, and padding.

      Args:
        sentences: a batch of sentence segments to be processed
        tokenizer: the built tokenizer for sentence tokenization
        max_segment_length: maximum length that can be processed
    """
    # Make dictionaries for the batch
    token_ids = []
    mask = []
    segment_ids = []
    for sentence in sentences:
        # Get each individual token, mask, and segment
        t, m, s = process_input(sentence, tokenizer, max_segment_length)
        # Append them to dicts
        token_ids.append(t)
        mask.append(m)
        segment_ids.append(s)
    # Return a dictionary for easy dataset construction
    return {'token_ids': token_ids, 'mask': mask, 'segment_ids': segment_ids}
    

def process_input(sentence, tokenizer, max_segment_length):
    """
      Processes sentence into correct tokens, masking, and padding.

      Args:
        sentence: a single sentence segment to be processed
        tokenizer: the built tokenizer for sentence tokenization
        max_segment_length: maximum length that can be processed
    """
    # Break into tokens
    tokens = tokenizer.tokenize(sentence)
    # Make sure it doesn't break the maximum size - 2 to leave space for special tokens
    if len(tokens) > max_segment_length - 2: 
        # If too large truncate down to size
        tokens = tokens[:max_segment_length-2]
    # Add start and stop tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Return token ids, masks, and segments
    return get_ids(tokens, tokenizer, max_segment_length), get_masks(tokens, max_segment_length), get_segments(max_segment_length)

def get_ids(tokens, tokenizer, max_segment_length):
    """
      Turn the tokens into their corresponding token ids and pad the segment to the correct length.

      Args:
        tokens: the wordpiece tokens of the sentence in an array
        tokenizer: the built tokenizer for sentence tokenization
        max_segment_length: maximum length that can be processed
    """
    # Get token ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Bad to max length
    input_ids = token_ids + [0] * (max_segment_length-len(token_ids))
    return input_ids

def get_masks(tokens, max_segment_length):
    """
      Produce masks for padding.

      Args:
        tokens: the wordpiece tokens of the sentence in an array
        max_segment_length: maximum length that can be processed
    """
    return [1]*len(tokens) + [0] * (max_segment_length - len(tokens))

def get_segments(max_segment_length):
    """
      Make segment lables. Since it is all 1 segment they are all 0.

      Args:
        max_segment_length: maximum length that can be processed
    """
    return [0]*max_segment_length