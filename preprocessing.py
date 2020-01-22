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