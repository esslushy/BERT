# Number of epochs
EPOCHS=150
# Batch Size
BATCH_SIZE=32
# Random Seed
SEED=7
# Mode to run in. 
# 'view' shows you some of the data we are working with.
# 'train' trains the model
# 'run' runs the model on user input
MODE='train'
# URL to the BERT model and Preprocessing model you wish to use
BERT_URL='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
PREPROCESSING_URL='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'