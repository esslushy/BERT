import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from settings import BATCH_SIZE, EPOCHS, SEED, MODE, TENSORBOARD_LOCATION, MODEL_LOCATION
from model import build_model

# Setup random seed and AUTOTUNER
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)

def load_data():
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
    return train_dataset, validation_dataset, test_dataset, class_names

if (MODE == 'view'):
    train_dataset, validation_dataset, test_dataset, class_names = load_data()
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
    train_dataset, validation_dataset, test_dataset, class_names = load_data()
    print('Building model.')
    classifier_model = build_model()
    print(classifier_model.summary())
    # Losses and metrics for the model
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()
    # Ready adam optimizer
    optimizer = tf.keras.optimizers.Adam(2e-5)
    # Compile model
    classifier_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    # Prepare model callbacks (done at the end of each training loop)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOCATION)]
    # Train model
    print('Started Training')
    classifier_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print('Finished Training')
    # Evaluate final model
    print('Evaluating Model')
    loss, metrics = classifier_model.evaluate(test_dataset)
    # Print out results
    print(f'Loss: {loss}')
    print(f'Accuracy: {metrics}')
    # Save the model
    print(f'Saving the model to {MODEL_LOCATION}')
    classifier_model.save(MODEL_LOCATION, include_optimizer=False)

if (MODE=='run'):
    print('Running the model')
    # Reload the model
    classifier_model = tf.saved_model.load(MODEL_LOCATION)
    while (True):
        # Get user input
        user_sentence = input('Write a sentence for the model to analyze!\nOtherwise type q to quit.\n')
        # Proces input
        if user_sentence == 'q':
            break
        # Get models prediction
        sentiment_score = classifier_model([user_sentence])
        # Print out the results (we know closer to 1 is a positive rating)
        print(f'Input: {user_sentence} ---- Score: {sentiment_score[0][0]} ({"pos" if sentiment_score[0][0] > 0.5 else "neg"})')
