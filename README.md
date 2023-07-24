# BERT Sentiment Classifier
This is a small personal project following the Tensorflow Tutorial on how to use BERT to make a sentiment classifier. I wanted to learn about BERT so I could use it on my own in other personal projects as well as understand why it is so much more effective than anything I can make by hand. Additionally, I am learning about transfer learning (using another model trained on one task to accomplish another task) and seeing how effective it can be. I am extremely interested in how we work with text data with machine learning as it differs from the traditional mode of numeric data input, so I am using this to start learning about the different methods of handling text.

## Dataset
1. Download the [dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and unzip it.
2. Transfer the `aclImdb` folder into a `dataset` folder in the base of this repository. The final directory should look like `./dataset/aclImdb/*`.
3. Delete the `./dataset/aclImdb/train/unsup` folder as it is unnecessary.

## Running
1. Run `pip install -r requirements.txt` to install all required python libraries.
2. Do `python Classifier.py` to run the model
