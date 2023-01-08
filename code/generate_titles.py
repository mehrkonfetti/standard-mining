import conf

import string
import numpy as np
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
import tensorflow as tf
from numpy.random import seed
import sqlite3

tokenizer = Tokenizer()

# Code structure taken and adapted from: https://www.analyticsvidhya.com/blog/2021/09/building-a-machine-learning-model-for-title-generation/


def get_articles(input_corpus):
    """
    takes: the sqlite3 corpus
    returns: a list of articles
    """
    query_fetch_articles = 'SELECT * FROM Articles;'  # extract the full Articles table
    cursor_corpus = input_corpus.cursor()
    cursor_corpus.execute(query_fetch_articles)
    return cursor_corpus.fetchall()  # return list of Rows


def clean_text(text):
    """
    A minimal preprocessing pipeline to keep the processing time as low as possible
    """
    text = ''.join(e for e in text if e not in string.punctuation).lower()
    text = text.encode('utf8').decode('ascii', 'ignore')
    return text


def index_to_word(predicted):
    for word, index in tokenizer.word_index.items():
        # iterate over all words in the dictionary and check if they match the calculated words
        if index == predicted:
            return word


def get_sequence_of_tokens(titles_text):
    """
    Transforms the text into a list of dictionary indices
    """
    tokenizer.fit_on_texts(titles_text)
    total_words = len(tokenizer.word_index) + 1

    # convert to sequence of indices in dictionary
    input_sequences = []
    for line in titles_text:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


def generate_padded_sequences(input_sequences):
    """
    Make all sequences the same length
    """
    max_sequence_len = max([len(x) for x in input_sequences])
    # make all sequences the same length, padding from the front
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))
    # label is the last entry
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    # transform into sparse matrix where only label word of sequence is 1 and rest 0
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


def create_model(max_sequence_len, total_words):
    """
    Create the neural network model with its layers
    """
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    # Input size total_words, output size 10
    model.add(Embedding(total_words, 10, input_length=input_len))

    # Add Hidden Layer 1 — LSTM Layer
    # Dimension of output space: 400
    model.add(LSTM(400, return_sequences=True))
    # Randomly sets input units to 0 with frequency of 0.1
    model.add(Dropout(0.1))

    # Add Hidden Layer 1 — LSTM Layer
    # Dimension of output space: 400
    model.add(LSTM(400))
    # Randomly sets input units to 0 with frequency of 0.1
    model.add(Dropout(0.1))

    # Add Output Layer
    # Densely connected NN layer
    # Output again has dimensionality of total_words
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.summary()
    return model


def generate_text(seed_text, next_words, model, max_sequence_len):
    """
    Generate the new title
    seed_text: Keyword to start with
    next_words: How many words to generate
    """
    for x in range(next_words):
        # current state of text -> previously generated words
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=(
            max_sequence_len - 1), padding='post')
        predicted_x = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_x, axis=1)

        output_word = index_to_word(predicted)
        seed_text += ' ' + output_word
    return seed_text.title()


if __name__ == '__main__':
    tf.random.set_seed(2)
    seed(1)
    connection_corpus = sqlite3.connect(
        conf.CORPUSDB)  # load corpus database
    connection_corpus.row_factory = sqlite3.Row
    articles = get_articles(connection_corpus)

    titles = [clean_text(article['Title']) for article in articles]

    input_sequences, total_words = get_sequence_of_tokens(titles)

    predictors, label, max_sequence_len = generate_padded_sequences(
        input_sequences)

    model = create_model(max_sequence_len, total_words)
    # Input data: predictors
    # Target data: label
    model.fit(predictors, label, epochs=30, verbose=2)

    print(generate_text('Politik', 5, model, max_sequence_len))
    print(generate_text('Österreich', 5, model, max_sequence_len))
    print(generate_text('Russland', 5, model, max_sequence_len))
    print(generate_text('Flüchtlinge', 10, model, max_sequence_len))
    print(generate_text('Kurz', 10, model, max_sequence_len))
    print(generate_text('Politik', 10, model, max_sequence_len))
