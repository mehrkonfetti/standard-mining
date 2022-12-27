import setup
import conf

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import operator
import numpy
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


def preprocess(text):
    tokenized = setup.get_lemmas(setup.get_pos_tags(setup.remove_stopwords(setup.tokenize_strings([text]))))[0]
    return [token[0] for token in tokenized]


if __name__ == '__main__': 
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    df = pd.read_sql_query("SELECT * FROM Articles;", connection_corpus)

    # split articles in test and training
    numpy.random.seed(123456)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(df)))
    shuffled_articles = df.iloc[numpy.random.permutation(df.index)].reset_index(drop=True)
    shuffled_articles['Path'] = shuffled_articles['Path'].apply(setup.clean_path)
    shuffled_articles = shuffled_articles[shuffled_articles['Path'].notna()]
    
    training_data = shuffled_articles.iloc[0:7001]
    test_data = shuffled_articles.iloc[7001:12087]
    
    # visualize distribution of test and training data
    if False:
        training_paths_split = training_data['Path'].dropna()
        training_paths_split.apply(operator.itemgetter(0)).value_counts().plot(kind='bar')
        plt.show()
        plt.clf()
        training_paths_split_newsroom = pd.Series([path for path in training_paths_split if path[0] == 'newsroom'])
        training_paths_split_newsroom.apply(operator.itemgetter(1)).value_counts().plot(kind='bar')
        plt.show()
        plt.clf()

        test_paths_split = test_data['Path'].apply(setup.clean_path).dropna()
        test_paths_split.apply(operator.itemgetter(0)).value_counts().plot(kind='bar')
        plt.show()
        plt.clf()
        test_paths_split_newsroom = pd.Series([path for path in test_paths_split if path[0] == 'newsroom'])
        test_paths_split_newsroom.apply(operator.itemgetter(1)).value_counts().plot(kind='bar')
        plt.show()
        plt.clf()

    # Prepare data sets
    training_data_first_item = training_data.copy(deep=True)
    training_data_first_item['Path'] = training_data_first_item['Path'].apply(operator.itemgetter(0))
    test_data_first_item = test_data.copy(deep=True)
    test_data_first_item['Path'] = test_data_first_item['Path'].apply(operator.itemgetter(0))

    training_data_second = training_data.copy(deep=True)
    training_data_second['Path'] = pd.Series([path[1] if path[0] == 'newsroom' else None for path in training_data_second['Path']]).values
    training_data_second = training_data_second.dropna(subset=['Path'])

    test_data_second = test_data.copy(deep=True)
    test_data_second['Path'] = pd.Series([path[1] if path[0] == 'newsroom' else None for path in test_data_second['Path']]).values
    test_data_second = test_data_second.dropna(subset=['Path'])

    EVALUATE = 'Bayes'
    EVALUATE_USING = 'Body'

    # Predicting with bayes
    if EVALUATE == 'Bayes':
        classifier_first = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])
        classifier_second = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])
    else:
        classifier_first = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', DummyClassifier(strategy="stratified")),
        ])
        classifier_second = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', DummyClassifier(strategy="stratified")),
        ])
    classifier_first.fit(training_data_first_item[EVALUATE_USING], training_data_first_item['Path'])
    prediction_first = classifier_first.predict(test_data_first_item[EVALUATE_USING])
    print(classification_report(test_data_first_item['Path'], prediction_first, zero_division=0))

    classifier_second.fit(training_data_second[EVALUATE_USING], training_data_second['Path'])
    prediction_second = classifier_second.predict(test_data_second[EVALUATE_USING])
    print(classification_report(test_data_second['Path'], prediction_second, zero_division=0))

    if False:
        # Confusion Matrix for second part
        plot_confusion_matrix(classifier_second, test_data_second[EVALUATE_USING], test_data_second['Path'])
        plt.show()

    if False:
        parameters = {
            'vect__binary': [True, False],  # set-of-words or bag-of-words
            'vect__max_features': [10000, 1000, None],  # consider only the most frequent features
            'vect__max_df': [1.0, 0.5, 0.1],  # find corpus-specific stop-words
            'vect__ngram_range': [(1, 1), (1, 2)],  # bigrams and unigrams
            'clf__alpha': [1, 0.5, 0.1, 0.01]  # smoothing parameters
        }
        searcher = GridSearchCV(classifier_second, parameters)
        searcher.fit(training_data_second[EVALUATE_USING], training_data_second['Path'])
        print(searcher.best_params_)
        prediction_search = searcher.best_estimator_.predict(test_data_second[EVALUATE_USING])
        print(classification_report(test_data_second['Path'], prediction_search))

