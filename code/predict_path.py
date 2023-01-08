import operator

import setup

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


def visualize_distribution(training_data, test_data):
    """
    takes: DataFrames
    returns: -
    Plot the number of different paths in both training and test data
    """
    training_paths_split = training_data['Path'].dropna()
    training_paths_split.apply(operator.itemgetter(
        0)).value_counts().plot(kind='bar')
    plt.show()
    plt.clf()
    training_paths_split_newsroom = pd.Series(
        [path for path in training_paths_split if path[0] == 'newsroom'])
    training_paths_split_newsroom.apply(
        operator.itemgetter(1)).value_counts().plot(kind='bar')
    plt.show()
    plt.clf()

    test_paths_split = test_data['Path'].apply(setup.clean_path).dropna()
    test_paths_split.apply(operator.itemgetter(
        0)).value_counts().plot(kind='bar')
    plt.show()
    plt.clf()
    test_paths_split_newsroom = pd.Series(
        [path for path in test_paths_split if path[0] == 'newsroom'])
    test_paths_split_newsroom.apply(
        operator.itemgetter(1)).value_counts().plot(kind='bar')
    plt.show()
    plt.clf()


def make_classifier_bayes():
    """
    takes: -
    returns: Pipelines for classifying into the paths
    """
    classifier_first = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])
    classifier_second = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])
    return classifier_first, classifier_second


def make_classifier_bayes_custom():
    """
    takes: -
    returns: Pipelines for classifying into the paths
    """
    classifier_first = Pipeline([
        ('vect', CountVectorizer(tokenizer=setup.custom_preprocess_html)),
        ('clf', MultinomialNB()),
    ])
    classifier_second = Pipeline([
        ('vect', CountVectorizer(tokenizer=setup.custom_preprocess_html)),
        ('clf', MultinomialNB()),
    ])
    return classifier_first, classifier_second


def make_classifier_dummy():
    """
    takes: -
    returns: Pipelines for classifying into the paths
    """
    classifier_first = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', DummyClassifier(strategy="stratified")),
    ])
    classifier_second = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', DummyClassifier(strategy="stratified")),
    ])
    return classifier_first, classifier_second


def build_confusion_matrix(classifier_second, test_data_second):
    """
    takes: Classifier Pipeline, DataFrame with test data
    returns: -
    """
    # Confusion Matrix for second part
    plot_confusion_matrix(
        classifier_second, test_data_second[EVALUATE_USING], test_data_second['Path'])
    plt.show()


def perform_grid_search(classifier_second, training_data_second, test_data_second):
    """
    takes: Classifier Pipeline, DataFrame with training data, DataFrame with test data
    returns: -
    """
    parameters = {
        'vect__binary': [True, False],  # set-of-words or bag-of-words
        # consider only the most frequent features
        'vect__max_features': [10000, 1000, None],
        'vect__max_df': [1.0, 0.5, 0.1],  # find corpus-specific stop-words
        'vect__ngram_range': [(1, 1), (1, 2)],  # bigrams and unigrams
        'clf__alpha': [1, 0.5, 0.1, 0.01]  # smoothing parameters
    }
    searcher = GridSearchCV(classifier_second, parameters)
    searcher.fit(
        training_data_second[EVALUATE_USING], training_data_second['Path'])
    print(searcher.best_params_)
    prediction_search = searcher.best_estimator_.predict(
        test_data_second[EVALUATE_USING])
    print(classification_report(test_data_second['Path'], prediction_search))


if __name__ == '__main__':
    training_data, test_data = setup.split_data()

    # visualize distribution of test and training data
    visualize_distribution(training_data, test_data)

    # Prepare data sets
    training_data_first = training_data.copy(deep=True)
    training_data_first['Path'] = training_data_first['Path'].apply(
        operator.itemgetter(0))
    test_data_first = test_data.copy(deep=True)
    test_data_first['Path'] = test_data_first['Path'].apply(
        operator.itemgetter(0))

    training_data_second = training_data.copy(deep=True)
    training_data_second['Path'] = pd.Series(
        [path[1] if path[0] == 'newsroom' else None for path in training_data_second['Path']]).values
    training_data_second = training_data_second.dropna(subset=['Path'])

    test_data_second = test_data.copy(deep=True)
    test_data_second['Path'] = pd.Series(
        [path[1] if path[0] == 'newsroom' else None for path in test_data_second['Path']]).values
    test_data_second = test_data_second.dropna(subset=['Path'])

    EVALUATE_USING = 'Body'

    classifier_first, classifier_second = make_classifier_bayes()
    classifier_first_dummy, classifier_second_dummy = make_classifier_dummy()

    classifier_first.fit(
        training_data_first[EVALUATE_USING], training_data_first['Path'])
    prediction_first = classifier_first.predict(
        test_data_first[EVALUATE_USING])
    print(classification_report(
        test_data_first['Path'], prediction_first, zero_division=0))

    classifier_second.fit(
        training_data_second[EVALUATE_USING], training_data_second['Path'])
    prediction_second = classifier_second.predict(
        test_data_second[EVALUATE_USING])
    print(classification_report(
        test_data_second['Path'], prediction_second, zero_division=0))

    classifier_first_custom, classifier_second_custom = make_classifier_bayes_custom()

    classifier_first_custom.fit(
        training_data_first[EVALUATE_USING], training_data_first['Path'])
    prediction_first_custom = classifier_first_custom.predict(
        test_data_first[EVALUATE_USING])
    print(classification_report(
        test_data_first['Path'], prediction_first_custom, zero_division=0))

    classifier_second_custom.fit(
        training_data_second[EVALUATE_USING], training_data_second['Path'])
    prediction_second_custom = classifier_second_custom.predict(
        test_data_second[EVALUATE_USING])
    print(classification_report(
        test_data_second['Path'], prediction_second_custom, zero_division=0))

    classifier_first_dummy, classifier_second_dummy = make_classifier_dummy()

    classifier_first_dummy.fit(
        training_data_first[EVALUATE_USING], training_data_first['Path'])
    prediction_first_dummy = classifier_first_dummy.predict(
        test_data_first[EVALUATE_USING])
    print(classification_report(
        test_data_first['Path'], prediction_first_dummy, zero_division=0))

    classifier_second_dummy.fit(
        training_data_second[EVALUATE_USING], training_data_second['Path'])
    prediction_second_dummy = classifier_second_dummy.predict(
        test_data_second[EVALUATE_USING])
    print(classification_report(
        test_data_second['Path'], prediction_second_dummy, zero_division=0))

    build_confusion_matrix(classifier_second, test_data_second)
    perform_grid_search(classifier_second,
                        training_data_second, test_data_second)
