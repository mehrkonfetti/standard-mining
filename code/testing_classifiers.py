import operator

import setup

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    training_data, test_data = setup.split_data()

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

    EVALUATE_USING = 'Title'
    # define classifiers to test
    # structure like in https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    names = [
        "Naive Bayes",
        "Linear SVM",
        "RadialBF SVM",
        "Sigmoid SVM",
        "DecisionTree",
        "RandomForest",
        "MLPerceptron",
        "KNNeighbor",
    ]

    # define parameters
    parameters = [
        {},
        {
            'clf__C': [7.0, 5.0, 3.0],
        },
        {
            'clf__C': [100.0, 10.0, 5.0],
            'clf__gamma': [0.5, 0.1, 0.05],
        },
        {
            'clf__C': [5.0, 1.0, 0.5],
            'clf__gamma': [5.0, 1.0, 0.5],
        },
        {
            'clf__max_depth': [40, 50, 80],
            'clf__min_samples_split': [4, 5, 8],
        },
        {
            'clf__max_depth': [40, 50, 80],
            'clf__n_estimators': [500, 1000, 2000],
        },
        {
            'clf__alpha': [0.5, 0.1, 0.05],
            'clf__max_iter': [350, 400, 600],
        },
        {
            'clf__n_neighbors': [8, 10, 15],
        },
    ]

    classifiers = [
        MultinomialNB(),
        SVC(kernel='linear', class_weight='balanced'),
        SVC(kernel='rbf', class_weight='balanced'),
        SVC(kernel='sigmoid', class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced'),
        MLPClassifier(solver='lbfgs'),
        KNeighborsClassifier(),
    ]
    optimal_classifiers = []

    # for every parameter - classifier pair make searcher and search
    for parameter, classifier in zip(parameters, classifiers):
        pipeline = Pipeline([
            ('vect', TfidfVectorizer(tokenizer=setup.custom_preprocess)),
            ('clf', classifier),
        ])
        searcher = GridSearchCV(pipeline, parameter)
        searcher.fit(
            training_data_second[EVALUATE_USING], training_data_second['Path'])
        optimal_classifiers.append(searcher.best_estimator_)
        print(searcher.best_params_)

    print("Classifier Name\t\tPrecision score\t\t\tRecall score\t\t\tF1 score")

    # for classifier in classifiers: predict and report
    for name, optimal_classifier in zip(names, optimal_classifiers):
        prediction = optimal_classifier.predict(
            test_data_second[EVALUATE_USING])
        precision = precision_score(
            test_data_second['Path'], prediction, average='weighted', zero_division=0)
        recall = recall_score(
            test_data_second['Path'], prediction, average='weighted', zero_division=0)
        f1score = f1_score(
            test_data_second['Path'], prediction, average='weighted', zero_division=0)
        print(f"{name}\t\t{precision}\t\t{recall}\t\t{f1score}")
