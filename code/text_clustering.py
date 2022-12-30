import setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN


def visualize_distribution(clusterer):
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    plt.ylabel('Number of documents')
    plt.xlabel('Document cluster')
    plt.xticks(unique)
    plt.bar(unique, counts)
    plt.show()


def print_cluster_keywords(clusterer, vectorizer, k_index):
    centers = clusterer.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(k_index):
        print(f"Cluster {i}: ", end="")
        for ind in centers[i, :10]:
            print(f"{terms[ind]} ", end="")
        print()


def unique_paths(paths):
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    #print(unique)
    return unique


def calculate_rand(size, clustered, training_data_stripped):
    pos_count = 0
    for i in range(0, size):
        for j in range(i, size):
            if i != j:
                if clustered[i] == clustered[j]:  # Same cluster and same category
                    if training_data_stripped['Path'][i] == training_data_stripped['Path'][j]:
                        pos_count += 1
                if clustered[i] != clustered[j]:  # Different cluster and different category
                    if training_data_stripped['Path'][i] != training_data_stripped['Path'][j]:
                        pos_count += 1
    rand_index = pos_count / ((size - 1) * size / 2)
    return rand_index


def find_optimal_kindex(clusterer, training_bodies, training_data_stripped):
    k_index = range(3, 150)
    print(f"Index\t\tRand")

    for index in k_index:
        clusterer = KMeans(n_clusters=index, n_init=7)
        clustered = clusterer.fit_predict(training_bodies[:500])
        # visualize_distribution(clusterer)
        # print_cluster_keywords(clusterer, vectorizer, optimal_k_index)
        # calculate rand index
        pos_count = 0
        for i in range(0, 500):
            for j in range(i, 500):
                if i != j:
                    if clustered[i] == clustered[j]:  # Same cluster and same category
                        if training_data_stripped['Path'][i] == training_data_stripped['Path'][j]:
                            pos_count += 1
                    if clustered[i] != clustered[j]:  # Different cluster and different category
                        if training_data_stripped['Path'][i] != training_data_stripped['Path'][j]:
                            pos_count += 1
        rand_index = pos_count / (499 * 500 / 2)
        print(f"{index}\t\t{rand_index}")


if __name__ == '__main__':
    training_data, test_data = setup.split_data()

    # Prepare data sets
    training_data_stripped = training_data.copy(deep=True)
    training_data_stripped['Path'] = pd.Series([[path[0], path[1]] if len(path) > 1 else None for path in training_data_stripped['Path']]).values
    training_data_stripped = training_data_stripped.dropna(subset=['Path'])
    # print(len(unique_paths(training_data_stripped['Path'])))

    vectorizer = TfidfVectorizer(tokenizer=setup.custom_preprocess_html)
    training_bodies = vectorizer.fit_transform(training_data_stripped['Body'])

    # clusterer = KMeans(n_clusters=21, n_init=7)
    # clustered = clusterer.fit_predict(training_bodies)
    # visualize_distribution(clusterer)
    # print_cluster_keywords(clusterer, vectorizer, 21)

    eps_vals = [0.2, 5.0, 50.0]
    for eps_val in eps_vals:
        clusterer_comp = DBSCAN(eps=0.2)
        clustered_comp = clusterer_comp.fit_predict(training_bodies)
        rand_index = calculate_rand(500, clustered_comp, training_data_stripped)
        print(f"{eps_val}\t\t{rand_index}")

