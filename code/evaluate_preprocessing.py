from collections import Counter

import setup
import conf
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


def evaluate_splitters_titles(input_titles):
    """
    Compare compound splitters
    """
    titles_tokenized = setup.tokenize_strings(input_titles)

    # Testing compound splitters
    conf.SPLIT_COMPOUND = True
    conf.SPLITTER = 'compound-split'
    titles_split_compound = setup.split_compounds(titles_tokenized)
    conf.SPLITTER = 'german-compound-splitter'
    titles_split_german = setup.split_compounds(titles_tokenized)

    for count, value in enumerate(titles_split_compound):
        print(titles_tokenized[count])
        print(titles_split_compound[count])
        print(titles_split_german[count])
        print("--------------")

    lengths_tokenized = 0
    lengths_compound = 0
    lengths_german = 0
    for count, value in enumerate(titles_tokenized):
        lengths_tokenized += len(value)
    for count, value in enumerate(titles_split_compound):
        lengths_compound += len(value)
    for count, value in enumerate(titles_split_german):
        lengths_german += len(value)

    avg_length_tokenized = lengths_tokenized / len(titles_tokenized)
    avg_length_compound = lengths_compound / len(titles_split_compound)
    avg_length_german = lengths_german / len(titles_split_german)

    print("Average # of tokens in titles_tokenized: " + str(avg_length_tokenized))
    print("Average # of tokens in titles_split_compound: " + str(avg_length_compound))
    print("Average # of tokens in titles_split_german: " + str(avg_length_german))


def evaluate_splitters_bodies(input_bodies):
    """
    Compare compound splitters
    """
    bodies_tokenized = setup.tokenize_strings(setup.remove_html(input_bodies))

    # Testing compound splitters
    conf.SPLIT_COMPOUND = True
    conf.SPLITTER = 'compound-split'
    bodies_split_compound = setup.split_compounds(bodies_tokenized)
    conf.SPLITTER = 'german-compound-splitter'
    bodies_split_german = setup.split_compounds(bodies_tokenized)

    for count, value in enumerate(bodies_split_compound):
        print(bodies_tokenized[count])
        print(bodies_split_compound[count])
        print(bodies_split_german[count])
        print("--------------")

    lengths_tokenized = 0
    lengths_compound = 0
    lengths_german = 0
    for count, value in enumerate(bodies_tokenized):
        lengths_tokenized += len(value)
    for count, value in enumerate(bodies_split_compound):
        lengths_compound += len(value)
    for count, value in enumerate(bodies_split_german):
        lengths_german += len(value)

    avg_length_tokenized = lengths_tokenized / len(bodies_tokenized)
    avg_length_compound = lengths_compound / len(bodies_split_compound)
    avg_length_german = lengths_german / len(bodies_split_german)

    print("Average # of tokens in bodies_tokenized: " + str(avg_length_tokenized))
    print("Average # of tokens in bodies_split_compound: " + str(avg_length_compound))
    print("Average # of tokens in bodies_split_german: " + str(avg_length_german))


def evaluate_pos_titles(input_titles):
    """
    Compare different POS taggers
    """
    conf.POS_TAGGER = "hanta"
    hanta_tags = setup.get_pos_tags(setup.remove_stopwords(setup.tokenize_strings(input_titles)))
    conf.POS_TAGGER = "someweta"
    someweta_tags = setup.get_pos_tags(setup.remove_stopwords(setup.tokenize_strings(input_titles)))
    conf.POS_TAGGER = "stanford"
    stanford_tags = setup.get_pos_tags(setup.remove_stopwords(setup.tokenize_strings(input_titles)))
    global_intersection = set()
    global_length = 0
    if False:
        for count, value in enumerate(hanta_tags):
            intersection = set(hanta_tags[count]) & set(someweta_tags[count])
            intersection = intersection & set(stanford_tags[count])
            global_intersection.update(intersection)
            global_length += len(hanta_tags[count])
            if False:
                print("Intersection:")
                print(intersection)
                print("Size of intersection:")
                print(len(intersection))
                print("Size of sets:")
                print(len(hanta_tags))
            if False:
                print("Differences:")
                print(set(hanta_tags[count]) - set(someweta_tags[count]))
                print(set(hanta_tags[count]) - set(stanford_tags[count]))
                print(set(someweta_tags[count]) - set(hanta_tags[count]))
                print(set(someweta_tags[count]) - set(stanford_tags[count]))
        print("Global intersection:")
        print(global_intersection)
        print(global_length)

    global_hanta = [item for sublist in hanta_tags for item in sublist]
    global_hanta_words, global_hanta_tags = zip(*global_hanta)
    global_someweta = [item for sublist in someweta_tags for item in sublist]
    global_someweta_words, global_someweta_tags = zip(*global_someweta)
    global_stanford = [item for sublist in stanford_tags for item in sublist]
    global_stanford_words, global_stanford_tags = zip(*global_stanford)

    if True:
        # plot number of words per tag
        counter_hanta = Counter(global_hanta_tags)
        counter_someweta = Counter(global_someweta_tags)
        counter_stanford = Counter(global_stanford_tags)
        plt.bar(counter_hanta.keys(), counter_hanta.values(), width=0.8)
        plt.xlabel("Tags")
        plt.ylabel("Count")
        plt.show()
        plt.clf()
        plt.bar(counter_someweta.keys(), counter_someweta.values(), width=0.8)
        plt.show()
        plt.clf()
        plt.bar(counter_stanford.keys(), counter_stanford.values(), width=0.8)
        plt.show()

    # plot number of tags per same word


if __name__ == '__main__':
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    connection_corpus.row_factory = conf.ROW_FACTORY
    articles = setup.get_articles(connection_corpus)

    titles = [article['Title'] for article in articles]
    bodies = [article['Body'] for article in articles]
    # evaluate_splitters_titles(titles)
    # evaluate_splitters_bodies(bodies)
    evaluate_pos_titles(titles)
