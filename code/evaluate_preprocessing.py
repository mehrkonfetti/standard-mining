from collections import Counter

import setup
import conf
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


def evaluate_splitters(tokens):
    """
    Compare compound splitters
    """

    # Testing compound splitters
    conf.SPLIT_COMPOUND = True
    conf.SPLITTER = 'compound-split'
    split_compound = setup.split_compounds(tokens)
    conf.SPLITTER = 'german-compound-splitter'
    split_german = setup.split_compounds(tokens)

    for count, value in enumerate(split_compound):
        print(tokens[count])
        print(split_compound[count])
        print(split_german[count])
        print("--------------")

    lengths_tokenized = 0
    lengths_compound = 0
    lengths_german = 0
    for count, value in enumerate(tokens):
        lengths_tokenized += len(value)
    for count, value in enumerate(split_compound):
        lengths_compound += len(value)
    for count, value in enumerate(split_german):
        lengths_german += len(value)

    avg_length_tokenized = lengths_tokenized / len(tokens)
    avg_length_compound = lengths_compound / len(split_compound)
    avg_length_german = lengths_german / len(split_german)

    print("Average # of tokens in input: " + str(avg_length_tokenized))
    print("Average # of tokens in split_compound: " + str(avg_length_compound))
    print("Average # of tokens in split_german: " + str(avg_length_german))


def evaluate_pos(input_strs):
    """
    Compare different POS taggers
    """
    evaluate_intersection = True  # whether to calculate the intersection and differences of the different taggers' results
    # whether to draw the distribution of POS tags between the different taggers
    plot_words_per_tag = True
    # whether to test stanford - requires a local server and longer processing time
    test_stanford = False

    conf.POS_TAGGER = "hanta"
    hanta_tags = setup.get_pos_tags(
        setup.remove_stopwords(setup.tokenize_strings(input_strs)))
    conf.POS_TAGGER = "someweta"
    someweta_tags = setup.get_pos_tags(
        setup.remove_stopwords(setup.tokenize_strings(input_strs)))

    if test_stanford:
        conf.POS_TAGGER = "stanford"
        stanford_tags = setup.get_pos_tags(
            setup.remove_stopwords(setup.tokenize_strings(input_strs)))

    if evaluate_intersection:
        global_intersection = set()
        global_length = 0
        for count, value in enumerate(hanta_tags):
            intersection = set(hanta_tags[count]) & set(someweta_tags[count])
            if test_stanford:
                intersection = intersection & set(stanford_tags[count])
            global_intersection.update(intersection)
            global_length += len(hanta_tags[count])
            print("Intersection:")
            print(intersection)
            print("Size of intersection:")
            print(len(intersection))
            print("Size of sets:")
            print(len(hanta_tags))
            print("Differences:")
            print(set(hanta_tags[count]) - set(someweta_tags[count]))
            if test_stanford:
                print(set(hanta_tags[count]) - set(stanford_tags[count]))
            print(set(someweta_tags[count]) - set(hanta_tags[count]))
            if test_stanford:
                print(set(someweta_tags[count]) - set(stanford_tags[count]))
        print("Global intersection:")
        print(global_intersection)
        print(global_length)

    global_hanta = [item for sublist in hanta_tags for item in sublist]
    global_hanta_words, global_hanta_tags = zip(*global_hanta)
    global_someweta = [item for sublist in someweta_tags for item in sublist]
    global_someweta_words, global_someweta_tags = zip(*global_someweta)
    if test_stanford:
        global_stanford = [
            item for sublist in stanford_tags for item in sublist]
        global_stanford_words, global_stanford_tags = zip(*global_stanford)

    if plot_words_per_tag:
        # plot number of words per tag
        counter_hanta = Counter(global_hanta_tags)
        counter_someweta = Counter(global_someweta_tags)
        if test_stanford:
            counter_stanford = Counter(global_stanford_tags)
        plt.bar(counter_hanta.keys(), counter_hanta.values(), width=0.8)
        plt.xlabel("Tags")
        plt.ylabel("Count")
        plt.show()
        plt.clf()
        plt.bar(counter_someweta.keys(), counter_someweta.values(), width=0.8)
        plt.show()
        if test_stanford:
            plt.clf()
            plt.bar(counter_stanford.keys(),
                    counter_stanford.values(), width=0.8)
            plt.show()


if __name__ == '__main__':
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    connection_corpus.row_factory = conf.ROW_FACTORY
    articles = setup.get_articles(connection_corpus)

    titles = [article['Title'] for article in articles]
    bodies = [article['Body'] for article in articles]
    print("Evaluating splitters on titles:")
    evaluate_splitters(setup.tokenize_strings(titles))
    print("Evaluating splitters on bodies:")
    evaluate_splitters(setup.tokenize_strings(setup.remove_html(bodies)))
    evaluate_pos(titles)
