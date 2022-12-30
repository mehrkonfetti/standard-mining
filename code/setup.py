import sqlite3
import pandas as pd
import numpy
from datetime import datetime
import re
import conf
import nltk
import simplemma
from germalemma import GermaLemma
from iwnlp.iwnlp_wrapper import IWNLPWrapper
from german_compound_splitter import comp_split
from compound_split import char_split
from someweta import ASPTagger
from nltk.parse import CoreNLPParser
from HanTa import HanoverTagger


def get_articles(input_corpus):
    """
    takes: the sqlite3 corpus
    returns: a list of articles
    """
    query_fetch_articles = 'SELECT * FROM Articles;'  # extract the full Articles table
    cursor_corpus = input_corpus.cursor()
    cursor_corpus.execute(query_fetch_articles)
    if conf.TESTING:
        return cursor_corpus.fetchmany(100)
    return cursor_corpus.fetchall()  # return list of Rows


def clean_paths(paths_as_str):
    """
    takes: a list of paths that articles reside in
    return: a tokenized list per path, where every token is one step in the path
    """
    output_paths = []
    for path in paths_as_str:
        # make sure the paths are in the right format of text/text/text
        if re.match(r'^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*$', path):
            path_fixed = path.casefold()  # more aggressive version of lower()
            # stop germans from being germans
            path_fixed = path_fixed.replace('ß', 'ss').replace(
                'ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
            output_paths.append(tuple(path_fixed.split('/')))
    return output_paths


def clean_path(path_as_str):
    """
    takes: a path that articles reside in
    return: a tokenized path, where every token is one step in the path
    """
    # make sure the paths are in the right format of text/text/text
    if re.match(r'^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*$', path_as_str):
        path_fixed = path_as_str.casefold()  # more aggressive version of lower()
        # stop germans from being germans
        path_fixed = path_fixed.replace('ß', 'ss').replace(
            'ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        return list(path_fixed.split('/'))
    return None


def clean_dates(date_strs):
    """
    takes: list of strings that represent dates
    returns: list of datetime objects
    """
    output_dates = [datetime.strptime(
        date, "%Y-%m-%d %H:%M:%S.%f") for date in date_strs]
    return output_dates


def split_data():
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    df = pd.read_sql_query("SELECT * FROM Articles;", connection_corpus)

    # split articles in test and training
    numpy.random.seed(123456)
    shuffled_articles = df.iloc[numpy.random.permutation(df.index)].reset_index(drop=True)
    shuffled_articles['Path'] = shuffled_articles['Path'].apply(clean_path)
    shuffled_articles = shuffled_articles[shuffled_articles['Path'].notna()]
    
    training_data = shuffled_articles.iloc[0:4000]
    test_data = shuffled_articles.iloc[7001:12087]
    return training_data, test_data


def tokenize_strings(list_of_strings):
    """
    takes: list of strings
    returns: list of lists of strings
    """
    tokens_combined = []
    for text in list_of_strings:
        tokens = text.casefold().split(' ')
        # filter out emails
        tokens = [token for token in tokens if not re.match(
            r'/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/gi', token)]
        tokens = [token.split('\n')
                  for token in tokens]  # remove literal newlines
        tokens = [item for sublist in tokens for item in sublist]
        # strip any symbols, including the German Anführungszeichen
        tokens = [word.strip('.,!?:;"-–+()„“"<>\'`*') for word in tokens]
        # remove 1-letter words, only words with letters in them
        tokens = [token for token in tokens if (
            len(token) > 1 and re.search(r'[a-zöüßA-ZÄÖÜ]', token))]
        tokens_combined.append(tokens)
    return tokens_combined


def get_bindestrich_strings(list_of_strings):
    """
    takes: list of strings
    returns: list of tokens
    """
    tokens = tokenize_strings(list_of_strings)
    tokens_bindestrich = []
    for list_of_tokens in tokens:
        for token in list_of_tokens:
            if re.search(r'-', token):
                tokens_bindestrich.append(token)
    return tokens_bindestrich


def split_compounds(input_list):
    """
    takes: list of lists of strings
    returns: list of lists of strings
    """
    output_list = []
    if conf.SPLITTER == "german-compound-splitter":
        ahocs = comp_split.read_dictionary_from_file(
            "subprojects/german/german_utf8_linux.dic")
    for tokens in input_list:
        split_tokens = []
        for token in tokens:
            if conf.SPLITTER == "compound-split":
                split_token = char_split.split_compound(token)[0]
                if split_token[0] > 0.75:
                    split_tokens.append(split_token[1])
                    split_tokens.append(split_token[2])
                else:
                    split_tokens.append(token)
            elif conf.SPLITTER == "german-compound-splitter":
                try:
                    split_token = comp_split.dissect(
                        token, ahocs, only_nouns=True)
                    for section in split_token:
                        split_tokens.append(section.lower())
                except IndexError:  # token couldn't be split
                    split_tokens.append(token)
        output_list.append(split_tokens)
    return output_list


def remove_stopwords(input_list):
    """
    takes: list of lists of strings
    returns: list of lists of strings
    """
    if conf.STOPWORDREMOVER == "stop-words":
        file = open("subprojects/stopwords/stop-words/german.txt",
                    encoding="utf8")
        content = file.read()
        stopwords = content.split()
    elif conf.STOPWORDREMOVER == "german-stopwords-plain":
        file = open(
            "subprojects/stopwords/german_stopwords/german_stopwords_plain.txt", encoding="utf8")
        content = file.read()
        stopwords = content.split()
    else:
        print("Error, unsupported stop words configuration!")
        stopwords = []

    # also add english stopwords
    file_english = open(
        "subprojects/stopwords/stop-words/english.txt", encoding="utf8")
    stopwords_english = file_english.read().split()

    output_list = []
    for tokens in input_list:
        tokens_cleaned = [token for token in tokens if token not in stopwords and token not in stopwords_english]
        output_list.append(tokens_cleaned)
    return output_list


def get_pos_tags(input_list):
    """
    takes: list of lists of string
    returns: list of lists of (string, pos_tag)
    """
    if conf.POS_TAGGER == "hanta":
        tagger = HanoverTagger.HanoverTagger('morphmodel_ger.pgz')
        tuples = []
        for tokens in input_list:
            tuples_sentence = []
            for token in tokens:
                # this returns a list of the most likely tags - I'm blankly using #1
                tags_list = tagger.tag_word(token, cutoff=0)
                tuples_sentence.append((token, tags_list[0][0]))
            tuples.append(tuples_sentence)
        return tuples
    if conf.POS_TAGGER == "someweta":
        model = "subprojects/german_newspaper_2020-05-28.model"
        tagger = ASPTagger()
        tagger.load(model)
        return [tagger.tag_sentence(tokens) for tokens in input_list]
    if conf.POS_TAGGER == "stanford":
        tagger = CoreNLPParser(url='http://localhost:9002', tagtype='pos')
        return [list(tagger.tag(tokens)) for tokens in input_list]
    print("Error, unsupported POS tagger configuration!")
    return [[]]


def get_stems(input_list):
    """
    takes: list of lists of (string, pos_tag)
    returns: list of lists of (string, pos_tag)
    """
    if conf.STEMMER == "cistem":
        stemmer = nltk.stem.Cistem()
    elif conf.STEMMER == "snowball":
        stemmer = nltk.stem.SnowballStemmer("german")
    elif conf.STEMMER == "snowball_swedish":
        stemmer = nltk.stem.SnowballStemmer("swedish")
    else:
        stemmer = nltk.stem.Cistem()
    output_list = []
    for tokens in input_list:
        stemmed_tokens = [(stemmer.stem(token[0]), token[1])
                          for token in tokens]
        output_list.append(stemmed_tokens)
    return output_list


def get_suffixes(input_list):
    """
    takes: list of lists of (string, pos_tag)
    returns: list of lists of (string, pos_tag)
    """
    stemmer = nltk.stem.Cistem()
    output_list = []
    for tokens in input_list:
        suffixes = [(stemmer.segment(token[0])[1], token[1])
                    for token in tokens]
        output_list.append(suffixes)
    return output_list


def get_lemmas(input_list):
    """
    takes: list of lists of (string, pos_tag)
    returns: list of lists of (string, pos_tag)
    """
    if conf.LEMMATIZER == "germalemma":
        lemmatizer = GermaLemma()
        output_list = []
        for tuples in input_list:
            lemmatuples = []
            for lemmatuple in tuples:
                try:
                    lemma = lemmatizer.find_lemma(lemmatuple[0], lemmatuple[1])
                except ValueError:  # Can only process a very limited number of tags
                    lemma = lemmatuple[0]  # use unlemmatized token
                lemmatuples.append((lemma, lemmatuple[1]))
            output_list.append(lemmatuples)
        return output_list
    if conf.LEMMATIZER == "iwnlp":
        lemmatizer = IWNLPWrapper(
            lemmatizer_path='subprojects/IWNLP.Lemmatizer_20181001.json')
        output_list = []
        for tuples in input_list:
            lemmatuples = []
            for lemmatuple in tuples:
                lemma = lemmatizer.lemmatize(lemmatuple[0], lemmatuple[1])
                if lemma is None:  # returns None for unknown words
                    # TODO: explore further which words are unknown to it - does it recognize that english words are unknown eg
                    lemma = lemmatuple[0]
                lemmatuples.append((lemma, lemmatuple[1]))
            output_list.append(lemmatuples)
        return output_list
    if conf.LEMMATIZER == "simplemma":
        output_list = []
        for tuples in input_list:
            lemmatuples = [(simplemma.lemmatize(
                tuple[0], lang='de'), tuple[1]) for tuple in tuples]
            output_list.append(lemmatuples)
        return output_list
    return []


def remove_html(list_of_strings):
    """
    takes: list of strings
    returns: list of strings
    https://medium.com/mlearning-ai/3-ways-to-clean-your-html-text-for-nlp-text-pre-processing-70bc5b876445
    """
    re_html = re.compile(r'<[^>]+>')
    return [re_html.sub('', text) for text in list_of_strings]


def clean_title(title):
    return remove_stopwords(tokenize_strings(title))


def custom_preprocess(text):
    tokenized = get_lemmas(get_pos_tags(remove_stopwords(tokenize_strings([text]))))[0]
    return [token[0] for token in tokenized]


def custom_preprocess_html(text):
    tokenized = get_lemmas(get_pos_tags(remove_stopwords(tokenize_strings(remove_html([text])))))[0]
    return [token[0] for token in tokenized]


if __name__ == '__main__':
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    connection_corpus.row_factory = conf.ROW_FACTORY
    articles = get_articles(connection_corpus)

    ids = [article['ID_Article'] for article in articles]
    paths = [article['Path'] for article in articles]
    dates = [article['publishingDate'] for article in articles]
    titles = [article['Title'] for article in articles]
    bodies = [article['Body'] for article in articles]

    paths_id = list(zip(ids, paths))

    paths_clean = clean_paths(paths)
    dates_clean = clean_dates(dates)

    titles_tokenized = tokenize_strings(titles)
    titles_bindestrich = get_bindestrich_strings(titles)  # save for later

    if conf.SPLIT_COMPOUND:
        titles_tokenized_split = split_compounds(titles_tokenized)

    titles_cleaned = remove_stopwords(titles_tokenized)
    titles_pos = get_pos_tags(titles_cleaned)

    titles_stems = get_stems(titles_pos)
    titles_suffixes = get_suffixes(titles_pos)  # save for later
    titles_lemmas = get_lemmas(titles_pos)

    bodies_rm_html = remove_html(bodies)
    bodies_tokenized = tokenize_strings(bodies_rm_html)
    bodies_bindestrich = get_bindestrich_strings(
        bodies_rm_html)  # save for later

    if conf.SPLIT_COMPOUND:
        bodies_tokenized_split = split_compounds(bodies_tokenized)

    bodies_cleaned = remove_stopwords(bodies_tokenized)
    bodies_pos = get_pos_tags(bodies_cleaned)

    bodies_stems = get_stems(bodies_pos)
    bodies_suffixes = get_suffixes(bodies_pos)  # save for later
    bodies_lemmas = get_lemmas(bodies_pos)
