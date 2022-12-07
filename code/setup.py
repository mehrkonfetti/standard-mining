import sqlite3
import conf
import re
from datetime import datetime
from stop_words import get_stop_words

def get_articles(connection_corpus):
    query_fetch_articles = 'SELECT * FROM Articles;' # extract the full Articles table
    cursor_corpus = connection_corpus.cursor()
    cursor_corpus.execute(query_fetch_articles)
    return cursor_corpus.fetchall() # return list of Rows

def clean_paths(paths):
    clean_paths = []
    for path in paths:
        if re.match(r'^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*$', path): # make sure the paths are in the right format of text/text/text
            path_fixed = path.casefold() # more aggressive version of lower()
            path_fixed = path_fixed.replace('ß', 'ss').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue') # stop germans from being germans
            clean_paths.append(tuple(path_fixed.split('/')))
    return clean_paths

def clean_dates(dates):
    clean_dates = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
 for date in dates]
    return clean_dates

def get_bindestrich_strings(texts): 
    # takes list of strings, returns list of tokens
    tokens_combined = []
    for text in texts:
        tokens = text.split(' ')
        tokens = [word.strip('.,!?:;"-–+()„“"<>\'`*') for word in tokens]
        tokens = [token for token in tokens if (len(token) > 1 and re.search(r'[a-zöüßA-ZÄÖÜ]', token))]
        bindestrich_tokens = [token for token in tokens if re.search(r'-', token)]
        for token in bindestrich_tokens:
            tokens_combined.append(token)
    return tokens_combined


def tokenize_strings(texts): 
    # takes list of strings
    # returns list of lists of strings
    tokens_combined = []
    for text in texts:
        tokens = text.casefold().split(' ')
        tokens = [word.strip('.,!?:;"-–+()„“"<>\'`*') for word in tokens] # strip any symbols, including the German Anführungszeichen
        tokens = [token for token in tokens if len(token) > 1] # there are no German words of length 1, so we can filter them + empty words
        tokens = [token for token in tokens if re.search(r'[a-zöüßA-ZÄÖÜ]', token)]
        tokens_combined.append(tokens)
    return tokens_combined

def remove_stopwords(tokens):
    # takes list of lists of strings
    # returns list of lists of strings
    if conf.STOPWORDREMOVER == "stop-words":
        file = open("subprojects/stop-words/german.txt")
        content = file.read()
        stopwords = content.split()
    elif conf.STOPWORDREMOVER == "german-stopwords-plain":
        file = open("subprojects/german_stopwords/german_stopwords_plain.txt")
        content = file.read()
        stopwords = content.split()
    else:
        print("Error, unsupported stop words detected!")
        stopwords = []

    file_english = open("subprojects/stop-words/english.txt")
    stopwords_english = file_english.read().split()

    tokens_cleaned = []
    for tokens_list in tokens:
        tokens_list_cleaned = []
        for token in tokens_list:
                if (token not in stopwords) and (token not in stopwords_english):
                    tokens_list_cleaned.append(token)
        tokens_cleaned.append(tokens_list_cleaned)
    return tokens_cleaned

if __name__ == '__main__':
    connection_corpus = sqlite3.connect(conf.CORPUSDB) # load corpus database
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
    titles_bindestrich = get_bindestrich_strings(titles)
    titles_cleaned = remove_stopwords(tokenize_strings(titles))
    print(titles_cleaned)