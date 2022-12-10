import sqlite3
import conf
import re
from datetime import datetime
import nltk

def get_articles(connection_corpus):
    query_fetch_articles = 'SELECT * FROM Articles;' # extract the full Articles table
    cursor_corpus = connection_corpus.cursor()
    cursor_corpus.execute(query_fetch_articles)
    if conf.TESTING:
        return cursor_corpus.fetchmany(40)
    else:
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
    clean_dates = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f") for date in dates]
    return clean_dates


def tokenize_strings(list_of_strings): 
    # takes list of strings
    # returns list of lists of strings
    tokens_combined = []
    for text in list_of_strings:
        tokens = text.casefold().split(' ')
        tokens = [token for token in tokens if not re.match(r'/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/gi', token)] # filter out emails
        tokens = [token.split('\n') for token in tokens] # remove literal newlines
        tokens = [item for sublist in tokens for item in sublist]
        tokens = [word.strip('.,!?:;"-–+()„“"<>\'`*') for word in tokens] # strip any symbols, including the German Anführungszeichen
        tokens = [token for token in tokens if (len(token) > 1 and re.search(r'[a-zöüßA-ZÄÖÜ]', token))] # remove 1-letter words, only words with letters in them
        tokens_combined.append(tokens)
    return tokens_combined


def get_bindestrich_strings(list_of_strings): 
    # takes list of strings
    # returns list of tokens
    tokens = tokenize_strings(list_of_strings)
    tokens_bindestrich = []
    for list_of_tokens in tokens:
        for token in list_of_tokens:
            if re.search(r'-', token):
                tokens_bindestrich.append(token)
    return tokens_bindestrich


def split_compounds(list_of_lists_of_tokens):
    # takes list of lists of strings
    # returns list of lists of strings
    list_of_lists_of_tokens_split = []
    if conf.SPLITTER == "german-compound-splitter":
        from german_compound_splitter import comp_split
        ahocs = comp_split.read_dictionary_from_file("subprojects/german/german_utf8_linux.dic")
    for list_of_tokens in list_of_lists_of_tokens:
        list_of_tokens_split = []
        for token in list_of_tokens:
            if conf.SPLITTER == "compound-split":
                from compound_split import char_split
                token_split = char_split.split_compound(token)[0]
                if token_split[0] > 0.75:
                    list_of_tokens_split.append(token_split[1])
                    list_of_tokens_split.append(token_split[2])
                else:
                    list_of_tokens_split.append(token)
            elif conf.SPLITTER == "german-compound-splitter":
                try:
                    token_split = comp_split.dissect(token, ahocs, only_nouns=True)
                    for section in token_split:
                        list_of_tokens_split.append(section.lower())
                except:
                    list_of_tokens_split.append(token)
        list_of_lists_of_tokens_split.append(list_of_tokens_split)
    return list_of_lists_of_tokens_split


def remove_stopwords(list_of_lists_of_tokens):
    # takes list of lists of strings
    # returns list of lists of strings
    if conf.STOPWORDREMOVER == "stop-words":
        file = open("subprojects/stopwords/stop-words/german.txt")
        content = file.read()
        stopwords = content.split()
    elif conf.STOPWORDREMOVER == "german-stopwords-plain":
        file = open("subprojects/stopwords/german_stopwords/german_stopwords_plain.txt")
        content = file.read()
        stopwords = content.split()
    else:
        print("Error, unsupported stop words configuration!")
        stopwords = []

    file_english = open("subprojects/stopwords/stop-words/english.txt") # also add english stopwords
    stopwords_english = file_english.read().split()

    tokens_cleaned = []
    for tokens_list in list_of_lists_of_tokens:
        tokens_list_cleaned = [token for token in tokens_list if ((token not in stopwords) and (token not in stopwords_english))]
        tokens_cleaned.append(tokens_list_cleaned)
    return tokens_cleaned


def get_pos_tags(lists_of_lists_of_tokens):
    # takes list of lists of string
    # returns list of lists of (string, pos_tag)
    if conf.POS_TAGGER == "hanta":
        from HanTa import HanoverTagger
        tagger = HanoverTagger.HanoverTagger('morphmodel_ger.pgz')

        lists_of_lists_of_tuples = []
        for list_of_tokens in lists_of_lists_of_tokens:
            list_of_tuples = []
            for token in list_of_tokens:
                tags_list = tagger.tag_word(token,cutoff=0) # this returns a list of the most likely tags - I'm blankly using #1
                list_of_tuples.append((token, tags_list[0][0]))
            lists_of_lists_of_tuples.append(list_of_tuples)
        return lists_of_lists_of_tuples
    elif conf.POS_TAGGER == "someweta":
        from someweta import ASPTagger
        model = "subprojects/german_newspaper_2020-05-28.model"
        tagger = ASPTagger()
        tagger.load(model)
        return [tagger.tag_sentence(list_of_tokens) for list_of_tokens in lists_of_lists_of_tokens]
    elif conf.POS_TAGGER == "stanford":
        from nltk.parse import CoreNLPParser
        tagger = CoreNLPParser(url='http://localhost:9002', tagtype='pos')
        return [list(tagger.tag(list_of_tokens)) for list_of_tokens in lists_of_lists_of_tokens]
    else:
        print("Error, unsupported POS tagger configuration!")
        return [[]]

def get_stems(lists_of_lists_of_tokens): # TODO: rewrite for tuples
    # takes list of lists of (string, pos_tag)
    # returns list of lists of (string, pos_tag)
    if conf.STEMMER == "cistem":
        stemmer = nltk.stem.Cistem()
    elif conf.STEMMER == "snowball":
        stemmer = nltk.stem.SnowballStemmer("german")
    elif conf.STEMMER == "snowball_swedish":
        stemmer = nltk.stem.SnowballStemmer("swedish")
    else:
        stemmer = nltk.stem.Cistem()
    lists_of_lists_of_tokens_stem = []
    for list_of_tokens in lists_of_lists_of_tokens:
        list_of_tokens_stem = [(stemmer.stem(token[0]), token[1]) for token in list_of_tokens]
        lists_of_lists_of_tokens_stem.append(list_of_tokens_stem)
    return lists_of_lists_of_tokens_stem


def get_suffixes(lists_of_lists_of_tokens): # TODO: rewrite for tuples
    # takes list of lists of (string, pos_tag)
    # returns list of lists of (string, pos_tag)
    stemmer = nltk.stem.Cistem()
    lists_of_lists_of_suffixes = []
    for list_of_tokens in lists_of_lists_of_tokens:
        list_of_suffixes = [(stemmer.segment(token[0])[1], token[1]) for token in list_of_tokens]
        lists_of_lists_of_suffixes.append(list_of_suffixes)
    return lists_of_lists_of_suffixes


def get_lemmas(lists_of_lists_of_tuples):
    # takes list of lists of (string, pos_tag)
    # returns list of lists of (string, pos_tag)
    if conf.LEMMATIZER == "germalemma":
        from germalemma import GermaLemma
        lemmatizer = GermaLemma()
        lists_of_lists_of_lemmatuples = []
        for list_of_tuples in lists_of_lists_of_tuples:
            list_of_lemmatuples = []
            for tuple in list_of_tuples:
                try:
                    lemma = lemmatizer.find_lemma(tuple[0], tuple[1])
                except: # Can only process a very limited number of tags
                    lemma = tuple[0]
                list_of_lemmatuples.append((lemma, tuple[1]))
            lists_of_lists_of_lemmatuples.append(list_of_lemmatuples)
        return lists_of_lists_of_lemmatuples
    elif conf.LEMMATIZER == "iwnlp":
        from iwnlp.iwnlp_wrapper import IWNLPWrapper
        lemmatizer = IWNLPWrapper(lemmatizer_path='subprojects/IWNLP.Lemmatizer_20181001.json')
        lists_of_lists_of_lemmatuples = []
        for list_of_tuples in lists_of_lists_of_tuples:
            list_of_lemmatuples = []
            for tuple in list_of_tuples:
                lemma = lemmatizer.lemmatize(tuple[0], tuple[1])
                if lemma == None: # returns None for unknown words
                    # TODO: explore further which words are unknown to it - does it recognize that english words are unknown eg
                    lemma = tuple[0]
                list_of_lemmatuples.append((lemma, tuple[1]))
            lists_of_lists_of_lemmatuples.append(list_of_lemmatuples)
        return lists_of_lists_of_lemmatuples
    elif conf.LEMMATIZER == "simplemma":
        import simplemma
        lists_of_lists_of_lemmatuples = []
        for list_of_tuples in lists_of_lists_of_tuples:
            list_of_lemmatuples = [(simplemma.lemmatize(tuple[0], lang='de'), tuple[1]) for tuple in list_of_tuples]
            lists_of_lists_of_lemmatuples.append(list_of_lemmatuples)
        return lists_of_lists_of_lemmatuples


def remove_html(list_of_strings):
    # takes list of strings
    # returns list of strings
    # source: https://medium.com/mlearning-ai/3-ways-to-clean-your-html-text-for-nlp-text-pre-processing-70bc5b876445
    re_html = re.compile(r'<[^>]+>')
    return [re_html.sub('', text) for text in list_of_strings]

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

    titles_tokenized = tokenize_strings(titles)
    titles_bindestrich = get_bindestrich_strings(titles) # save for later

    if conf.SPLIT_COMPOUND:
        titles_tokenized_split = split_compounds(titles_tokenized)

    titles_cleaned = remove_stopwords(titles_tokenized)
    titles_pos = get_pos_tags(titles_cleaned)
    
    titles_stems = get_stems(titles_pos)
    titles_suffixes = get_suffixes(titles_pos) # save for later
    titles_lemmas = get_lemmas(titles_pos)


    bodies_rm_html = remove_html(bodies)
    bodies_tokenized = tokenize_strings(bodies_rm_html) # TODO: double check and sanity check that they are actually clean
    bodies_bindestrich = get_bindestrich_strings(bodies_rm_html) # save for later

    if conf.SPLIT_COMPOUND:
        bodies_tokenized_split = split_compounds(bodies_tokenized)

    bodies_cleaned = remove_stopwords(bodies_tokenized)
    bodies_pos = get_pos_tags(bodies_cleaned)
    
    bodies_stems = get_stems(bodies_pos)
    bodies_suffixes = get_suffixes(bodies_pos) # save for later
    bodies_lemmas = get_lemmas(bodies_pos)

    #for i in range(0, len(bodies_stems)):
        #for j in range(0, len(bodies_stems[i])):
            #print(bodies_stems[i][j])
            #print(bodies_lemmas[i][j])
            #print("--------------")