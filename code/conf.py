import sqlite3

TESTING = True # when testing do not use the full data set
CORPUSDB = 'data/million_post_corpus/corpus.sqlite3'
ROW_FACTORY =  sqlite3.Row # use sqlite Rows, replace by None to read raw data instead
STOPWORDREMOVER = "stop-words" # option are "stop-words" and "german-stopwords-plain"
POS_TAGGER = "someweta" # options are "hanta" and "someweta" and "stanford"
STEMMING_OR_LEMMATIZING = "lemmatizing" # options are "stemming" and "lemmatizing"
STEMMER = "snowball" # options are "cistem" and "snowball" and "snowball_swedish"
LEMMATIZER = "germalemma" # options are "germalemma" and "iwnlp" and "simplemma"
SPLIT_COMPOUND = True # whether or not to split compound words and put them in as separate words
SPLITTER = "german-compound-splitter" # options are "compound-split" and "german-compound-splitter"