import sqlite3

CORPUSDB = 'data/million_post_corpus/corpus.sqlite3'
ROW_FACTORY =  sqlite3.Row # use sqlite Rows, replace by None to read raw data instead
STOPWORDREMOVER = "stop-words" # option are "stop-words" and "german-stopwords-plain"