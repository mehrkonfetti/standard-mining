import sqlite3

import conf
import setup


if __name__ == '__main__':
    connection_corpus = sqlite3.connect(conf.CORPUSDB)  # load corpus database
    connection_corpus.row_factory = conf.ROW_FACTORY
    articles = setup.get_articles(connection_corpus)

    titles = [article['Title'] for article in articles]
    bodies = [article['Body'] for article in articles]

    titles = setup.tokenize_strings(titles)
    bodies = setup.remove_html(bodies)
    bodies = setup.tokenize_strings(bodies)

    tokens = [item for sublist in titles for item in sublist]
    tokens.append([item for sublist in bodies for item in sublist])

    file = open("subprojects/german/austriazismen.txt", encoding="iso-8859-1")
    content = file.read()
    austriazisms = content.split()

    austrian_tokens = [token for token in tokens if token in austriazisms]
    print(austrian_tokens)

    titles_lemmatized = setup.get_lemmas(setup.get_pos_tags(setup.remove_stopwords(titles)))
    bodies_lemmatized = setup.get_lemmas(setup.get_pos_tags(setup.remove_stopwords(bodies)))

    tokens_lemmatized = [item for sublist in titles_lemmatized for item in sublist]
    tokens_lemmatized.append([item for sublist in bodies_lemmatized for item in sublist])

    austrian_tokens_lemmatized = [token for token in tokens_lemmatized if token in austriazisms]
    print(austrian_tokens_lemmatized)
