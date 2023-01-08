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
    austriacisms = content.split()

    austrian_tokens = [token for token in tokens if token in austriacisms]
    print(austrian_tokens)
