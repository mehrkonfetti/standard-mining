# How to use the project
Unfortunately, the data set can not be included as a submodule and needs to be downloaded manually.
Download the full data set from [here](https://ofai.github.io/million-post-corpus/) and place in standard-mining/data/million_post_corpus.

File structure:
```
standard-mining
+-- code  
|   +-- austriazismen.py  
|   +-- ...  
+-- data  
|   +-- million_post_corpus  
|   |   +-- corpus.sqlite3  
|   |   +-- database_schema.md  
|   |   +-- ...  
```

# How to use the Stanford Tagger

Download:

```
cd ~
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip
cd stanford-corenlp-full-2018-02-27
wget http://nlp.stanford.edu/software/stanford-german-corenlp-2018-02-27-models.jar
```

Usage:
```
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-serverProperties StanfordCoreNLP-german.properties \
-preload tokenize,ssplit,pos,ner,parse \
-status_port 9002  -port 9002 -timeout 15000
```
