# How to use
1. To run the existing experiments on the data set, go to million_post_corpus/experiments and run './run.sh'


cd ~
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip
cd stanford-corenlp-full-2018-02-27
wget http://nlp.stanford.edu/software/stanford-german-corenlp-2018-02-27-models.jar

then to use: java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-serverProperties StanfordCoreNLP-german.properties \
-preload tokenize,ssplit,pos,ner,parse \
-status_port 9002  -port 9002 -timeout 15000