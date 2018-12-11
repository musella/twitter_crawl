#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

## make
## if [ ! -e text8 ]; then
##   if hash wget 2>/dev/null; then
##     wget http://mattmahoney.net/dc/text8.zip
##   else
##     curl -O http://mattmahoney.net/dc/text8.zip
##   fi
##   unzip text8.zip
##   rm text8.zip
## fi

PREFIX=../data/models/hastags
CORPUS=../data/preproc/hash_corpus.txt
VOCAB_FILE=$PREFIX/hash_vocab.txt
COOCCURRENCE_FILE=$PREFIX/hash_cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$PREFIX/hash_cooccurrence.shuf.bin
BUILDDIR=../GloVe/build
SAVE_FILE=$PREFIX/hash_vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=10
MAX_ITER=50000
WINDOW_SIZE=5
BINARY=2
NUM_THREADS=8
X_MAX=10

set -x

## mkdir -p $PREFIX
## 
## $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
## 
## $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
## 
## $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
## 
## 
## $BUILDDIR/glove -save-file $SAVE_FILE.d5 -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size 5 -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
## 
## $BUILDDIR/glove -save-file $SAVE_FILE.d10 -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size 10 -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

./glove2word2vec.py $SAVE_FILE.d5.txt $SAVE_FILE.d5.gensim.txt
./glove2word2vec.py $SAVE_FILE.d10.txt $SAVE_FILE.d10.gensim.txt
