# Code and scripts to train a sequence-generating model on a stream of tweets

More information can be found in my post _Tweet-to-tweet learning_ on http://musella.github.io 


1. Tweets are streamed by the [tweet_streamer.py](scripts/tweet_streamer.py) script.

1. Data cleaning and preprocessing is performed by the [preprocess.ipynb](notebooks/preprocess.ipynb) notebook.

1. The script [hash_embed.sh](scripts/hash_embed.sh) trains a [GloVe](https://nlp.stanford.edu/projects/glove/) embedding on the dataset hashtags. The embedding can be visualized through the notebook [hashtag_embedding.ipynb](notebooks/hashtag_embedding.ipynb)

1. The sequence generator is trained using [train_rnn.ipynb](notebooks/train_rnn.ipynb).

1. Words sequences are then generated through `beam search` and served using [Flask](http://flask.pocoo.org) (flask_app/app.py)



