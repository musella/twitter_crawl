#!/usr/bin/env python

from __future__ import absolute_import, print_function


from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import tweepy
from tweepy import Stream

import yaml

with open('../config.yml') as fin:
    # use safe_load instead load
    config = yaml.safe_load(fin)

print(config.keys())

consumer_key=config["consumer_key"]
consumer_secret=config["consumer_secret"]

access_token=config["access_token"]
access_token_secret=config["access_token_secret"]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


import glob

class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    
    def __init__(self,folder="../data/raw"):
        tweets = glob.glob(folder+"/tweet_*.txt")
        ## print(tweets[:10])
        latest = max(map(lambda x: int(x.rsplit(".")[-2].rsplit("_")[1]), tweets))
        self.next_id = latest + 1
        self.folder = folder
        print("Next id %d" % self.next_id)
        
    def on_data(self, data):
        with open("%s/tweet_%d.txt" % (self.folder,self.next_id), "w+") as fout:
            fout.write( data )
            print('.',)
        self.next_id += 1
        if self.next_id % 20 == 0:
            print()
        return True

    def on_error(self, status):
        print(status)

l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

stream = Stream(auth, l)
stream.filter(track=['NeurIPS2018','MachineLearning']) # ,'AI','ArtificialIntelligence'


