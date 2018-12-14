import zipfile
import re
import json
import glob
import numpy as np
import pandas as pd

def load(fname):
    with open(fname) as fin:
        try:
            return json.load(fin)
        except Exception as e:
            print("Error reading %s: %s" % (fname, str(e)))
    return None

def load_all_tweets(path,keys=["id","text","lang","retweeted","retweet_count","truncated","user/name"],lang="en"):
    tweets = glob.glob(path+"/tweet_*.txt")

    filter_lang = (lang is not None and lang != "")
    if filter_lang and not "lang" in keys:
        keys.append("lang")

    df = pd.DataFrame( [ info for info in map(lambda x: extract_info(x,keys), sorted(tweets)) if info is not None ] )
    if lang is not None and lang != "":
        df = df[df["lang"] == lang]

    return df

def get_info(key,content):
    path = key.split("/")
    for p in path:
        content = content[p]
    return content

def extract_info(fname,keep_keys):
    content = load(fname)
    if content is None: return None
    return {key:get_info(key,content) for key in keep_keys}

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won[\'’]t", "will not", phrase)
    phrase = re.sub(r"can[\'’]t", "can not", phrase)

    # general
    phrase = re.sub(r"n[\'’]t", " not", phrase)
    phrase = re.sub(r"[\'’]re", " are", phrase)
    # phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"[\'’]d", " would", phrase)
    phrase = re.sub(r"[\'’]t", " not", phrase)
    phrase = re.sub(r"[\'’]ve", " have", phrase)
    phrase = re.sub(r"[\'’]m", " am", phrase)
    phrase = re.sub(r"([Hh]ere|[Ii]t|[Ww]hat|[Tt]hat)[\'’]", r"\1 is", phrase)
    phrase = re.sub("'[Ss]"," 's",phrase)
    phrase = re.sub(r"'([\w]+)'",r"\1",phrase)
    phrase = re.sub(r"'([\w]+)",r"\1",phrase)
    phrase = re.sub(r"([\w]+)'",r"\1",phrase)
    return phrase

def cleanup_text(phrase,pre_filters,post_filters):
    phrase = re.sub(r"[%s]" % pre_filters, " ", phrase)
    phrase = decontracted(phrase)
    #phrase = re.sub(r"[^ ]+…","<trunc>",phrase)
    phrase = re.sub(r"[^ ]+…"," ",phrase)

    # translated to python from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    phrase = re.sub(r"https?:\/\/([\w+\.\/])+"," <url>",phrase)
    # phrase = re.sub(r"&\w;","",phrase)

    phrase = re.sub("/"," / ",phrase) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    phrase = re.sub(r"([^\w])[-+]?[.\d]*[\d]+[:,.\d]*",r" \1<number> ",phrase)
    phrase = re.sub(r"^[-+]?[.\d]*[\d]+[:,.\d]*", r" <number> ",phrase)
    phrase = re.sub(r"@\w+","<user>",phrase) # 
    phrase = re.sub(r"#{"+eyes+r"}#{"+nose+"r}[)d]+|[)d]+#{"+nose+"r}#{"+eyes+"}",r" <smile>",phrase) # /i
    phrase = re.sub(r"#{"+eyes+r"}#{"+nose+"r}p+",r"<lolface>",phrase) # /i
    phrase = re.sub(r"#{"+eyes+r"}#{"+nose+"r}\(+|\)+#{"+nose+"r}#{"+eyes+"}",r" <sadface>",phrase)
    phrase = re.sub(r"#{"+eyes+r"}#{"+nose+"r}[\/|l*]",r" <neutralface>",phrase)
    phrase = re.sub(r"<3",r" <heart>",phrase)
    phrase = re.sub(r"#\w+",lambda x: " <hashtag> "+x.group().lower(), phrase) # make hastags lowercase    
    phrase = re.sub(r"([!?.]){2,}",r"\1 <repeat>",phrase) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>",phrase)
    phrase = re.sub("\b(\S*?)(.)\2{2,}\b",r"\1 \2 <elong>",phrase) # Mark elongated words (eg. "wayyyy" => "way <ELONG>",phrase)

    phrase = re.sub(r"[%s]" % post_filters, " ", phrase)
    phrase = re.sub(r"[ ]+"," ",phrase)
    return phrase+" <stop>"

def get_uknown_words(words_dict,word_embed,max_word):
    missing = []
    hashtags = []

    for word,iword in filter(lambda x: x[1]<max_word and x[0] not in  word_embed.keys(), words_dict.items()):
        if word.startswith("#"):
            hashtags.append((word,iword))
        else:
            missing.append((word,iword))
    return missing, hashtags

def glove_embedding_path(glove_dim,prefix="../input/"):
    return prefix+'glove.twitter.27B.' + str(glove_dim) + 'd.txt'

    
def load_glove_embedding(glove_file):
    emb_dict = {}
    glove = open(glove_file)

    for line in glove.readlines():
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
    glove.close()
    return emb_dict



def fill_embedding_matrix(words_dict,emb_dict,max_word,emb_dim):
    emb_mtx = np.zeros( (max_word+1,emb_dim) )
    unknown = []
    
    for word, pos in words_dict.items():
        if pos >= max_word: continue
        if not word in emb_dict:
            # print('Unkown word '+word)
            unknown.append(word)
            continue
        emb_mtx[pos] = emb_dict.get(word)
    return emb_mtx, unknown
            