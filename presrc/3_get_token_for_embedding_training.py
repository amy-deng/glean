import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import string
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
print(os.getcwd())

# tokenize sentences, remove special characters, lower case, for training word embedding

def check_exist(outf):
    return os.path.isfile(outf)


def nltk_lemmatize(word_l):
    ''' full-form '''
    return [WordNetLemmatizer().lemmatize(i) for i in word_l]


def nltk_stem(word_l):
    return [PorterStemmer().stem(i) for i in word_l]


def get_stopwords():
    file = '../rawdat/stopwords-en.txt'
    assert check_exist(file), "can not find stopwords file {}".format(file)
    return open(file).read().split('\n')

def get_stopwords_basic():
    file = '../rawdat/stopwords-en-basic.txt'
    assert check_exist(file), "can not find stopwords file {}".format(file)
    return open(file).read().split('\n')


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def tokenize_pipeline2(text):
    stop_words = get_stopwords_basic()
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    words = tokens
    stripped = [w.translate(string.punctuation) for w in tokens]
    words = [word for word in stripped if word.isalpha() and is_english(word)]
    words = [w for w in words if not w in stop_words]
    words = nltk_stem(words)  # nltk_lemmatize
    words = [w for w in words if not w in stop_words] # final remove
    return words


def process_texts(texts):
    ''' process texts list '''
    l = []
    for t in texts:
        l.append(tokenize_pipeline2(t))
    return l


def get_tokens(args):
    text_f = os.path.join(args.dp, args.dn, 'text.txt')
    with open(text_f) as f:
        sent_l = f.readlines()
    
    sent_l = process_texts(sent_l)

    text_f = os.path.join(args.dp, args.dn, 'text_token.txt')
    outf = open(text_f, 'w')
    for sent in sent_l:
        sent = ' '.join(sent)
        outf.write("{}\n".format(sent))
    outf.close()
    print(text_f, 'saved')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    if not os.path.exists(os.path.join(args.dp, args.dn, 'text.txt')):
        print('Check if the text.txt exists')
        exit()

    get_tokens(args)


''' 
Note
after executing this file, we use sent2vec model to obtain the word embedding model.

e.g.
run next command under ~/workspace/sent2vec
./fasttext sent2vec -input <path>/AFG/text_token.txt -lr 0.2 -lrUpdateRate 100 -dim 100 -epoch 5 -minCount 8 -minCountLabel 0 -neg 10 -wordNgrams 3 -loss ns -bucket 2000000 -thread 2  -t 0.0001 -dropoutK  4 -verbose 2 -numCheckPoints 1 -output  <path>/AFG/s2v_100

Other word embedding methods or pretrained embeddings can be applied instead.
'''