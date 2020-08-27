import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import sent2vec
import pickle
print(os.getcwd())

def check_exist(outf):
    return os.path.isfile(outf)


def get_stopwords_basic():
    file = '../rawdat/stopwords-en-basic.txt'
    assert check_exist(file), "can not find stopwords file {}".format(file)
    return open(file).read().split('\n')

 
def get_s2v_model(args):
    model = sent2vec.Sent2vecModel()
    model_file = os.path.join(args.dp, args.dn, 's2v_{}.bin'.format(args.dim)) 
    model.load_model(model_file)
    return model


def get_sent_voc_vectors(args):
    model = get_s2v_model(args)

    text_f = os.path.join(args.dp, args.dn, 'text_token.txt')
    with open(text_f) as f:
        tokens_l = f.readlines()
    
    ## save sentence embedding
    # sent_vec_file = os.path.join(args.dp,args.dn,'{}.s_emb'.format(args.dim))
    # if not check_exist(sent_vec_file):
    #     embs = model.embed_sentences(tokens_l)
    #     print(type(embs),embs.shape)
    #     with open(sent_vec_file, 'wb') as f: 
    #         pickle.dump(embs, f)
    #     print(sent_vec_file, 'saved!')
    # else:
    #     print(sent_vec_file, 'exists')

    # word embedding, vocabulary
    vocab = model.get_vocabulary()
    words = list(vocab.keys())
    stopwords = get_stopwords_basic() #  remove stopwords
    words = [w for w in words if not w in stopwords]

    text_vocab_file = os.path.join(args.dp, args.dn, 'vocab.txt')
    if not check_exist(text_vocab_file):
        print('vocabulary saved for the first time')
        outf = open(text_vocab_file, 'w')
        for word in words:
            outf.write("{}\n".format(word))
        outf.close()
        print(text_vocab_file, 'saved!')
    else:
        print(text_vocab_file,'exists')

    word_vec_file = os.path.join(args.dp,args.dn,'{}.w_emb'.format(args.dim))
    if not check_exist(word_vec_file):
        uni_embs = model.embed_unigrams(words)
        print(uni_embs.shape,'uni_embs')
        with open(word_vec_file, 'wb') as f: 
            pickle.dump(uni_embs, f)
        print(word_vec_file, 'saved!')
    else:
        print(word_vec_file,'exists')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")
    ap.add_argument("--dim", default=100, type=int, help="embedding dim (word/sentence)")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    if not os.path.exists(os.path.join(args.dp, args.dn, 'text.txt')):
        print('Check if the text.txt exists')
        exit()

    get_sent_voc_vectors(args)