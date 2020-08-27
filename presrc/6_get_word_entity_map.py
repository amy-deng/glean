import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import pickle
import string
print(os.getcwd())

def check_exist(outf):
    return os.path.isfile(outf)

def get_word_entity_map(args):
    map_file = os.path.join(args.dp, args.dn, 'word_entity_map.txt')
    if not check_exist(map_file):
        # get vocab list
        vocab_f = os.path.join(args.dp, args.dn, 'vocab.txt')
        with open(vocab_f, 'r') as f:
            vocab_l = f.read().splitlines()   

        # get entity2id
        ent_file = os.path.join(args.dp, args.dn, 'entity2id.txt') 
        ent_df = pd.read_csv(ent_file, sep='\t',names=['name','id'])

        # get word embedding of words that related to the entities in current graph
        word_idx_l = []
        for i,row in ent_df.iterrows():
            ent = row['name'].lower()
            related_words = []
            for j in range(len(vocab_l)):
                if len(vocab_l[j])>=5 and vocab_l[j] in ent:
                    related_words.append(j)
            word_idx_l.append(related_words)
        
        arr_list = np.array(word_idx_l,dtype=object)
        with open(map_file, 'wb') as fp:
            pickle.dump(word_idx_l, fp)
        print(map_file, 'saved!')
    else:
        print(map_file, 'exists!')

def get_word_rel_map(args):
    map_file = os.path.join(args.dp, args.dn, 'word_relation_map.txt')
    if not check_exist(map_file):
        # get vocab list
        vocab_f = os.path.join(args.dp, args.dn, 'vocab.txt')
        with open(vocab_f, 'r') as f:
            vocab_l = f.read().splitlines()   

        # get entity2id
        ent_file = os.path.join(args.dp, args.dn, 'relation2id.txt') 
        ent_df = pd.read_csv(ent_file, sep='\t',names=['name','id'])

        # get word embedding of words that related to the entities in current graph
        # list = [(the first entity)[word index],[]]
        word_idx_l = []
        for i,row in ent_df.iterrows():
            ent = row['name'].lower()
            related_words = []
            for j in range(len(vocab_l)):
                if len(vocab_l[j])>=5 and vocab_l[j] in ent:
                    related_words.append(j)
            word_idx_l.append(related_words)
        
        arr_list = np.array(word_idx_l, dtype=object)
        with open(map_file, 'wb') as fp:
            pickle.dump(word_idx_l, fp)
        print(map_file, 'saved!')
    else:
        print(map_file, 'exists!')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    if not os.path.exists(os.path.join(args.dp, args.dn, 'vocab.txt')):
        print('Check if the vocab.txt exists')
        exit()

    if not os.path.exists(os.path.join(args.dp, args.dn, 'entity2id.txt')):
        print('Check if the entity2id.txt exists')
        exit()

    get_word_entity_map(args)
    get_word_rel_map(args)