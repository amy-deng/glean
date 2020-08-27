import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import pandas as pd
import argparse
import scipy.sparse as sp
import dgl
import torch
import pickle
from math import log

print(os.getcwd())

 
def get_indices_with_t(data, time):
    idx = [i for i in range(len(data)) if data[i][3] == time]
    return np.array(idx)

def get_token_vocab(args):
    text_f = os.path.join(args.dp, args.dn, 'text_token.txt')
    with open(text_f) as f:
        tokens_l = f.read().splitlines()  
    tokens_l = [text.split(' ') for text in tokens_l]
    tokens_l = np.array(tokens_l)

    vocab_f = os.path.join(args.dp, args.dn, 'vocab.txt')
    with open(vocab_f, 'r') as f:
        vocab_l = f.read().splitlines()
    return tokens_l, vocab_l

def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_idx(vocab_l, nodes):
    # nodes = [node for n in nodes if n in vocab_l]
    # idx = [vocab_l.index(w) for w in nodes if w in vocab_l]
    idx = [vocab_l.index(w) for w in nodes]
    return np.array(idx)  #  np.narray type
 
# use words
def get_pmi(words_l, nodes):
    vocab = nodes
    vocab_size = len(vocab)
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
    # # filter words not in vocab
    new_words_l = []
    for l in words_l:
        l = set(l[:100]) # unique
        new_words_l.append([w for w in l if w in nodes])
    windows = new_words_l
    # calculate term freq in all docs
    word_window_freq = {}
    for window in windows:
        window = list(set(window))
        for i in range(len(window)):
            if word_window_freq.get(window[i]):
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):

                word_i = window[i]
                word_j = window[j]

                if word_id_map.get(word_i) != None and word_id_map.get(word_j) != None:

                    word_i_id = word_id_map[word_i]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count * num_window) / (1.0 * word_freq_i * word_freq_j))
        # print(pmi,'before round pmi')
        pmi = round(pmi, 3)
        # print(pmi,'round pmi')
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
    adj = sp.csr_matrix((weight, (row, col)), shape=(vocab_size, vocab_size))
    adj_new = normalize_adj(adj)
    _row = adj_new.row
    _col = adj_new.col
    _data = adj_new.data
    _data = [ round(elem, 3) for elem in _data ]
    __row, __col, __data = [], [], []
    for k in range(len(_data)):
        if _data[k] > 0:
            __row.append(_row[k])
            __col.append(_col[k])
            __data.append(_data[k])
        # else:
            # print('empty')
    return __row, __col, __data

def check_exist(outf):
    return os.path.isfile(outf)
 
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

 
# undirected graph
def get_big_word_graph(row, col, weight, node_idx, vocab_s=None):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_idx))
    g.add_edges(row, col)
    g.edata['w'] = torch.from_numpy(np.array(weight)).view(-1, 1) 
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    # print(g.edges())
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)
    g.ndata['id'] = torch.from_numpy(node_idx).long().view(-1, 1)
    
    # print(g)
    return g

def build_word_graphs(args):
    file = os.path.join(args.dp+args.dn, 'wg_dict_truncated.txt')
    if not check_exist(file):
        total_data, total_time = load_quadruples(args.dp+args.dn, 'train.txt', 'valid.txt', 'test.txt')

        word_graph_dict_full = {}
        text_l, vocab_l = get_token_vocab(args)
        word_graph_dict_full = {}
        for time in total_time:
            idx = get_indices_with_t(total_data, time)
            text_l_by_time = text_l[idx]
            nodes = [item for sublist in text_l_by_time for item in sublist]
            nodes = list(set(nodes))
            nodes = [x for x in nodes if x in vocab_l] # filter empty strings
            node_idx = get_idx(vocab_l, nodes)
            row, col, weight = get_pmi(text_l_by_time, nodes) # words here
            if time % 200 == 0:
                print('len of ',len(nodes), ' time',time, '\tof ',max(total_time))
            
            word_graph_dict_full[time] = get_big_word_graph(row, col, weight, node_idx, len(vocab_l))
        with open(file, 'wb') as fp:
            pickle.dump(word_graph_dict_full, fp)
        print(file,'saved')
    else:
       print(file,'exists')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")
    ap.add_argument("--dim", default=100, type=int, help="dim of word embedding")

    args=ap.parse_args()
    print(args)


    build_word_graphs(args)  