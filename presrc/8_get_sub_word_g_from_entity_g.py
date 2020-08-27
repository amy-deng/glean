import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import pickle
import dgl
import torch
# import utils
from math import log
import scipy.sparse as sp
print(os.getcwd())

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])
 
 
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
    # windows = words_l
    # calculate term freq in all docs
    word_window_freq = {}
    for window in windows:
        window = list(set(window))
        for i in range(len(window)):
            if word_window_freq.get(window[i]):
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
    # print(word_window_freq)

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
    # print(word_pair_count)

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
    # print(weight[:10])
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
    # exit()
    return __row, __col, __data

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_big_word_graph(row, col, weight, node_idx):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_idx))
    g.add_edges(row, col)
    g.edata['w'] = torch.from_numpy(np.array(weight)).view(-1, 1) 
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    g.ndata['id'] = torch.from_numpy(node_idx).long().view(-1, 1)
    
    # print(g)
    return g

def get_idx(vocab_l, nodes):
    idx = [vocab_l.index(w) for w in nodes]
    return np.array(idx)  #  np.narray type

def get_token_vocab(args):
    text_f = os.path.join('../data', args.dn, 'text_token.txt')
    with open(text_f) as f:
        tokens_l = f.read().splitlines()  
    tokens_l = [text.split(' ') for text in tokens_l]
    tokens_l = np.array(tokens_l)

    vocab_f = os.path.join('../data', args.dn, 'vocab.txt')
    with open(vocab_f, 'r') as f:
        vocab_l = f.read().splitlines()
    return tokens_l, vocab_l
 
def get_eids(g, r, reverse=False):
    rel_types = g.edata['type'] 
    eids = (rel_types == r).nonzero(as_tuple=False).view(-1) 
    return eids
 
def get_edge_subgraph_ids(g, r, reverse=False):
    # g: one day graph g is g[tim]
    eids = get_eids(g, r, reverse)
    src, dst = g.find_edges(eids) # return the source and destination node ID array
    in_src_n, in_dst_n, in_eids = g.in_edges(src, 'all') # incoming edges
    multi_layer_eids = torch.cat((eids, in_eids), dim=0) 
    eids = g.edata['eid'][multi_layer_eids]
    return eids


def get_word_graph_by_eids(g, text_token, vocab, eids):
    tokens_l = text_token[eids]
    nodes = [item for sublist in tokens_l for item in sublist]
    nodes = list(set(nodes))  
    nodes = [x for x in nodes if x in vocab]
    node_idx = get_idx(vocab, nodes)
    row, col, weight = get_pmi(tokens_l, nodes)
    wg = get_big_word_graph(row, col, weight, node_idx)
    return wg

def get_word_g_by_r_edge(args):
    # edge_graph
    tokens_l, vocab_l = get_token_vocab(args)
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')
    rel_lists = list(range(num_rels))
    if args.k > 0:
        # random sample k from all rels
        random_rel_lists = rel_lists[:args.k]
        file = os.path.join(args.dp+args.dn, 'wg_r_dict_truncated_rand_{}.txt'.format(args.k))
        random_rel_lists_file = os.path.join(args.dp+args.dn, 'rel_rand_{}.txt'.format(args.k))
    else:
        random_rel_lists = rel_lists
        file = os.path.join(args.dp+args.dn, 'wg_r_dict_truncated.txt')
        random_rel_lists_file = None
    print(num_rels,random_rel_lists)
    with open(args.dp + args.dn+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)
    wg_r_dict = {}
    for r in random_rel_lists:
        if r % 10 == 0:
            print(r,'done')
        r_dict = {}
        for t in graph_dict:
            g = graph_dict[t]
            eids = get_edge_subgraph_ids(g, r, False)
            if len(eids):
                wg = get_word_graph_by_eids(g, tokens_l, vocab_l, eids)
                r_dict[t] = wg
        wg_r_dict[r] = r_dict
    
    with open(file, 'wb') as fp:
        pickle.dump(wg_r_dict, fp)
    print(wg_r_dict.keys())
    print(file, 'saved! ')
    if random_rel_lists_file:
        with open(random_rel_lists_file, 'wb') as fp:
            pickle.dump(random_rel_lists, fp)
        print(random_rel_lists_file, 'saved! ')

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")
    ap.add_argument("--k", default=20, type=int, help="consider random k relations")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    get_word_g_by_r_edge(args)