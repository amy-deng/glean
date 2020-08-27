import glob
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
import torch
import dgl
import scipy.sparse as sp
import os
import time
import argparse
import string
from math import log
print(os.getcwd())


def check_exist(outf):
    return os.path.isfile(outf)
 
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_token_vocab(args):
    text_f = os.path.join('../data', args.dn, 'text_token.txt')
    with open(text_f) as f:
        tokens_l = f.read().splitlines()  
    tokens_l = [text.split(' ') for text in tokens_l]
    tokens_l = np.array(tokens_l,dtype=object)

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

def get_idx(vocab_l, nodes):
    idx = [vocab_l.index(w) for w in nodes]
    return np.array(idx)  #  np.narray type

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
    return g


def get_word_graph_by_eids(g, text_token, vocab, eids):
    tokens_l = text_token[eids]
    nodes = [item for sublist in tokens_l for item in sublist]
    nodes = list(set(nodes))  
    nodes = [x for x in nodes if x in vocab]
    node_idx = get_idx(vocab, nodes)
    row, col, weight = get_pmi(tokens_l, nodes)
    wg = get_big_word_graph(row, col, weight, node_idx)
    return wg


def get_random_r(args):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')
    rel_lists = list(range(num_rels))
    np.random.shuffle(rel_lists)
    random_rel_lists = rel_lists[:args.k]
    return random_rel_lists

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])



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
 

def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None

 

def get_tr_dataset(args, set_name):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')
    file_path = '{}{}/tr_data_{}_sl{}.npy'.format(args.dp, args.dn, set_name, args.seq_len)
    if not os.path.exists(file_path):
        print('build tr_data ...',args.dn,set_name)
        data, times = load_quadruples(args.dp+args.dn, set_name + '.txt') # triplets
        filename = '{}{}/{}_history_rel_{}.txt'.format(args.dp, args.dn, set_name, args.seq_len)
        with open('{}{}/{}_history_rel_{}.txt'.format(args.dp, args.dn, set_name, args.seq_len), 'rb') as f: # r history data
            r_history_data = pickle.load(f)
        hist_r, hist_r_t = r_history_data[0], r_history_data[1]

        df = pd.DataFrame(data=data, columns=['s','r','o','t'])  
        trdf = df.groupby(['t','r']).size().reset_index().rename(columns={0:'count'})
        t_data, r_data = [], []
        true_prob_s, true_prob_o = None, None
        r_hist, r_hist_t = [], []
        for i,row in trdf.iterrows():
            if i % 500 == 0:
                print(i)
            if row['t'] == 0 or row['t'] == 1096 : # do not train samples of t=0, r history data can be empty
                continue
            idx = get_data_idx_with_t_r(data, row['t'], row['r']) # shouldn't return None
            cur_hist_r = hist_r[idx]
            cur_hist_r_t = hist_r_t[idx]
            t_data.append(row['t'])
            r_data.append(row['r'])
            r_hist.append(cur_hist_r)
            r_hist_t.append(cur_hist_r_t)

            sodf = df.loc[(df['r']==row['r']) & (df['t']==row['t'])][['s','o']]
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
            so_arr = sodf.values
            for s in so_arr[:,0]:
                true_s[s] += 1
            for o in so_arr[:,1]:
                true_o[o] += 1

            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
            else:                
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
        t_data = np.array(t_data)
        r_data = np.array(r_data)
        true_prob_s = sp.csr_matrix(true_prob_s)
        true_prob_o = sp.csr_matrix(true_prob_o)
        with open(file_path, 'wb') as fp:
            pickle.dump([t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o], fp)
    else:
        print('load tr_data ...',args.dn,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o


def get_scaled_tr(args, r_list, set_name='train'):
    new_file_path = args.dp + args.dn + '/tr_data_{}_sl{}_rand_{}.npy'.format(set_name, args.seq_len, args.k)
    if not check_exist(new_file_path):
        # load tr_datasets
        file_path = args.dp + args.dn + '/tr_data_{}_sl{}.npy'.format(set_name, args.seq_len)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
        new_t_data = []
        new_r_data = []
        new_r_hist = []
        new_r_hist_t = []
        idx = [i for i in range(len(r_data)) if r_data[i] in r_list]
        r_data = r_data[idx]
        t_data = t_data[idx]
        r_hist = [r_hist[i] for i in range(len(r_hist)) if i in idx]
        r_hist_t = [r_hist_t[i] for i in range(len(r_hist_t)) if i in idx]
        true_prob_s = true_prob_s.toarray()
        true_prob_o = true_prob_o.toarray()
        true_prob_s = true_prob_s[idx]
        true_prob_o = true_prob_o[idx]
        true_prob_s = sp.csr_matrix(true_prob_s)
        true_prob_o = sp.csr_matrix(true_prob_o)  
        with open(new_file_path, 'wb') as fp:
            pickle.dump([t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o], fp)
        print(new_file_path,'saved!')
    else:
        print(new_file_path,'exists!')

 

def build_scaled_wg_r(args, r_list):
    print('scale sub word graph dict')
    new_file_path = args.dp + args.dn + '/wg_r_dict_sl{}_rand_{}.txt'.format(args.seq_len, args.k)
    if not check_exist(new_file_path):
        tokens_l, vocab_l = get_token_vocab(args)
        with open(args.dp + args.dn+'/dg_dict.txt', 'rb') as f:
            graph_dict = pickle.load(f)

        wg_r_dict = {}
        for r in r_list:
            if r % 10 == 0:
                print(r,'r  (r % 10 == 0)')
            r_dict = {}
            for t in graph_dict:
                g = graph_dict[t]
                eids = get_edge_subgraph_ids(g, r, False)
                if len(eids):
                    wg = get_word_graph_by_eids(g, tokens_l, vocab_l, eids)
                    r_dict[t] = wg
            wg_r_dict[r] = r_dict

        
        with open(new_file_path, 'wb') as fp:
            pickle.dump(wg_r_dict, fp)
        
        print(new_file_path,'saved!')
    else:
        print(new_file_path,'exists!')


def get_scaled_g_r(args, r_list):
    print('sub entity graph dict')
    new_file_path = args.dp + args.dn + '/dg_r_dict_sl{}_rand_{}.txt'.format(args.seq_len, args.k)
    if not check_exist(new_file_path):
        file_path = args.dp + args.dn + '/dg_r_dict.txt'
        with open(file_path, 'rb') as f:
            graph_dict = pickle.load(f)
        new_graph_dict = {}
        for r in r_list:
            new_graph_dict[r] = graph_dict[r]
        
        with open(new_file_path, 'wb') as fp:
            pickle.dump(new_graph_dict, fp)
        
        print(new_file_path,'saved!')
    else:
        print(new_file_path,'exists!')

def saveRlist(args, r_list):
    if not r_list:
        print('r_list is empty  exit()')
        exit()
    file_path = args.dp + args.dn + '/r_list_sl{}_rand_{}.txt'.format(args.seq_len, args.k)
    if not check_exist(file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(r_list, fp)
            print(file_path,'saved!')
    else:
        print(file_path,'exists!')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")
    ap.add_argument("--k", default=20, type=int, help="top k event types")
    ap.add_argument("--seq-len", default=7, type=int, help="")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    # get tr dataset (predict s/o given r)
    get_tr_dataset(args,'train')
    get_tr_dataset(args,'valid')
    get_tr_dataset(args,'test')

    # exit()
    r_list = get_random_r(args)
    print(r_list,'random r list - ',args.k)
    # save it ......
    saveRlist(args, r_list) # build data for baseline methods
    build_scaled_wg_r(args,r_list)
    get_scaled_g_r(args,r_list)
    get_scaled_tr(args,r_list,'train')
    get_scaled_tr(args,r_list,'valid')
    get_scaled_tr(args,r_list,'test')
    
    