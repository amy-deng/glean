import numpy as np
import os
from math import log
import scipy.sparse as sp

import dgl
import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss
from sklearn.metrics import jaccard_score

 
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

 
def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None

 
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
 
'''
Customized collate function for Pytorch data loader
'''
 
def collate_4(batch):
    batch_data = [item[0] for item in batch]
    s_prob = [item[1] for item in batch]
    r_prob = [item[2] for item in batch]
    o_prob = [item[3] for item in batch]
    return [batch_data, s_prob, r_prob, o_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

 
'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]  
    idx_zero = idx[num_non_zero-1:]  
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r 
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids) 
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None
    
    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero
 
 

'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

 
'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o
 
'''
Empirical distribution
'''
def get_true_distributions(path, data, num_nodes, num_rels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)
    if not os.path.exists(file_path):
        print('build true distributions...',dataset,set_name)
        time_l = list(set(data[:,-1]))
        time_l = sorted(time_l,reverse=False)
        true_prob_s = None
        true_prob_o = None
        true_prob_r = None
        for cur_t in time_l:
            triples = get_data_with_t(data, cur_t)
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
            true_r = np.zeros(num_rels)
            s_arr = triples[:,0]
            o_arr = triples[:,2]
            r_arr = triples[:,1]
            for s in s_arr:
                true_s[s] += 1
            for o in o_arr:
                true_o[o] += 1
            for r in r_arr:
                true_r[r] += 1
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            true_r = true_r / np.sum(true_r)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
                true_prob_r = true_r.reshape(1, num_rels)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, num_rels)), axis=0)
             
        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_s,true_prob_r,true_prob_o], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_s, true_prob_r, true_prob_o] = pickle.load(f)
    true_prob_s = torch.from_numpy(true_prob_s)
    true_prob_r = torch.from_numpy(true_prob_r)
    true_prob_o = torch.from_numpy(true_prob_o)
    return true_prob_s, true_prob_r, true_prob_o 

 

'''
Evaluation metrics
'''
# Label based
 
def print_eval_metrics(true_rank_l, prob_rank_l, prt=True):
    m = MultiLabelBinarizer().fit(true_rank_l)
    m_actual = m.transform(true_rank_l)
    m_predicted = m.transform(prob_rank_l)
    recall = recall_score(m_actual, m_predicted, average='weighted')
    f1 = f1_score(m_actual, m_predicted, average='weighted')
    beta=2
    f2 = fbeta_score(m_actual, m_predicted, average='weighted', beta=beta)
    hloss = hamming_loss(m_actual, m_predicted)
    if prt:
        print("Rec  weighted: {:.4f}".format(recall))
        print("F1  weighted: {:.4f}".format(f1))
        print("F{}  weighted: {:.4f}".format(beta,f2))
        print("hamming loss: {:.4f}".format(hloss))
    return hloss, recall, f1, f2

def print_hit_eval_metrics(total_ranks):
    total_ranks += 1
    mrr = np.mean(1.0 / total_ranks) 
    mr = np.mean(total_ranks)
    hits = []
    for hit in [1, 3, 10]: # , 20, 30
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("Hits @ {}: {:.4f}".format(hit, avg_count))
    # print("MRR: {:.4f} | MR: {:.4f}".format(mrr,mr))
    return hits, mrr, mr

