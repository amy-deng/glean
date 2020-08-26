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
    num_non_zero = len(torch.nonzero(r_len))
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
def get_scaled_tr_dataset(num_nodes, dataset='india', set_name='train', seq_len=10, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '../data/{}/tr_data_{}_sl{}_rand_{}.npy'.format(dataset, set_name, seq_len, num_r)
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

def get_tr_dataset(num_nodes, dataset='india', set_name='train', seq_len=10):
    import pandas as pd
    from scipy import sparse
    if seq_len < 10:
        file_path = '../data/{}/tr_data_{}_sl{}.npy'.format(dataset, set_name, seq_len)
        print(file_path,'new')
    else:
        file_path = '../data/{}/tr_data_{}.npy'.format(dataset, set_name)
    if not os.path.exists(file_path):
        print('build tr_data ...',dataset,set_name)
        data, times = load_quadruples('../data/' + dataset, set_name + '.txt') # triplets
        if seq_len < 10:
            filename = '../data/{}/{}_history_rel_{}.txt'.format(dataset, set_name, seq_len)
            print(filename,'read')
            with open('../data/{}/{}_history_rel_{}.txt'.format(dataset, set_name, seq_len), 'rb') as f: # r history data
                r_history_data = pickle.load(f)
        else:
            with open('../data/{}/{}_history_rel.txt'.format(dataset, set_name), 'rb') as f: # r history data
                r_history_data = pickle.load(f)
        hist_r, hist_r_t = r_history_data[0], r_history_data[1]

        df = pd.DataFrame(data=data, columns=['s','r','o','t'])  
        trdf = df.groupby(['t','r']).size().reset_index().rename(columns={0:'count'})
        t_data = []
        r_data = []
        true_prob_s = None
        true_prob_o = None  
        r_hist = []
        r_hist_t = []
        for i,row in trdf.iterrows():
            if i % 500 == 0:
                print(i)
            if row['t'] == 0 or row['t'] == 1096 : # do not train samples of t=0, r history data can be empty
                continue
            idx = get_data_idx_with_t_r(data, row['t'], row['r']) # shouldn't return None
            cur_hist_r = hist_r[idx]
            cur_hist_r_t = hist_r_t[idx]
            ## if len(cur_hist_r_t)  < 2: # history data less than 2
            ##     continue
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
            # if i ==500 :
            #     break
        t_data = np.array(t_data)
        r_data = np.array(r_data)
        true_prob_s = sparse.csr_matrix(true_prob_s)
        true_prob_o = sparse.csr_matrix(true_prob_o)
        with open(file_path, 'wb') as fp:
            pickle.dump([t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o], fp)
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
def get_true_distributions(data, num_nodes, num_rels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    file_path = '../data/{}/true_probs_{}.npy'.format(dataset, set_name)
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
def multilabel_f1beta(actual, predicted, beta=2, average='macro'): # list of lists
    """
    average = [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    score_micro = fbeta_score(m_actual, m_predicted, average='micro', beta=beta)
    score_macro = fbeta_score(m_actual, m_predicted, average='macro', beta=beta)
    score_weighted = fbeta_score(m_actual, m_predicted, average='weighted', beta=beta)
    score_samples = fbeta_score(m_actual, m_predicted, average='samples', beta=beta)
    return score_micro, score_macro, score_weighted, score_samples
 
def multilabel_f1(actual, predicted, average='macro'): # list of lists
    """
    average = [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    score_micro = f1_score(m_actual, m_predicted, average='micro')
    score_macro = f1_score(m_actual, m_predicted, average='macro')
    score_weighted = f1_score(m_actual, m_predicted, average='weighted')
    score_samples = f1_score(m_actual, m_predicted, average='samples')
    return score_micro, score_macro, score_weighted, score_samples
 
def multilabel_recall(actual, predicted, average='macro'): # list of lists
    """
    average = [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    score_micro = recall_score(m_actual, m_predicted, average='micro')
    score_macro = recall_score(m_actual, m_predicted, average='macro')
    score_weighted = recall_score(m_actual, m_predicted, average='weighted')
    score_samples = recall_score(m_actual, m_predicted, average='samples')
    return score_micro, score_macro, score_weighted, score_samples
 
def multilabel_precision(actual, predicted, average='macro'): # list of lists
    """
    average = [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    score_micro = precision_score(m_actual, m_predicted, average='micro')
    score_macro = precision_score(m_actual, m_predicted, average='macro')
    score_weighted = precision_score(m_actual, m_predicted, average='weighted')
    score_samples = precision_score(m_actual, m_predicted, average='samples')
    return score_micro, score_macro, score_weighted, score_samples
 
def multilabel_hamming_loss(actual, predicted): # list of lists
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    hloss = hamming_loss(m_actual, m_predicted)
    return hloss
 
 
def multilabel_jaccard_score(actual, predicted): # list of lists
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html
    """
    m = MultiLabelBinarizer().fit(actual)
    m_actual = m.transform(actual)
    m_predicted = m.transform(predicted)
    score_micro = jaccard_score(m_actual, m_predicted, average='micro')
    score_macro = jaccard_score(m_actual, m_predicted, average='macro')
    score_weighted = jaccard_score(m_actual, m_predicted, average='weighted')
    score_samples = jaccard_score(m_actual, m_predicted, average='samples')
    return score_micro, score_macro, score_weighted, score_samples
 

def print_eval_metrics2(true_rank_l, prob_rank_l, prt=True):
    # a1, a2, a3, a4 = multilabel_jaccard_score(true_rank_l, prob_rank_l)
    # accuracy = [a1, a2, a3, a4]
    # print("Acc micro: {:.4f} | macro: {:.4f} | weighted: {:.4f} | sample: {:.4f}".format(
    #         a1, a2, a3, a4))
    
    # b1, b2, b3, b4 = multilabel_precision(true_rank_l, prob_rank_l)
    # precision = [b1, b2, b3, b4]
    # print("Prec micro: {:.4f} | macro: {:.4f} | weighted: {:.4f} | sample: {:.4f}".format(
    #         b1, b2, b3, b4))
    
    c1, c2, c3, c4 = multilabel_recall(true_rank_l, prob_rank_l)
    recall = [c1, c2, c3, c4]
    d1, d2, d3, d4 = multilabel_f1(true_rank_l, prob_rank_l)
    f1 = [d1, d2, d3, d4]
    beta=2
    e1, e2, e3, e4 = multilabel_f1beta(true_rank_l, prob_rank_l,beta=2)
    f2 = [e1, e2, e3, e4]
    hloss = multilabel_hamming_loss(true_rank_l, prob_rank_l)
    if prt:
        print("Rec  weighted: {:.4f}".format(c3))
        print("F1  weighted: {:.4f}".format(d3))
        print("F{}  weighted: {:.4f}".format(beta,e3))
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

