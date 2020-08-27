import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch
import sys

print(sys.argv)
try:
    history_len = int(sys.argv[1])
    inPath = sys.argv[2]
except:
    print('Usuage: history_len (e.g., 7), path (../data/AFG/)')
    exit()
print('history_len =',history_len, 'path',inPath)

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


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])
 

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def get_data_with_ts(data, t_min, t_max):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if (quad[3] <= t_max) and (quad[3] >= t_min)]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g



train_data, train_times = load_quadruples(inPath, 'train.txt')
test_data, test_times = load_quadruples(inPath, 'test.txt')
dev_data, dev_times = load_quadruples(inPath, 'valid.txt')
total_data, total_times = load_quadruples(inPath, 'train.txt', 'valid.txt', 'test.txt')

num_e, num_r = get_total_number(inPath, 'stat.txt')

 
# '''
# rel
print('rel info...')
#################
r_his = [[] for _ in range(num_r)]
r_his_t = [[] for _ in range(num_r)]
r_history_data = [[] for _ in range(len(train_data))]
r_history_data_t = [[] for _ in range(len(train_data))]

latest_t = 0
r_his_cache = [[] for _ in range(num_r)]
r_his_cache_t = [None for _ in range(num_r)]


for i, train in enumerate(train_data):
    if i % 10000 == 0:
        print("train", i, len(train_data))
    # if i == 10000:
    #     break
    t = train[3]
    if latest_t != t:
        for rr in range(num_r):
            if len(r_his_cache[rr]) != 0:
                if len(r_his[rr]) >= history_len:
                    r_his[rr].pop(0)
                    r_his_t[rr].pop(0)

                r_his[rr].append(r_his_cache[rr].copy())
                r_his_t[rr].append(r_his_cache_t[rr])
                r_his_cache[rr] = []
                r_his_cache_t[rr] = None 
        latest_t = t
    s = train[0]
    r = train[1]
    o = train[2]
    r_history_data[i] = r_his[r].copy()
    r_history_data_t[i] = r_his_t[r].copy() 
    if len(r_his_cache[r]) == 0:
        r_his_cache[r] = np.array([[s, o]])
    else:
        r_his_cache[r] = np.concatenate((r_his_cache[r], [[s, o]]), axis=0)
    r_his_cache_t[r] = t

print(len(r_history_data),len(r_history_data_t), 'train_history_rel')
with open(os.path.join(inPath, 'train_history_rel_{}.txt'.format(history_len)), 'wb') as fp:
    pickle.dump([r_history_data, r_history_data_t], fp)

r_history_data_dev = [[] for _ in range(len(dev_data))]
r_history_data_dev_t = [[] for _ in range(len(dev_data))]

for i, dev in enumerate(dev_data):
    if i % 10000 == 0:
        print("valid", i, len(dev_data))
    t = dev[3]
    if latest_t != t:
        for rr in range(num_r):
            if len(r_his_cache[rr]) != 0:
                if len(r_his[rr]) >= history_len:
                    r_his[rr].pop(0)
                    r_his_t[rr].pop(0)
                r_his_t[rr].append(r_his_cache_t[rr])
                r_his[rr].append(r_his_cache[rr].copy())
                r_his_cache[rr] = []
                r_his_cache_t[rr] = None
            
        latest_t = t
    s = dev[0]
    r = dev[1]
    o = dev[2]
    r_history_data_dev[i] = r_his[r].copy()
    r_history_data_dev_t[i] = r_his_t[r].copy()
    if len(r_his_cache[r]) == 0:
        r_his_cache[r] = np.array([[s, o]])
    else:
        r_his_cache[r] = np.concatenate((r_his_cache[r], [[s, o]]), axis=0)
    r_his_cache_t[r] = t

print(len(r_history_data_dev),len(r_history_data_dev_t), 'valid_history_rel') 
with open(os.path.join(inPath, 'valid_history_rel_{}.txt'.format(history_len)), 'wb') as fp:
    pickle.dump([r_history_data_dev, r_history_data_dev_t], fp)


r_history_data_test = [[] for _ in range(len(test_data))]
r_history_data_test_t = [[] for _ in range(len(test_data))]


for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print("valid", i, len(test_data))
    t = test[3]
    if latest_t != t:
        for rr in range(num_r):
            if len(r_his_cache[rr]) != 0:
                if len(r_his[rr]) >= history_len:
                    r_his[rr].pop(0)
                    r_his_t[rr].pop(0)
                r_his_t[rr].append(r_his_cache_t[rr])
                r_his[rr].append(r_his_cache[rr].copy())
                r_his_cache[rr] = []
                r_his_cache_t[rr] = None
            
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    r_history_data_test[i] = r_his[r].copy()
    r_history_data_test_t[i] = r_his_t[r].copy()
    if len(r_his_cache[r]) == 0:
        r_his_cache[r] = np.array([[s, o]])
    else:
        r_his_cache[r] = np.concatenate((r_his_cache[r], [[s, o]]), axis=0)
    r_his_cache_t[r] = t

print(len(r_history_data_test),len(r_history_data_test_t), 'test_history_rel') 
with open(os.path.join(inPath, 'test_history_rel_{}.txt'.format(history_len)), 'wb') as fp:
    pickle.dump([r_history_data_test, r_history_data_test_t], fp)
    # print(train)

 