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
print(os.getcwd())

# get direct graph

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

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

def get_indices_with_t(data, time):
    idx = [i for i in range(len(data)) if data[i][3] == time]
    return np.array(idx)

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def check_exist(outf):
    return os.path.isfile(outf)
 
def get_all_graph_dict(args):
    file = os.path.join(args.dp+args.dn, 'dg_dict.txt')
    if not check_exist(file):
        num_e, num_r = get_total_number(args.dp+args.dn, 'stat.txt')

        graph_dict = {}
        total_data, total_times = load_quadruples(args.dp+args.dn, 'train.txt', 'valid.txt', 'test.txt')
        print(total_data.shape,total_times.shape)

        for time in total_times:
            if time % 100 == 0:
                print(str(time)+'\tof '+str(max(total_times)))
            data = get_data_with_t(total_data, time)
            edge_indices = get_indices_with_t(total_data, time) # search from total_data (unsplitted)

            g = get_big_graph_w_idx(data, num_r, edge_indices)
            graph_dict[time] = g
        
        with open(file, 'wb') as fp:
            pickle.dump(graph_dict, fp)
        print('dg_dict.txt saved! ')
    else:
        print('dg_dict.txt exists! ')

def get_big_graph_w_idx(data, num_rels, edge_indices):  
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)  
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    g.add_edges(src, dst, {'eid': torch.from_numpy(edge_indices)}) # array list
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    get_all_graph_dict(args)