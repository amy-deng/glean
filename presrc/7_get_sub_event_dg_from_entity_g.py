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
from math import log
import scipy.sparse as sp
print(os.getcwd())

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


 
def comp_deg_norm(g):
    in_deg = g.in_degrees(g.nodes()).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def get_eids(g, r, reverse=False):
    rel_types = g.edata['type']
    eids = (rel_types == r).nonzero(as_tuple=False).view(-1) 
    return eids
 

def make_edge_subgraphs(g, r, reverse=False):
    # g: one day graph g is g[tim]
    eids = get_eids(g, r, reverse)
    src, dst = g.find_edges(eids) # return the source and destination node ID array
   
    in_src_n, in_dst_n, in_eids = g.in_edges(src, 'all') # incoming edges first level
    # in_src_n2, in_dst_n2, in_eids2 = g.in_edges(in_src_n, 'all') # second level
    # multi_layer_eids = torch.cat((eids, in_eids, in_eids2), dim=0)
    multi_layer_eids = torch.cat((eids, in_eids), dim=0)
    sub_g = g.edge_subgraph(multi_layer_eids) # edge and node ids have been changed
    # print( sub_g,sub_g.ndata,sub_g.edata)
    # sub_g.ndata.update({k: g.ndata[k][sub_g.parent_nid] for k in g.ndata if k != 'norm'}) # work in 0.3.x version's dgl
    # sub_g.edata.update({k: g.edata[k][sub_g.parent_eid] for k in g.edata})
    sub_g.ndata.update({k: g.ndata[k][sub_g.ndata[dgl.NID]] for k in g.ndata if k != 'norm'}) 
    sub_g.edata.update({k: g.edata[k][sub_g.edata[dgl.EID]] for k in g.edata})
    sub_g.ids = {}
    norm = comp_deg_norm(sub_g)
    sub_g.ndata['norm'] = norm.view(-1,1)
    node_id = sub_g.ndata['id'].view(-1).tolist()
    sub_g.ids.update(zip(node_id, list(range(sub_g.number_of_nodes()))))
    return sub_g, sub_g.number_of_edges()

def get_g_r_by_r_edge(args):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')
    with open(args.dp + args.dn+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)

    file = os.path.join(args.dp+args.dn, 'dg_r_dict.txt')
    g_r_dict = {}
    r_l = list(range(num_rels))
    for r in r_l:
        print(r,' r done')
        r_dict = {}
        for t in graph_dict:
            g = graph_dict[t]
            sub_g, sub_g_n_edges = make_edge_subgraphs(g, r, False)
            if not sub_g.number_of_edges():
                continue
            else:
                r_dict [t] = sub_g
        g_r_dict[r] = r_dict
    
    with open(file, 'wb') as fp:
        pickle.dump(g_r_dict, fp)
    print(g_r_dict.keys())
    print('dg_r_dict.txt saved! ')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    get_g_r_by_r_edge(args)