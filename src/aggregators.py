import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from propagations import *
from modules import *
import time
from itertools import groupby


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout)) # activation or None

    def forward(self, g, features=None): # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

 

# aggregator for event forecasting 
class aggregator_event(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.se_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)

        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')       

    def forward(self, t_list, ent_embeds, rel_embeds, word_embeds, graph_dict, word_graph_dict, ent_map, rel_map):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)  
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        # a = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]).data.cpu()
        # print(a.is_cuda)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        # word graph
        wg_list = [word_graph_dict[tim.item()] for tim in unique_t]
        batched_wg = dgl.batch(wg_list) 
        batched_wg = batched_wg.to(device)
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        #     move_dgl_to_cuda(batched_wg)
        batched_wg.ndata['h'] = word_embeds[batched_wg.ndata['id']].view(-1, word_embeds.shape[1]) 

        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)
        batched_wg.ndata['h'] = self.se_aggr(batched_wg) 
        word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
        id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))
        
        # cpu operation for nodes
        g_node_embs = batched_g.ndata.pop('h').data.cpu()
        g_node_ids = batched_g.ndata['id'].view(-1)
        max_query_ent = 0
        num_nodes = len(g_node_ids)
        c_g_node_ids = g_node_ids.data.cpu().numpy()
        c_unique_ent_id = list(set(c_g_node_ids))
        ent_gidx_dict = {} # entid: [[gidx],[word_idx]]
        for ent_id in c_unique_ent_id:
            word_ids = ent_map[ent_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_node_ids==ent_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                ent_gidx_dict[ent_id] = [gidx, word_idx]
                max_query_ent = max(max_query_ent, len(word_idx))

        # cpu operation on edges
        g_edge_embs = batched_g.edata.pop('e_h').data.cpu() ####
        g_edge_types = batched_g.edata['type'].view(-1)
        num_edges = len(g_edge_types)
        max_query_rel = 0
        c_g_edge_types = g_edge_types.data.cpu().numpy()
        c_unique_type_id = list(set(c_g_edge_types))
        type_gidx_dict = {}
        for type_id in c_unique_type_id:
            word_ids = rel_map[type_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_edge_types==type_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                type_gidx_dict[type_id] = [gidx, word_idx]
                max_query_rel = max(max_query_rel, len(word_idx))

        max_query = max(max_query_ent, max_query_rel)
        # initialize a batch
        wg_node_embs = batched_wg.ndata['h'].data.cpu()
        Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
        Q_mx_rel = g_edge_embs.view(num_edges , 1, self.h_dim)
        Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
        H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))
        
        for ent in ent_gidx_dict:
            [gidx, word_idx] = ent_gidx_dict[ent]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        for e_type in type_gidx_dict:
            [gidx, word_idx] = type_gidx_dict[e_type]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        if torch.cuda.is_available():
            H_mx = H_mx.cuda()
            Q_mx = Q_mx.cuda()
        output, weights = self.attn(Q_mx, H_mx) # output (batch,1,h_dim)
        batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
        batched_g.edata['e_h'] = output[num_nodes:].view(-1, self.h_dim)
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
            global_word_info = dgl.max_nodes(batched_wg, 'h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')
            global_word_info = dgl.mean_nodes(batched_wg, 'h')
        global_node_info = torch.cat((global_node_info, global_edge_info, global_word_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 3*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero


 
# aggregator for actor forecasting 
class aggregator_actor(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)
        self.se_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)
        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')

    def forward(self, t, r, r_hist, r_hist_t, ent_embeds, rel_embeds, word_embeds, graph_dict, word_graph_dict, ent_map, rel_map):
        reverse = False
        batched_g, batched_wg, len_non_zero, r_ids_graph, idx, num_non_zero = get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if batched_g: 
            r_sort = r[idx]
            batched_g = batched_g.to(device) # torch.device('cuda:0')
            batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
            if torch.cuda.is_available():
                type_data = batched_g.edata['type'].cuda()
            else:
                type_data = batched_g.edata['type']
            batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)
            # if torch.cuda.is_available():
            #     move_dgl_to_cuda(batched_g)
                 
            self.re_aggr1(batched_g, reverse)
            self.re_aggr2(batched_g, reverse)
            embeds_g_r = batched_g.edata.pop('e_h')
            embeds_g_r = embeds_g_r[torch.LongTensor(r_ids_graph)].data.cpu()
             
            if batched_wg:
                batched_wg = batched_wg.to(device) 
                batched_wg.ndata['h'] = word_embeds[batched_wg.ndata['id']].view(-1, word_embeds.shape[1])
                # if torch.cuda.is_available():
                #     move_dgl_to_cuda(batched_wg)
                batched_wg.ndata['h'] = self.se_aggr(batched_wg)
                
                word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
                id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))
                g_node_embs = batched_g.ndata.pop('h').data.cpu()
                g_node_ids = batched_g.ndata['id'].view(-1)
                max_query_ent = 0
                num_nodes = len(g_node_ids)
                # cpu operation for nodes
                c_g_node_ids = g_node_ids.data.cpu().numpy()
                c_unique_ent_id = list(set(c_g_node_ids))
                ent_gidx_dict = {} # entid: [[gidx],[word_idx]]
                for ent_id in c_unique_ent_id:
                    word_ids = ent_map[ent_id]
                    word_idx = []
                    for w in word_ids:
                        try:
                            word_idx.append(id_dict[w])
                        except:
                            continue
                    if len(word_idx)>1:
                        gidx = (c_g_node_ids==ent_id).nonzero()[0]
                        word_idx = torch.LongTensor(word_idx)
                        ent_gidx_dict[ent_id] = [gidx, word_idx]
                        max_query_ent = max(max_query_ent, len(word_idx))
                 
                # cpu operation for rel
                num_edges = len(embeds_g_r)
                max_query_rel = 0
                c_r_sort = r_sort.data.cpu().numpy()
                 
                type_gidx_dict_one = {} # typeid: [[gidx, word_idx]]
                for i in range(len(r_sort)):
                    type_id = c_r_sort[i]
                    word_ids = rel_map[type_id]
                    word_idx = []
                    for w in word_ids:
                        try:
                            word_idx.append(id_dict[w])
                        except:
                            continue
                    if len(word_idx)>1:
                        word_idx = torch.LongTensor(word_idx)
                        # print(i,r_ids_graph[i],'====')
                        type_gidx_dict_one[r_ids_graph[i]] = word_idx
                        max_query_rel = max(max_query_rel, len(word_idx))
                 
                 
                max_query = max(max_query_ent, max_query_rel,1)
                # initialize a batch
                wg_node_embs = batched_wg.ndata['h'].data.cpu()
                Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
                Q_mx_rel = embeds_g_r.view(num_edges , 1, self.h_dim)
                Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
                H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))
                 
                for ent in ent_gidx_dict:
                    [gidx, word_idx] = ent_gidx_dict[ent]
                    embeds = wg_node_embs.index_select(0, word_idx)
                    if len(gidx) > 1: 
                        for i in gidx:
                            H_mx[i,range(len(word_idx)),:] = embeds
                    else:
                        H_mx[gidx,range(len(word_idx)),:] = embeds
                 
                ii = num_nodes
                for e_id in type_gidx_dict_one: # some rel do not have corresponding words
                    word_idx = type_gidx_dict_one[e_id]
                    H_mx[ii,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
                    ii += 1
                 
                if torch.cuda.is_available():
                    H_mx = H_mx.cuda()
                    Q_mx = Q_mx.cuda()
                
                output, weights = self.attn(Q_mx, H_mx) # output (batch,1,h_dim)
                 
                batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
                embeds_g_r = output[num_nodes:].view(-1, self.h_dim)
            g_list = dgl.unbatch(batched_g)
            node_emb_temporal = np.zeros((self.num_nodes, self.seq_len, self.h_dim))
            for i in range(len(g_list)):
                g = g_list[i]
                feature = g.ndata['h'].data.cpu().numpy()
                indices = g.ndata['id'].data.cpu().view(-1).numpy()
                node_emb_temporal[indices,i,:]  = feature

            node_emb_temporal = torch.FloatTensor(node_emb_temporal)
            if torch.cuda.is_available():
                node_emb_temporal = node_emb_temporal.cuda()
 
            embeds_split = torch.split(embeds_g_r, len_non_zero.tolist())
            embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 1 * self.h_dim)
            if torch.cuda.is_available():
                embed_seq_tensor = embed_seq_tensor.cuda()
            for i, embeds in enumerate(embeds_split): 
                embed_seq_tensor[i, torch.arange(0,len(embeds)), :] = embeds
            embed_seq_tensor = self.dropout(embed_seq_tensor)
        else:
            node_emb_temporal = None
            embed_seq_tensor = None
        return embed_seq_tensor, len_non_zero, idx, node_emb_temporal 

