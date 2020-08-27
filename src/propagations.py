import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import math
# Graph Propagation models

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)
        return {'h' : h}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.0):
        super().__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, feature):
        def gcn_msg(edge):
            msg = edge.src['h'] * edge.data['w'].float()
            return {'m': msg}

        # feature = g.ndata['h']
        if self.dropout:
            feature = self.dropout(feature)

        g.ndata['h'] = feature
        g.update_all(gcn_msg, fn.sum(msg='m', out='h'))
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
 
 
# CompGCN based on direct graphs. We do not have inversed edges
class CompGCN_dg(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)     
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, g, reverse=False):  
      
        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['e_h'])
            return {'e_h': e_h}

        g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.mean('m', 'h_o_r')) 
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        g.ndata['h_s_r_o'] = h_o_r 
        g.update_all(fn.copy_src(src='h_s_r_o', out='m'), fn.sum(msg='m', out='h'), apply_func)
        g.apply_edges(apply_edge)

 
 
 