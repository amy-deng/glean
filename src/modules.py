import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import *

 
# general, dot and add (ictive) attention (cat and linear)
class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general','add']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=True)
        if self.attention_type == 'add':
            self.linear_in = nn.Linear(2* dimensions, dimensions//2, bias=True)
            self.v = nn.Parameter(torch.Tensor(dimensions//2, 1))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        
        query_len = context.size(1)
        # print(query.size(),context.size(), query_len)
        if self.attention_type == 'add':
            querys = query.repeat(1,query_len,1) # [B,H] -> [T,B,H]
            feats = torch.cat((querys, context), dim=-1) # [B,T,H*2]
            energy = self.tanh(self.linear_in(feats)) # [B,T,H*2] -> [B,T,H]
            # compute attention scores
            v = self.v.t().repeat(batch_size,1,1) # [H,1] -> [B,1,H]
            energy = energy.permute(0,2,1)#.contiguous() #[B,H,T]
            attention_weights = torch.bmm(v, energy) # [B,1,H]*[B,H,T] -> [B,1,T]
            # weight values
            mix = torch.bmm(attention_weights, context)#.squeeze(1) # [B,1,T]*[B,T,H] -> [B,H]
            # concat -> (batch_size * output_len, 2*dimensions)
            combined = torch.cat((mix, query), dim=2)
            combined = combined.view(batch_size * output_len, 2 * dimensions)
            # Apply linear_out on every 2nd dimension of concat
            # output -> (batch_size, output_len, dimensions)
            output = self.linear_out(combined).view(batch_size, output_len, dimensions)
            output = self.tanh(output)
            return output, attention_weights
            
        elif self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights
