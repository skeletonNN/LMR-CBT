from typing_extensions import final
import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

class CrossModalAttentionLayer(nn.Module):
    # y attends x
    def __init__(self, k, x_channels, y_size, spatial=True):
        super(CrossModalAttentionLayer, self).__init__()
        self.k = k
        self.spatial = spatial

        if spatial:
            self.channel_affine = nn.Linear(x_channels, k)

        self.y_affine = nn.Linear(y_size, k, bias=False)
        self.attn_weight_affine = nn.Linear(k, 1)

    def forward(self, x, y):
        # x -> [(bs, S , dim)], len(x) = bs
        # y -> (bs, D)
        bs = y.size(0)
        y_k = self.y_affine(y) # (bs, k)
        all_spatial_attn_weights_softmax = []

        for i in range(bs):
            if self.spatial:
                x_k = self.channel_affine(x[i]) # (S, d)
                x_k += y_k[i]
                x_k = torch.tanh(x_k)
                all_spatial_attn_weights_softmax.append(F.softmax(x_k,dim=-1))

        return torch.cat(all_spatial_attn_weights_softmax, dim=0)

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        
        self.output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        self.combined_dim = self.d_l + self.d_v + self.d_a

        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=3, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=3, padding=0, bias=False)

        self.lstm = nn.LSTM(input_size=self.orig_d_l, hidden_size=self.d_l//2, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)

        self.bn_l = nn.LayerNorm(self.d_l)
        self.bn_a = nn.BatchNorm1d(self.d_a)
        self.bn_v = nn.BatchNorm1d(self.d_v)

        self.trans_l_mem = self.get_network(self_type='l_mem', layers=1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=1)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=1)

        self.cross_v = nn.ModuleList([
            CrossModalAttentionLayer(k=self.d_l, x_channels=self.d_l, y_size=self.d_l, spatial=True),
        ])

        self.cross_a = nn.ModuleList([
            CrossModalAttentionLayer(k=self.d_l, x_channels=self.d_l, y_size=self.d_l, spatial=True),
        ])

        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.output_dim)

    def get_network(self, self_type='l_mem', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        proj_x_l = x_l.permute(2, 0, 1)

        # 1. Project the textual/visual/audio features
        proj_x_l, (final_hidden_state, final_cell_state) = x_l if self.orig_d_l == self.d_l else self.lstm(proj_x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_a = self.bn_a(proj_x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = self.bn_v(proj_x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        proj_x_l = self.bn_l(proj_x_l)

        # 2. Let audio throungh a transformer
        h_vs = self.trans_v_mem(proj_x_v)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # 3. Let visual throungh a transformer
        h_as = self.trans_a_mem(proj_x_a)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]

        proj_x_l = proj_x_l.permute(1, 0, 2)
        orig_x_l = proj_x_l

        # 4. Crossmodal visual & audio -> language
        for i, _ in enumerate(self.cross_v):
            b, s, f = proj_x_l.size()
            proj_x_l = self.cross_v[i](proj_x_l, last_h_v)
            proj_x_l = proj_x_l.reshape(b, s, f)
        
        for i, _ in enumerate(self.cross_a):
            b, s, f = proj_x_l.size()
            proj_x_l = self.cross_a[i](proj_x_l, last_h_a)
            proj_x_l = proj_x_l.reshape(b, s, f)
        
        proj_x_l = proj_x_l + orig_x_l
        proj_x_l = proj_x_l.permute(1, 0, 2)
        h_ls = self.trans_l_mem(proj_x_l)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        # 5. cat features
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # 6. fusion output
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        output = self.out_layer(last_hs_proj)
        return output, last_hs