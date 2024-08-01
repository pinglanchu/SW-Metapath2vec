import torch
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import MetaPath2Vec


class HeteroDotPredictor(torch.nn.Module):
    def forward(self, hg, h, etype):
        with hg.local_scope():
            hg.ndata['h'] = h
            hg.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return hg.edges[etype].data['score']
        

class Structure_Weighted_Embedding(torch.nn.Module):
    def __init__(self, feat_in, feat_hidden):
        super().__init__()
        self.SW = torch.nn.Sequential(
            torch.nn.Linear(feat_in, feat_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(feat_hidden, 128)
        )

    def forward(self, hg, etype, asetype, feats):
        etype_hg = dgl.edge_type_subgraph(hg, [etype, asetype])
        etype_g = dgl.to_homogeneous(etype_hg)
        adj = etype_g.adjacency_matrix().to_dense()
        part1 = adj @ adj @ adj * adj
        part2_ = torch.diag(adj @ adj)
        part2 = part2_.reshape(-1, 1) @ part2_.reshape(1, -1)
        Gamma = part1 / part2 
        Gamma = torch.nan_to_num(Gamma)
        Gamma = F.normalize(Gamma, p=2, dim=1)
        h = torch.cat([feats[etype[0]], feats[etype[2]]], dim=0)
        sw_h = self.SW(Gamma)
        h = torch.cat([h, sw_h], dim=1)
        return h


class SW_MetaPath2Vec(torch.nn.Module):
    def __init__(self, hg, metapath, window_size):
        super().__init__()
        self.mp2v = MetaPath2Vec(hg, metapath, window_size, sparse=False)
        self.sw = Structure_Weighted_Embedding(2000, 256)
        self.pred = HeteroDotPredictor()
        self.hg = hg
    def forward(self, pos_u, pos_v, neg_v, pos_hg, neg_hg, etype, asetype):
        loss = self.mp2v(pos_u, pos_v, neg_v)
        hs = {ntype: self.mp2v.node_embed(torch.LongTensor(self.mp2v.local_to_global_nid[ntype])) for ntype in self.hg.ntypes}
        h = self.sw(self.hg, etype, asetype, hs)
        hs[etype[0]] = h[:self.hg.num_nodes(etype[0]), :]
        hs[etype[2]] = h[self.hg.num_nodes(etype[0]): , :]

        return loss, self.pred(pos_hg, hs, etype), self.pred(neg_hg, hs, etype)
       