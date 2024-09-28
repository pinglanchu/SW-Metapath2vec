import networkx as nx
import numpy as np
from model import * # type: ignore
from utils import * # type: ignore
from torch.optim import SparseAdam, Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import dgl.nn as dglnn


def construct_train_test_hg(hg, etype, asetype, test_ratio):
    u, v = hg.edges(etype=etype)
    print(u.shape, v.shape)
    eids = np.arange(hg.number_of_edges(etype=etype))
    eids = np.random.permutation(eids)  # shuffle
    test_size = int(len(eids) * test_ratio)
    test_u, test_v = u[eids[:test_size]], v[eids[:test_size]]
    test_hg = dgl.heterograph({etype: (test_u, test_v),
                                   asetype: (test_v, test_u)
                                   }, num_nodes_dict={ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})
    train_hg = dgl.remove_edges(hg, eids[:test_size], etype)
    return train_hg, test_hg


class HeteroDotPredictor(torch.nn.Module):
    def forward(self, hg, h, etype):
        with hg.local_scope():
            hg.ndata['h'] = h
            hg.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return hg.edges[etype].data['score']


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class R_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.layer1 = dglnn.HeteroGraphConv({
            # ('paper', 'pv', 'venue'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            # ('venue', 'vp', 'paper'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            # ('author', 'ap', 'paper'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            # ('paper', 'pa', 'author'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            # ('paper', 'pt', 'term'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            # ('term', 'tp', 'paper'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('user', 'ua', 'artist'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('artist', 'au', 'user'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('user', 'uu0', 'user'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('user', 'uu1', 'user'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('artist', 'at', 'tag'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
            ('tag', 'ta', 'artist'): dglnn.GATConv(in_feats, hidden_feats, num_heads),
        })
        self.layer2 = dglnn.HeteroGraphConv({
            # ('paper', 'pv', 'venue'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            # ('venue', 'vp', 'paper'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            # ('author', 'ap', 'paper'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            # ('paper', 'pa', 'author'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            # ('paper', 'pt', 'term'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            # ('term', 'tp', 'paper'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('user', 'ua', 'artist'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('artist', 'au', 'user'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('user', 'uu0', 'user'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('user', 'uu1', 'user'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('artist', 'at', 'tag'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),
            ('tag', 'ta', 'artist'): dglnn.GATConv(hidden_feats * num_heads, out_feats, 1),

        })
        self.pred = HeteroDotPredictor()

    def forward(self, graph, neg_graph, inputs, etype):
        h = self.layer1(graph, inputs)
        # print('h', h['paper'].shape)
        h = {k: v.reshape(-1, self.hidden_feats*self.num_heads) for k, v in h.items()}  # 拼接头的输出
        h = self.layer2(graph, h)
        h = {k: torch.mean(v, dim=1) for k, v in h.items()}  # 取平均
        return  self.pred(graph, h, etype), self.pred(neg_graph, h, etype)


class HANLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(HANLayer, self).__init__()
        self.attn = dglnn.HeteroGraphConv({
            # ('paper', 'pv', 'venue'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            # ('venue', 'vp', 'paper'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            # ('author', 'ap', 'paper'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            # ('paper', 'pa', 'author'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            # ('paper', 'pt', 'term'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            # ('term', 'tp', 'paper'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('user', 'ua', 'artist'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('artist', 'au', 'user'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('user', 'uu0', 'user'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('user', 'uu1', 'user'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('artist', 'at', 'tag'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
            ('tag', 'ta', 'artist'): dglnn.GATConv(in_feats, out_feats, num_heads=4),
        })
    def forward(self, graph, inputs):
        h = self.attn(graph, inputs)
        h = {k: torch.mean(v, dim=1) for k, v in h.items()}  # 取平均
        return h

class HAN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HAN, self).__init__()
        self.layer1 = HANLayer(in_feats, hidden_feats)
        self.layer2 = HANLayer(hidden_feats, out_feats)
        self.pred = HeteroDotPredictor()
    def forward(self, graph, neg_graph, inputs, etype):
        h = self.layer1(graph, inputs)
        h = self.layer2(graph, h)
        return self.pred(graph, h, etype), self.pred(neg_graph, h, etype)


# dblp_data = np.loadtxt('./data/DBLP_link.dat')
# paper_venue = dblp_data[dblp_data[:, 2]==2][:, 0:3]
# venue_paper = dblp_data[dblp_data[:, 2]==5][:, 0:3]
# author_paper = dblp_data[dblp_data[:, 2]==0][:, 0:3]
# paper_author = dblp_data[dblp_data[:, 2]==3][:, 0:3]
# paper_term = dblp_data[dblp_data[:, 2]==1][:, 0:3]
# term_paper = dblp_data[dblp_data[:, 2]==4][:, 0:3]
# hg = dgl.heterograph({
#     ('paper', 'pv', 'venue'): (paper_venue[:, 0], paper_venue[:, 1]),
#     ('venue', 'vp', 'paper'): (venue_paper[:, 0], venue_paper[:, 1]),
#     ('author', 'ap', 'paper'): (author_paper[:, 0], author_paper[:, 1]),
#     ('paper', 'pa', 'author'): (paper_author[:, 0], paper_author[:, 1]),
#     ('paper', 'pt', 'term'): (paper_term[:, 0], paper_term[:, 1]),
#     ('term', 'tp', 'paper'): (term_paper[:, 0], term_paper[:, 1]),
# })
# etype =  ('paper', 'pt', 'term')# ('paper', 'pa', 'author')
# asetype =  ('term', 'tp', 'paper')# ('author', 'ap', 'paper')
# metapath =  ['pt', 'tp']# ['pa', 'ap', 'pf', 'fp']
# paper_feats = nn.Embedding(hg.num_nodes('paper'), 32).weight
# venue_feats = nn.Embedding(hg.num_nodes('venue'), 32).weight
# author_feats = nn.Embedding(hg.num_nodes('author'), 32).weight
# term_feats = nn.Embedding(hg.num_nodes('term'), 32).weight
# node_features = {'paper': paper_feats, 'venue': venue_feats, 'author': author_feats, 'term': term_feats}


lastfm = np.loadtxt('./data/LastFM_link.dat')
user_artist = lastfm[lastfm[:, 2]==0][:, 0:3]
user_user = lastfm[lastfm[:, 2]==1][:, 0:3]
artist_tag = lastfm[lastfm[:, 2]==2][:, 0:3]

hg = dgl.heterograph({
    ('user', 'ua', 'artist'): (user_artist[:, 0], user_artist[:, 1]),
    ('artist', 'au', 'user'): (user_artist[:, 1], user_artist[:, 0]),
    ('user', 'uu0', 'user'): (user_user[:, 0], user_user[:, 1]),
    ('user', 'uu1', 'user'): (user_user[:, 1], user_user[:, 0]),
    ('artist', 'at', 'tag'): (artist_tag[:, 0], artist_tag[:, 1]),
    ('tag', 'ta', 'artist'): (artist_tag[:, 1], artist_tag[:, 0]),
})
print(hg)
etype = ('artist', 'at', 'tag')
asetype = ('tag', 'ta', 'artist')
user_feats = nn.Embedding(hg.num_nodes('user'), 32).weight
artist_feats = nn.Embedding(hg.num_nodes('artist'), 32).weight
tag_feats = nn.Embedding(hg.num_nodes('tag'), 32).weight
node_features = {'user': user_feats, 'artist': artist_feats, 'tag': tag_feats}

test_ratio = 0.6
train_hg, test_hg = construct_train_test_hg(hg, etype, asetype, test_ratio)


model = R_GCN(32, 64, 32, hg.etypes)
# model = GAT(32, 64, 1, 4)
# model = HAN(32, 32, 1)
k = 5
opt = torch.optim.Adam(model.parameters())

auc = 0
precision = 0
for repeat in range(3):
    for epoch in range(100):
        negative_graph = construct_negative_graph(train_hg, k, etype)
        # pos_score, neg_score = model(train_hg, negative_graph, node_features, etype)
        pos_score, neg_score = model(train_hg, negative_graph, node_features, etype)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print('loss:', loss.item())
    with torch.no_grad():
        test_negative_graph = construct_negative_graph(test_hg, k, etype)
        test_pos_score, test_neg_score = model(test_hg, test_negative_graph, node_features, etype)
        test_auc, test_precision = compute_metrics(test_pos_score, test_neg_score)
        auc += test_auc
        precision += test_precision
print('mean auc:%4f, mean precision:%.4f' % (auc/3, precision/3))
        # print('loss:%.4f, AUC:%.4f' % (loss.item(), test_auc))