import networkx as nx
import dgl
from model import * # type: ignore
from utils import * # type: ignore
from train_test_construction import *
from torch.optim import SparseAdam, Adam
from torch.utils.data import DataLoader


# 定义ER随机异构网络
num_nodes = 1000  # 节点数量
p = 0.01  # 边的概率
G = nx.erdos_renyi_graph(num_nodes, p)
# G = nx.watts_strogatz_graph(num_nodes, 10, 0.3)
g = dgl.DGLGraph(G)
src, dst = g.edges()
hg = dgl.heterograph({
    ('cl', 'ch', 'hr'): (src, dst),
    ('hr', 'hc', 'cl'): (dst, src)
})

etype =  ('cl', 'ch', 'hr')# ('paper', 'pa', 'author')
asetype =  ('hr', 'hc', 'cl')# ('author', 'ap', 'paper')
metapath =  ['ch', 'hc']# ['pa', 'ap', 'pf', 'fp']
begin_node_type = 'cl'
test_ratio = 0.3

train_hg, train_pos_hg, train_neg_hg, test_pos_hg, test_neg_hg = construct_train_test_hg(
    hg, etype, asetype, test_ratio)

model = SW_MetaPath2Vec(train_hg, metapath, window_size=1)
dataloader = DataLoader(torch.arange(train_hg.num_nodes(begin_node_type)), batch_size=128,
                        shuffle=True, collate_fn=model.mp2v.sample)
optimizer = Adam(model.parameters(), lr=0.01) # SparseAdam

epochs = 50
for epoch in range(epochs):
    for (pos_u, pos_v, neg_v) in dataloader:
        embed_loss, pos_score, neg_score = model(pos_u, pos_v, neg_v, train_pos_hg, train_neg_hg, etype, asetype)
        classification_loss = compute_loss(pos_score, neg_score)

        _, test_pos_score, test_neg_score = model(pos_u, pos_v, neg_v, test_pos_hg, test_neg_hg, etype, asetype)
        test_auc = compute_metrics(test_pos_score, test_neg_score)

        loss = embed_loss + classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss: %.4f, AUC:%.4f' % (loss.item(), test_auc))