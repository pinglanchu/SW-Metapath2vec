import dgl
import numpy as np
import scipy.sparse as sp

def construct_train_test_hg(hg, etype, asetype, test_ratio):
    u, v = hg.edges(etype=etype)
    print(u.shape, v.shape)
    eids = np.arange(hg.number_of_edges(etype=etype))
    eids = np.random.permutation(eids)  # shuffle
    test_size = int(len(eids) * test_ratio)
    # train_size = hg.number_of_edges(etype=etype) - test_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense()
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), hg.number_of_edges(etype=etype))

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_pos_hg = dgl.heterograph({etype: (train_pos_u, train_pos_v), 
                                    asetype: (train_pos_v, train_pos_u)}, 
                                    num_nodes_dict = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})
    
    train_neg_hg = dgl.heterograph({etype: (train_neg_u, train_neg_v), 
                                    asetype: (train_neg_v, train_neg_u)}, 
                                    num_nodes_dict = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})

    test_pos_hg = dgl.heterograph({etype:(test_pos_u, test_pos_v),
                                asetype: (test_pos_v, test_pos_u)
                                }, num_nodes_dict = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})

    test_neg_hg = dgl.heterograph({etype:(test_neg_u, test_neg_v), 
                                asetype: (test_neg_v, test_neg_u)
                                }, num_nodes_dict = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})
    train_hg = dgl.remove_edges(hg, eids[:test_size], etype) 
    return train_hg, train_pos_hg, train_neg_hg, test_pos_hg, test_neg_hg