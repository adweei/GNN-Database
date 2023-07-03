import dgl
import torch
import torch.nn.functional as F
import torch
from copy import deepcopy
from tqdm import tqdm
import time
import numpy as np
from eval import *
from gat import *
from functions import *

# ==============================================================================
device = '' # computing device -> gpu or cpu
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# loading graph list from .bin file, will return a list which storing dgl.graph
graph_list = Load_GraphList(file_name='graph_list.bin')

# metric evaluation object, the code is in eval.py
metrics = evaluation()

"""
dgl.graph storing node & edge features:
    
    node features stores in .ndata => shape is |V|*k where |V| is number of nodes, k is number of features
    edge features stores in .edata => shape is |E|*p where |E| is number of edges, p is number of features

both ndata & edata are dict(), so you have to assign a key to .ndata or .edata while you storing

ex.
    g  = dgl.graph
    g.ndata['feature'] = node_feature
    g.edata['weight'] = edge_weight
"""
for g in graph_list:
    hashtag_matrix = g.ndata['hashtags']
    user_feature = F.normalize(g.ndata['feature'], dim= 1)
    label = g.ndata['label'].to(torch.long)
    feature = torch.cat((hashtag_matrix, user_feature), dim= 1).to(device= device)

    # GAT Model
    net = GAT(
        g= g.to(device= device), 
        in_dim= feature.size()[1], 
        hidden_dim= 8, # hidden layer dimension
        out_dim= 2, # output dimension, equal to number of class
        num_heads= 3 # multi-head attention
    )

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr= 0.001)

    # train_test mask split
    train_mask, test_mask = train_test_split_easy(labels= label)

    # training for this graph
    dur = []
    for epoch in range(30):
        if epoch > 0:
            t0 = time.time()
        
        logits = net(feature.float().to(device))
        logp = F.log_softmax(logits, 1) # shape = |V| * classes
        loss = F.nll_loss(logp[train_mask].to(device), label[train_mask].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # evaluate the accuracy and precision of testing set
        acc, precision = metrics.Binary_class(logits= logp, labels= label, mask= test_mask)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))

        print("Acc {:.2f} | Prec {:.2f}".format(acc, precision))
        print("=" * 80)
