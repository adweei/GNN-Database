import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl 
from gcn_model import Net 
from load_database import load_database 
from evaluate import model_evaluate

orgin_u = [0, 1, 1, 2, 2, 3, 3, 3]#soruce
orgin_v = [3, 2, 3, 1, 3, 0, 1, 2]#destion
u, v = th.tensor(orgin_u),th.tensor(orgin_v)
test_graph = dgl.graph((u,v))
feature = th.tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3]], dtype=th.float32)
test_graph.ndata['feature'] = feature
test_graph.edata['weight'] = th.tensor([[3], [2], [3], [2], [3], [3], [3], [3]])
train_mask = th.tensor([True, True, False, True])
train_mask = th.tensor([False, False, True, False])
label = th.tensor([ 0, 1, 1, 0])
test_graph.ndata['label'] = label
test_graph = test_graph.to('cuda:0')
label = test_graph.ndata['label']
net = Net(3)
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(1):
    if epoch >=3:
        t0 = time.time()

net.train()
logits = net(test_graph, test_graph.ndata['feature'])
logp = F.log_softmax(logits, 1)
loss = F.nll_loss(logp[train_mask], label[train_mask])
            
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")