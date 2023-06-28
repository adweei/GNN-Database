import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats,device="cuda:0")#device:cpu cuda:0

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
net = Net()
print(net)

from dgl.data import CoraGraphDataset
def load_cora_data():
    dataset = CoraGraphDataset()
    graph = dataset[0]
    cuda_g = graph.to('cuda:0')
    features = cuda_g.ndata['feat']
    labels = cuda_g.ndata['label']
    train_mask = cuda_g.ndata['train_mask']
    test_mask = cuda_g.ndata['test_mask']
    return cuda_g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

import time
import numpy as np
cora_graph, features, labels, train_mask, test_mask = load_cora_data()
# Add edges between each node and itself to preserve old node representations
cora_graph.add_edges(cora_graph.nodes(), cora_graph.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []

# for epoch in range(1000):
#     if epoch >=3:
#         t0 = time.time()

#     net.train()
#     logits = net(cora_graph, features)
#     logp = F.log_softmax(logits, 1)
#     loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if epoch >=3:
#         dur.append(time.time() - t0)
    
#     acc = evaluate(net, cora_graph, features, labels, test_mask)
#     print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
#             epoch, loss.item(), acc, np.mean(dur)))
