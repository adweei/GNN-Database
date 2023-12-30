import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

def send_source(edges):
    return {'m': edges.src['h'], 'w': edges.data['weight']}

def simple_reduce(nodes):
    # print('現在batch內的節點編號: ',nodes.nodes())
    # print('收到的資訊 ',nodes.mailbox['m'])
    # print('收到的資訊的weight ',nodes.mailbox['w'])
    # print(nodes.mailbox['w'] * nodes.mailbox['m'])
    #nodes.mailbox['m'] = nodes.mailbox['w'] * nodes.mailbox['m']
    # print(nodes.mailbox['m'].sum(1))
    # print('經過weight: ',nodes.mailbox['w'] * nodes.mailbox['m'])
    # print('自己的feature和收到的資訊總和: ',nodes.data['h']+(nodes.mailbox['w'] * nodes.mailbox['m']).sum(1))
    # print('------------------------------------')
    return {'h': (nodes.data['h'] * (nodes.data['self_weight'] + 1)) + ((nodes.mailbox['w'] + 1) * nodes.mailbox['m']).sum(1)}

class GCNLayer_embedding():
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, device = "cuda:0")#device:cpu cuda:0

    def forward(self, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(send_source, simple_reduce)
            h = g.ndata['h']
            return self.linear(h)
        
class GCNLayer(nn.Module): 
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, device = "cuda:0")#device:cpu cuda:0

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(send_source, simple_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self,feature):
        super(Net, self).__init__()
        self.layer0 = GCNLayer_embedding()
        self.layer1 = GCNLayer(feature, 16)
        self.layer2 = GCNLayer(16, 2)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x