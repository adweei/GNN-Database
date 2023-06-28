import torch as th
from gcn_model import Net
from dgl.data.utils import load_graphs
from dgl.data import CoraGraphDataset
from torchmetrics.classification import BinaryConfusionMatrix
import dgl
import json

class model_test():
    confusionmatrix = list([])

    def test_model(self,model, g, features, label, mask):
        # since we're not training, we don't need to calculate the gradients for our outputs
        logits = model(g, features)
        logits = logits[mask]
        labels = label[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        bcm = BinaryConfusionMatrix().to(device = 'cuda:0')
        self.confusionmatrix = bcm(indices, labels)
        class_count = th.bincount(label)
        active_count = th.bincount(mask.type(th.int))
        self.test_value = dict()
        self.test_value['retweet_count'] = class_count[1].item()
        self.test_value['active_user'] = active_count[0].item()

        return correct.item() * 1.0 / len(labels)
    
    def model_save(self,PATH):
        #print('----------------------------------------------------------------------------------------------')                       
        self.test_value['TP'] = int(self.confusionmatrix[1][1])                   #   T   F   
        self.test_value['FP'] = int(self.confusionmatrix[0][1])                   #N  TN  FP
        self.test_value['FN'] = int(self.confusionmatrix[1][0])                   #P  FN  TP
        self.test_value['TN'] = int(self.confusionmatrix[0][0])
        try:
            self.test_value['Accuracy'] = (self.test_value['TP'] + self.test_value['TN']) / (self.test_value['TP'] +  self.test_value['FP'] + self.test_value['FN'] + self.test_value['TN'])
        except:
            self.test_value['Accuracy'] = 0
        try:
            self.test_value['Precision'] = self.test_value['TP'] / (self.test_value['TP'] + self.test_value['FP'])
        except:
            self.test_value['Precision'] = 0
        try:
            self.test_value['Recall'] = self.test_value['TP'] / (self.test_value['TP'] + self.test_value['FN'])
        except:
            self.test_value['Recall'] = 0
        try:
            self.test_value['F1'] = 2 / ((1 / self.test_value['Precision']) + (1 / self.test_value['Recall']))
        except:
            self.test_value['F1'] = 0
        try:
            self.test_value['RCR'] = (self.test_value['active_user'] + self.test_value['TP'] + self.test_value['FP']) / self.test_value['retweet_count']
        except:
            self.test_value['RCR'] = 0
        
        with open(PATH+".json", "w") as outfile:
            json.dump(self.test_value, outfile)

# #whitehouse的最大值為9764

# orgin_u = [0, 1, 1, 2, 2, 3, 3, 9763]#soruce
# orgin_v = [3, 2, 3, 1, 3, 0, 1, 2]#destion
# u, v = th.tensor(orgin_u),th.tensor(orgin_v)
# test_graph = dgl.graph((u,v))
# feature = th.rand(9764, 204,dtype=th.float32)
# test_graph.ndata['feature'] = feature
# test_graph.edata['weight'] = th.rand(8,1)
# labels = th.rand(9764)
# labels = labels < 0.3
# test_graph.ndata['label'] = labels
# test_graph = test_graph.to('cuda:0')
# feature = test_graph.ndata['feature'].to('cuda:0')
# labels = test_graph.ndata['label'].to('cuda:0')
# net = Net(204)
# net.load_state_dict(th.load('D:/GNN/code/gcn/cifar_net.pth'))
# a = model_test()
# print('acc is: {:.6f}'.format(a.test_model(net,test_graph,feature,labels)))

# dataset,label = load_graphs("D:/GNN/code/graph/whitehose_Tweet10_30.bin")
# Graph = dataset[0]
# cuda_g = Graph.to('cuda:0')
# features = cuda_g.ndata['feature']
# labels = cuda_g.ndata['label']
# net = Net(len(features[0]))
# net.load_state_dict(th.load('D:/GNN/code/gcn/cifar_net.pth'))
# a = model_test()
# print('acc is: {:.6f}'.format(a.test_model(net,cuda_g,features,labels)))
# for i in range(0,5,1):
#     print('acc is: {:.4f}'.format(a.test_model(net,cuda_g,features,labels)))