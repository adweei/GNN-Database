from dgl.data import CoraGraphDataset
from dgl.data.utils import load_graphs
import torch as th
import pandas as pd
import numpy as np

class load_database():
    def load_cora_data(self):
        dataset = CoraGraphDataset()
        graph = dataset[0]
        cuda_g = graph.to('cuda:0')
        features = cuda_g.ndata['feat']
        labels = cuda_g.ndata['label']
        train_mask = cuda_g.ndata['train_mask']
        test_mask = cuda_g.ndata['test_mask']
        return cuda_g, features, labels, train_mask, test_mask

    def load_whitehose_dataset(self):
        dataset,label = load_graphs("D:/GNN/code/graph/whitehose_Tweet19_30.bin")
        Graph = dataset[0]
        cuda_g = Graph.to('cuda:0')
        features = cuda_g.ndata['feature']
        labels = cuda_g.ndata['label']
        return cuda_g, features, labels
    
    def load_selfdefinemask(self,round):
        label_floder = "D:/GNN/code/label/whitehose_Tweet19_30/"
        train_floder = label_floder+"train/train"
        test_floder = label_floder+"test/test"
        train_mask = pd.read_csv(train_floder+"_label_"+str(round)+".txt",sep="\n",header=None)
        test_mask = pd.read_csv(test_floder+"_label_"+str(round)+".txt",sep="\n",header=None)
        train_mask = th.tensor(train_mask[0].tolist())
        test_mask = th.tensor(test_mask[0].tolist())
        return train_mask,test_mask
    
    def load_elon_dataset(self,number_of_graph,round):
        elon_model_graph_data_dir = 'D:/GNN/MyResearch-main/MyResearch-main/ElonMusk/2023-02-16/base_graph_for_model/'
        dataset,label = load_graphs(elon_model_graph_data_dir+str(number_of_graph)+'/'+str(number_of_graph)+".bin")
        Graph = dataset[0]
        cuda_g = Graph.to('cuda:0')#cpu cuda:0
        features = cuda_g.ndata['feature']
        labels = cuda_g.ndata['label']
        train_mask = np.load(elon_model_graph_data_dir+str(number_of_graph)+'/train/'+str(round)+'.npy')
        test_mask = np.load(elon_model_graph_data_dir+str(number_of_graph)+'/test/'+str(round)+'.npy')
        return cuda_g, features, labels, th.from_numpy(train_mask), th.from_numpy(test_mask)