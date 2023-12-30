from dgl.data import CoraGraphDataset
from dgl.data.utils import load_graphs
import torch as th
import pandas as pd
import numpy as np

class load_database():    
    def load_elon_dataset(self, number_of_graph, round):
        # print('weight graph')
        elon_model_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'
        dataset, label = load_graphs(elon_model_graph_data_dir + str(number_of_graph) + '/' + str(number_of_graph) + ".bin")
        Graph = dataset[0]
        cuda_g = Graph.to('cuda:0')         #cpu cuda:0
        features = cuda_g.ndata['feature']
        labels = cuda_g.ndata['label']
        train_mask = np.load(elon_model_graph_data_dir + str(number_of_graph) + '/train/' + str(round) + '.npy')
        test_mask = np.load(elon_model_graph_data_dir + str(number_of_graph) + '/test/' + str(round) + '.npy')
        # print("edge number: ",cuda_g.number_of_edges())
        # print("差值: ",cuda_g.number_of_edges() - cuda_g.number_of_nodes())
        return cuda_g, features, labels, th.from_numpy(train_mask), th.from_numpy(test_mask)
    
    def load_elon_no_weight_dataset(self,number_of_graph,round):
        # print('no weight graph')
        elon_model_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/base_graph_for_model/'
        dataset, label = load_graphs(elon_model_graph_data_dir + str(number_of_graph) + '/' + str(number_of_graph) + "noweight.bin")
        # print('dir: ', dataset)
        Graph = dataset[0]
        cuda_g = Graph.to('cuda:0')         #cpu cuda:0
        features = cuda_g.ndata['feature']
        labels = cuda_g.ndata['label']
        train_mask = np.load(elon_model_graph_data_dir + str(number_of_graph) + '/train/' + str(round)+'.npy')
        test_mask = np.load(elon_model_graph_data_dir + str(number_of_graph) + '/test/' + str(round)+'.npy')
        return cuda_g, features, labels, th.from_numpy(train_mask), th.from_numpy(test_mask)