import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.ModuleList()
        for _ in range(num_heads):
            self.fc.append(nn.Linear(in_dim, out_dim))
        self.fc = nn.Linear(embedding_dim * num_heads, 1)

    def forward(self, g):
        x = g.ndata['feat']  # 使用節點的特徵向量
        for attn in self.gat_layer:
            x = attn(g, x)  # 使用GAT層進行多頭注意力
        x = torch.sigmoid(self.fc(x))
        return x

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.ModuleList()
        for _ in range(num_heads):
            self.fc.append(nn.Linear(in_dim, out_dim))
        self.attn = nn.ModuleList([
            nn.Linear(out_dim * 2, 1)
            for _ in range(num_heads)
        ])

    def edge_attention(self, edges):
        src, dst = edges.src['h'], edges.dst['h']
        z = torch.cat([src, dst], dim=1)
        return {'e': sum(attn(z) for attn in self.attn)}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = nodes.mailbox['h']
        return {'h': torch.sum(alpha.unsqueeze(-1) * h, dim=1)}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return torch.cat([fc(g.ndata['h']) for fc in self.fc], dim=1)

# 創建虛擬圖數據（節點和邊緣）
embedding_dim = 64
num_nodes = 100
num_edges = 500
g = dgl.DGLGraph()
g.add_nodes(num_nodes)
g.ndata['feat'] = torch.randn(num_nodes, embedding_dim)  # 添加特徵向量，這裡假設特徵數為1
edges = [(np.random.randint(0, num_nodes), np.random.randint(0, num_nodes)) for _ in range(num_edges)]
src, dst = zip(*edges)
g.add_edges(src, dst)

# 創建訓練數據，包括正例和負例
num_negative_samples = 5
positive_samples = [(node, 1.0) for node in range(num_nodes)]
negative_samples = [(node, 0.0) for node in range(num_nodes) for _ in range(num_negative_samples)]
training_data = positive_samples + negative_samples

# 初始化模型
embedding_dim = 64
num_heads = 2  # 設置注意力頭的數量，可以根據需要調整
model = GAT(num_nodes, embedding_dim, num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()  # 二元交叉熵損失

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for node, label in training_data:
        optimizer.zero_grad()
        output = model(g)
        loss = criterion(output[node], torch.tensor([label], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")

# 現在，模型已經訓練完成，可以使用它進行預測，以確定每個節點的類別
