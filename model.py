import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool

# === Base GNN Model for Graph Classification ===
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_type='sage', dropout_rate=0.5):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        if conv_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif conv_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        # Global pooling to obtain graph-level embedding
        x = global_mean_pool(x, batch)

        # Fully connected layer for classification
        x = self.fc(x)
        return x


# === GCN Model for Graph Classification ===
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        # Global pooling to obtain graph-level embedding
        x = global_mean_pool(x, batch)

        # Fully connected layer for classification
        x = self.fc(x)
        return x


class CombinedGCNGNN(nn.Module):
    def __init__(self, gcn, gnn, gcn_output_dim, hidden_channels, out_channels):
        super(CombinedGCNGNN, self).__init__()
        self.gcn = gcn
        self.gnn = gnn
        self.fc_gcn_to_gnn = nn.Linear(gcn_output_dim, gnn.in_channels)
        self.fc_graph = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # Ensure edge_index validity
        assert data.edge_index.max() < data.x.shape[0], "Invalid edge_index detected in CombinedGCNGNN."

        # Pass data through GCN
        gcn_output = self.gcn(data)

        # Map GCN output to GNN input
        data.x = self.fc_gcn_to_gnn(gcn_output)

        # Pass data through GNN
        gnn_output = self.gnn(data)

        # Return GNN output
        return gnn_output
