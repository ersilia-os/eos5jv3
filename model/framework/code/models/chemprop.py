import torch
from torch.nn import Linear, ReLU, Sequential, Dropout
from torch_geometric.nn import global_mean_pool

from data_tools.pyg_chemprop_utils import (directed_mp,
                                           aggregate_at_nodes)


class DMPNNEncoder(torch.nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3, dropout=0):
        super(DMPNNEncoder, self).__init__()
        self.act_func = ReLU()
        self.W1 = Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = Linear(hidden_size, hidden_size, bias=False)
        self.W3 = Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth

        self.mlp = Sequential(
            Dropout(p=dropout, inplace=False),
            Linear(hidden_size, hidden_size, bias=True),
            ReLU(),
            Dropout(p=dropout, inplace=False),
            Linear(hidden_size, 1, bias=True),
        )

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        x = global_mean_pool(node_attr, batch)

        return self.mlp(x), x

    def representation(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        return global_mean_pool(node_attr, batch)
