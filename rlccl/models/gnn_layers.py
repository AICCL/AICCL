"""Graph Neural Network layers for collective communication optimization."""

import torch
import torch.nn as nn


class ECDUGNNLayer(nn.Module):
    """Edge-Centric Dual-Update GNN Layer.
    
    This layer updates both node and edge features by:
    1. Updating edges based on source/destination node features
    2. Aggregating edge features to update nodes
    
    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension for transformations
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, h_nodes, h_edges, edge_src, edge_dst, num_nodes):
        """Forward pass.
        
        Args:
            h_nodes: Node features, shape (V, node_dim)
            h_edges: Edge features, shape (E, edge_dim)
            edge_src: Source node indices, shape (E,)
            edge_dst: Destination node indices, shape (E,)
            num_nodes: Number of nodes (V)
            
        Returns:
            Updated node and edge features
        """
        # Update edges based on connected nodes
        src_feats = h_nodes[edge_src]
        dst_feats = h_nodes[edge_dst]
        edge_input = torch.cat([h_edges, src_feats, dst_feats], dim=-1)
        h_edges_new = self.edge_update(edge_input)
        
        # Aggregate edge features to nodes
        in_agg = torch.zeros(num_nodes, h_edges_new.size(1), device=h_nodes.device)
        in_agg.index_add_(0, edge_dst, h_edges_new)
        out_agg = torch.zeros(num_nodes, h_edges_new.size(1), device=h_nodes.device)
        out_agg.index_add_(0, edge_src, h_edges_new)
        
        # Update nodes
        node_input = torch.cat([h_nodes, in_agg, out_agg], dim=-1)
        h_nodes_new = self.node_update(node_input)
        
        return h_nodes_new, h_edges_new
