"""Slot-level policy network for collective communication scheduling."""

import torch
import torch.nn as nn
import numpy as np
import math

from .gnn_layers import ECDUGNNLayer


class SlotLevelPolicy(nn.Module):
    """Slot-level Policy Network for collective communication scheduling.
    
    This network encodes the topology state and outputs:
    1. Actor logits for selecting (chunk, edge) pairs
    2. Critic value estimation for PPO training
    
    Args:
        node_feat_dim: Dimension of node features (default: 5)
        edge_feat_dim: Dimension of edge features (default: 2)
        cand_feat_dim: Dimension of candidate dynamic features (default: 11)
        chunk_feat_dim: Dimension of chunk features (default: 3)
        hidden_dim: Hidden dimension for all layers (default: 64)
    """
    
    def __init__(
        self, 
        node_feat_dim: int = 5,
        edge_feat_dim: int = 2,
        cand_feat_dim: int = 11,
        chunk_feat_dim: int = 3,
        hidden_dim: int = 64,
        use_stop: bool = False,
        target_reservations: bool = False,
        candidate_coverage: bool = False,
        relational_features: bool = False,
        actor_prior_scale: float = 0.0,
        arrival_aware: bool = False,
        shared_redesign: bool = False,
    ):
        super().__init__()
        if isinstance(actor_prior_scale, bool) or not math.isfinite(actor_prior_scale) or actor_prior_scale < 0:
            raise ValueError('Actor prior scale must be finite and nonnegative')
        if actor_prior_scale and not use_stop:
            raise ValueError('Actor prior requires the stored STOP probability schema')
        self.shared_redesign = bool(shared_redesign)
        self.hidden_dim = hidden_dim
        self.use_stop = bool(use_stop)
        self.target_reservations = bool(target_reservations)
        self.candidate_coverage = bool(candidate_coverage)
        self.relational_features = bool(relational_features)
        self.arrival_aware = bool(arrival_aware)
        if self.arrival_aware and (not self.use_stop or not self.relational_features or cand_feat_dim != 20):
            raise ValueError('Arrival actor requires STOP, relational features and the 20-feature schema')
        if self.relational_features and not self.use_stop:
            raise ValueError('Relational features require the stored STOP rollout schema')
        
        # Encoders
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # GNN layers
        self.layer1 = ECDUGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        self.layer2 = ECDUGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Actor: scores candidate (c, e) pairs
        self.actor = nn.Sequential(
            nn.Linear((9 if self.relational_features else 5) * hidden_dim + cand_feat_dim + (20 if self.shared_redesign else 0), hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Critic: V(s_t) at slot level
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )
        if self.use_stop:
            self.stop_actor = nn.Sequential(
                nn.Linear(hidden_dim + (11 if self.arrival_aware else 8), hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.constant_(self.stop_actor[-1].bias, -4.0)
        if self.relational_features:
            self.pending_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim + 2, hidden_dim),
                nn.LeakyReLU(0.1), nn.Linear(hidden_dim, hidden_dim))
        if self.shared_redesign:
            if actor_prior_scale != 0 or not self.relational_features:
                raise ValueError('Shared redesign requires random policy without a fixed prior')
            self.feature_skip = nn.Linear(cand_feat_dim + 20, 1, bias=False)
            nn.init.normal_(self.feature_skip.weight, std=0.02)
        if self.use_stop:
            self.register_buffer('actor_prior_scale', torch.tensor(float(actor_prior_scale)))
            if actor_prior_scale:
                # A zero neural residual starts from the declared strong prior.
                # The actor/GNN remain trainable under the same joint-slot PPO.
                nn.init.zeros_(self.actor[-1].weight)
                nn.init.zeros_(self.actor[-1].bias)
                nn.init.zeros_(self.stop_actor[-1].weight)
                nn.init.constant_(self.stop_actor[-1].bias, -4.0)

    @staticmethod
    def candidate_prior(dynamic_feats):
        if dynamic_feats.ndim != 2 or dynamic_feats.shape[1] < 11:
            raise ValueError('Actor prior needs the bound candidate feature schema')
        base = (8.0 * dynamic_feats[:, 5] + 4.0 * dynamic_feats[:, 0]
                + 2.0 * dynamic_feats[:, 4] + dynamic_feats[:, 10])
        if dynamic_feats.shape[1] == 20:
            # Cost/distance stay actor inputs; fixed tie biases regressed in the bound v8 ablation.
            base = base - 8.0 * dynamic_feats[:, 15]
        return base

    def get_stop_logit(self, g_ctx, stop_feats, candidate_count, candidate_dynamic_feats=None):
        if not self.use_stop or candidate_count < 1:
            raise ValueError('Learned STOP requires a nonempty candidate pool and a STOP-enabled policy')
        # A candidate-count offset keeps initial STOP probability comparable
        # across pool sizes. Recompute uses the exact saved pool size.
        offset = g_ctx.new_tensor(float(candidate_count)).log()
        if self.arrival_aware:
            if candidate_dynamic_feats is None or candidate_dynamic_feats.shape != (candidate_count, 20):
                raise ValueError('Arrival STOP needs the exact stored 20-feature candidate pool')
            waiting = candidate_dynamic_feats[:, 15]
            summaries = torch.stack((waiting.min(), waiting.mean(), candidate_dynamic_feats[:, 16].min()))
            stop_feats = torch.cat((stop_feats, summaries))
        logit = self.stop_actor(torch.cat((g_ctx, stop_feats))).reshape(()) + offset
        if candidate_dynamic_feats is None:
            if float(self.actor_prior_scale) != 0.0:
                raise ValueError('Prior STOP needs the exact stored candidate pool')
            return logit
        if len(candidate_dynamic_feats) != candidate_count:
            raise ValueError('Prior STOP candidate count differs from its pool')
        prior = self.actor_prior_scale * self.candidate_prior(candidate_dynamic_feats)
        if self.arrival_aware:
            # A soft group WAIT prior only when every currently available
            # proposal moves a singleton partial while its partner is in flight.
            # Mixed pools retain the original STOP probability. Neural residuals
            # can override either choice; the stored pool defines exact PPO replay.
            all_waiting = candidate_dynamic_feats[:, 15].min()
            prior = prior + self.actor_prior_scale * 8.0 * all_waiting
        # Preserve the initial group STOP probability across pool sizes/prior
        # offsets. Recompute this from each exact saved micro-action pool.
        return logit + torch.logsumexp(prior, dim=0) - offset

    def encode_state(self, node_feats, edge_feats, edge_src, edge_dst, chunk_feats):
        """Encode state and return embeddings for actor/critic.
        
        Args:
            node_feats: Node features, shape (V, node_feat_dim)
            edge_feats: Edge features, shape (E, edge_feat_dim)
            edge_src: Source node indices, shape (E,)
            edge_dst: Destination node indices, shape (E,)
            chunk_feats: Chunk features, shape (C, chunk_feat_dim)
            
        Returns:
            h_v: Node embeddings, shape (V, hidden_dim)
            h_e: Edge embeddings, shape (E, hidden_dim)
            h_c: Chunk embeddings, shape (C, hidden_dim)
            g_ctx: Global context, shape (hidden_dim,)
        """
        num_nodes = node_feats.size(0)
        h_v = self.node_encoder(node_feats)
        h_e = self.edge_encoder(edge_feats)
        
        h_v, h_e = self.layer1(h_v, h_e, edge_src, edge_dst, num_nodes)
        h_v, h_e = self.layer2(h_v, h_e, edge_src, edge_dst, num_nodes)
        
        g_ctx = self.global_pool(h_v.mean(dim=0))
        h_c = self.chunk_encoder(chunk_feats)
        
        return h_v, h_e, h_c, g_ctx

    def get_value(self, g_ctx):
        """Critic: V(s_t) for slot-level value estimation.
        
        Args:
            g_ctx: Global context, shape (hidden_dim,)
            
        Returns:
            Value estimate, shape (1,)
        """
        # The optional training mode gives the policy encoder only policy gradients.
        context = g_ctx if getattr(self, 'critic_encoder_grad', True) else g_ctx.detach()
        return self.critic(context)

    def get_candidate_logits(
        self, 
        h_v, 
        h_e, 
        h_c, 
        g_ctx, 
        edge_src, 
        edge_dst, 
        cand_e, 
        cand_c, 
        cand_dynamic_feats,
        relations=None,
    ):
        """Actor: compute logits for candidate (c, e) pairs.
        
        Args:
            h_v: Node embeddings, shape (V, hidden_dim)
            h_e: Edge embeddings, shape (E, hidden_dim)
            h_c: Chunk embeddings, shape (C, hidden_dim)
            g_ctx: Global context, shape (hidden_dim,)
            edge_src: Source node indices, shape (E,)
            edge_dst: Destination node indices, shape (E,)
            cand_e: Candidate edge indices, shape (N_cand,)
            cand_c: Candidate chunk indices, shape (N_cand,)
            cand_dynamic_feats: Dynamic features for candidates, shape (N_cand, cand_feat_dim)
            
        Returns:
            Logits for candidates, shape (N_cand,)
        """
        cand_edge_emb = h_e[cand_e]
        cand_src_emb = h_v[edge_src[cand_e]]
        cand_dst_emb = h_v[edge_dst[cand_e]]
        cand_chunk_emb = h_c[cand_c]
        
        # Expand global context to match number of candidates
        g_ctx_expanded = g_ctx.unsqueeze(0).expand(len(cand_e), -1)
        
        pieces = [
            cand_edge_emb, cand_src_emb, cand_dst_emb, 
            cand_chunk_emb, cand_dynamic_feats, g_ctx_expanded
        ]
        if self.relational_features:
            if relations is None:
                raise ValueError('Relational actor needs exact resident, target and pending features')
            for key in ('source', 'missing', 'targets'):
                mask = torch.as_tensor(relations[key], dtype=h_v.dtype, device=h_v.device)
                pieces.append((mask @ h_v) / mask.sum(dim=1, keepdim=True).clamp_min(1))
            pending = relations['pending']
            pooled = h_v.new_zeros((relations['chunks_count'], self.hidden_dim))
            if pending:
                masks = torch.as_tensor(np.asarray([p['contributions'] for p in pending]),
                                        dtype=h_v.dtype, device=h_v.device)
                contribution_context = masks @ h_v / masks.sum(dim=1, keepdim=True).clamp_min(1)
                dst = torch.as_tensor([p['dst'] for p in pending], dtype=torch.long, device=h_v.device)
                # Wait is encoded in reference units; partial flag preserves phase.
                timing = torch.as_tensor([[p['wait_reference'], float(p['partial'])] for p in pending],
                                         dtype=h_v.dtype, device=h_v.device)
                embeddings = self.pending_encoder(torch.cat((h_v[dst], contribution_context, timing), dim=1))
                pending_c = torch.as_tensor([p['chunk'] for p in pending], dtype=torch.long, device=h_v.device)
                pooled.index_add_(0, pending_c, embeddings)
                counts = h_v.new_zeros((relations['chunks_count'], 1))
                counts.index_add_(0, pending_c, h_v.new_ones((len(pending), 1)))
                pooled = pooled / counts.clamp_min(1)
            pieces.append(pooled[cand_c])
        if self.shared_redesign:
            route = torch.as_tensor(relations['route'], dtype=h_v.dtype, device=h_v.device)
            task = torch.as_tensor(relations['task'], dtype=h_v.dtype, device=h_v.device)
            pieces.extend([route, task])
        actor_input = torch.cat(pieces, dim=-1)
        
        residual = self.actor(actor_input).squeeze(-1)
        if self.shared_redesign:
            residual = residual + self.feature_skip(torch.cat((cand_dynamic_feats, route, task), -1)).squeeze(-1)
        if self.use_stop:
            return residual + self.actor_prior_scale * self.candidate_prior(cand_dynamic_feats)
        return residual
