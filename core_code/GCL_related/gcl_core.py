"""
Core Graph Contrastive Learning (GCL) implementation used in the study.

This module intentionally contains only the methodological core:
    graph augmentation -> GCN encoder -> attention pooling ->
    projection head -> NT-Xent contrastive learning -> graph embeddings

Experiment scheduling, sensitivity analysis, plotting, runtime benchmarking,
checkpoint management, and downstream clustering evaluation are excluded.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GlobalAttention, global_mean_pool
from torch_geometric.utils import negative_sampling


def seed_everything(seed: int) -> None:
    """Use the same random-seed setup as the formal GCL experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder with attention-based graph pooling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout
        self.use_attention = use_attention

        if use_attention:
            self.gate_nn = nn.Linear(out_dim, 1)
            self.att_pool = GlobalAttention(self.gate_nn)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x

        for conv in self.layers[:-1]:
            h = F.relu(conv(h, edge_index, edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.layers[-1](h, edge_index, edge_weight)

        if self.use_attention:
            return self.att_pool(h, batch)

        return global_mean_pool(h, batch)

    def forward_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return node embeddings before graph-level pooling."""
        h = x

        for conv in self.layers[:-1]:
            h = F.relu(conv(h, edge_index, edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.layers[-1](h, edge_index, edge_weight)


class ProjectionHead(nn.Module):
    """Projection head used only for the contrastive objective."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def graph_augment(
    data: Data,
    drop_edge_rate: float = 0.05,
    drop_feat_rate: float = 0.10,
    weight_sensitive: bool = True,
) -> Data:
    """
    Generate one augmented graph view.

    Edges are sampled without replacement. When edge weights are available and
    ``weight_sensitive=True``, larger-weight edges have higher retention
    probability. Node-feature entries are independently masked.
    """
    edge_index = data.edge_index.clone()
    edge_weight = (
        data.edge_weight.clone()
        if data.edge_weight is not None
        else None
    )
    x = data.x.clone()

    num_edges = edge_index.size(1)
    keep_num = int(num_edges * (1.0 - drop_edge_rate))

    if edge_weight is not None and weight_sensitive:
        probabilities = edge_weight / edge_weight.sum()
        keep_idx = torch.multinomial(
            probabilities,
            keep_num,
            replacement=False,
        )
    else:
        keep_idx = torch.randperm(
            num_edges,
            device=edge_index.device,
        )[:keep_num]

    edge_index = edge_index[:, keep_idx]

    if edge_weight is not None:
        edge_weight = edge_weight[keep_idx]

    feature_mask = (
        torch.rand_like(x) > drop_feat_rate
    ).float()

    x = x * feature_mask

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=data.num_nodes,
    )


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float = 2.0,
) -> torch.Tensor:
    """Standard NT-Xent / InfoNCE loss used in the formal GCL runs."""
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    logits = (
        torch.mm(representations, representations.t())
        / tau
    )

    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z1.device,
    )
    logits.masked_fill_(self_mask, -1e9)

    labels = torch.cat(
        [
            torch.arange(
                batch_size,
                device=z1.device,
            ) + batch_size,
            torch.arange(
                batch_size,
                device=z1.device,
            ),
        ]
    )

    return F.cross_entropy(logits, labels)


def reconstruction_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Graph reconstruction term retained from the original training function.

    In the formal GCL configuration beta=0, so this term does not contribute
    to the optimization objective. It is still evaluated here to preserve the
    random-number consumption of the original formal training implementation.
    """
    positive_loss = -torch.log(
        torch.sigmoid(
            (
                z[edge_index[0]]
                * z[edge_index[1]]
            ).sum(dim=1)
        )
        + 1e-15
    ).mean()

    negative_edge_index = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1),
    )

    negative_loss = -torch.log(
        1.0
        - torch.sigmoid(
            (
                z[negative_edge_index[0]]
                * z[negative_edge_index[1]]
            ).sum(dim=1)
        )
        + 1e-15
    ).mean()

    return positive_loss + negative_loss


def _prepare_dataset(
    graphs: Mapping[Any, Mapping[str, torch.Tensor]],
) -> Tuple[list[Data], list[Any]]:
    """Convert the trajectory-graph dictionary to PyG Data objects."""
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")

    dataset: list[Data] = []
    graph_ids: list[Any] = []

    for graph_id, graph in graphs.items():
        data = Data(
            x=graph["X"],
            edge_index=graph["edge_index"],
            edge_weight=graph.get("edge_weight"),
        )
        dataset.append(data)
        graph_ids.append(graph_id)

    return dataset, graph_ids


def train_gcl(
    graphs: Mapping[Any, Mapping[str, torch.Tensor]],
    *,
    hidden: int = 128,
    out_dim: int = 64,
    proj_dim: int = 64,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    tau: float = 2.0,
    drop_edge_rate: float = 0.05,
    drop_feat_rate: float = 0.10,
    weight_sensitive: bool = True,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> Tuple[
    np.ndarray,
    list[Any],
    GCNEncoder,
    ProjectionHead,
    list[float],
]:
    """
    Train the GCL model used for trajectory-pattern representation.

    The defaults correspond to the reference configuration in the formal
    experiments:
        hidden=128
        embedding dimension=64
        projection dimension=64
        epochs=300
        learning rate=1e-3
        temperature=2
        edge removal=0.05
        feature masking=0.10

    The formal experiments used pure instance-level contrastive learning
    (prototype_train=False, beta=0).

    Returns
    -------
    graph_embeddings:
        Final graph embeddings from the encoder (not the projection head).
    graph_ids:
        IDs in the same order as ``graph_embeddings``.
    encoder:
        Trained GCN encoder.
    projection_head:
        Trained contrastive projection head.
    loss_history:
        Total training loss for each epoch.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if not (0.0 <= drop_edge_rate < 1.0):
        raise ValueError("drop_edge_rate must be in [0, 1).")

    if not (0.0 <= drop_feat_rate < 1.0):
        raise ValueError("drop_feat_rate must be in [0, 1).")

    if tau <= 0:
        raise ValueError("tau must be positive.")

    seed_everything(seed)
    device = torch.device(device)

    dataset, graph_ids = _prepare_dataset(graphs)
    batch = Batch.from_data_list(dataset).to(device)

    encoder = GCNEncoder(
        in_dim=batch.x.size(1),
        hidden_dim=hidden,
        out_dim=out_dim,
    ).to(device)

    projection_head = ProjectionHead(
        in_dim=out_dim,
        hidden_dim=hidden,
        out_dim=proj_dim,
    ).to(device)

    # The original formal implementation initialized K=6 prototype parameters
    # even though prototype_train=False. They do not affect the loss, but the
    # initialization is retained to reproduce the original RNG sequence.
    unused_prototypes = nn.Parameter(
        torch.randn(6, proj_dim, device=device)
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(projection_head.parameters())
        + [unused_prototypes],
        lr=learning_rate,
    )

    loss_history: list[float] = []

    for _epoch in range(1, epochs + 1):
        encoder.train()
        projection_head.train()
        optimizer.zero_grad()

        view1 = Batch.from_data_list(
            [
                graph_augment(
                    data,
                    drop_edge_rate=drop_edge_rate,
                    drop_feat_rate=drop_feat_rate,
                    weight_sensitive=weight_sensitive,
                )
                for data in dataset
            ]
        ).to(device)

        view2 = Batch.from_data_list(
            [
                graph_augment(
                    data,
                    drop_edge_rate=drop_edge_rate,
                    drop_feat_rate=drop_feat_rate,
                    weight_sensitive=weight_sensitive,
                )
                for data in dataset
            ]
        ).to(device)

        graph_z1 = encoder(
            view1.x,
            view1.edge_index,
            view1.batch,
            view1.edge_weight,
        )
        graph_z2 = encoder(
            view2.x,
            view2.edge_index,
            view2.batch,
            view2.edge_weight,
        )

        projected_z1 = projection_head(graph_z1)
        projected_z2 = projection_head(graph_z2)

        contrastive_loss = nt_xent_loss(
            projected_z1,
            projected_z2,
            tau=tau,
        )

        # Formal configuration: beta = 0.
        # The reconstruction pass is nevertheless retained because the original
        # formal implementation evaluated it at every epoch, consuming random
        # numbers through negative sampling.
        node_embeddings = encoder.forward_nodes(
            batch.x,
            batch.edge_index,
            batch.edge_weight,
        )

        auxiliary_reconstruction = reconstruction_loss(
            node_embeddings,
            batch.edge_index,
            batch.num_nodes,
        )

        loss = contrastive_loss + 0.0 * auxiliary_reconstruction

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite GCL loss encountered."
            )

        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))

    # Final representation used by HDBSCAN:
    # encoder output before the projection head.
    encoder.eval()

    with torch.no_grad():
        full_batch = Batch.from_data_list(dataset).to(device)

        graph_embeddings = encoder(
            full_batch.x,
            full_batch.edge_index,
            full_batch.batch,
            full_batch.edge_weight,
        )

    return (
        graph_embeddings.detach().cpu().numpy(),
        graph_ids,
        encoder,
        projection_head,
        loss_history,
    )
