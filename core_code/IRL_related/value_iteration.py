"""Soft value iteration used by the formal Deep MaxEnt IRL experiments.

The road network is represented as a directed graph: nodes are states and
outgoing directed edges are actions. The implementation below is the core
value-iteration routine used by the parameter/feature robustness experiments.
"""

import numpy as np
from scipy.special import logsumexp


def classic_value_iteration_action(
    rewards,
    gamma=0.85,
    nodes=None,
    adj=None,
    edge_to_idx=None,
    node_to_idx=None,
    tolerance=1e-3,
):
    """Compute soft state values and directed-edge action values.

    Bellman update:
        Q(s, a) = R(s, a) + gamma * V(s')
        V(s) = logsumexp_a Q(s, a)

    This function preserves the implementation used in the formal robustness
    experiments, including the 1000-iteration safety cap and zero continuation
    value for nodes without valid outgoing edges.
    """
    num_nodes = len(nodes)
    V = np.zeros(num_nodes)

    # Precompute outgoing edge and destination indices for each node.
    graph_structure = [[] for _ in range(num_nodes)]
    for u_idx, u in enumerate(nodes):
        outgoing_edges = adj.get(u, [])
        for s, d in outgoing_edges:
            edge_idx = edge_to_idx.get((s, d))
            d_idx = node_to_idx.get(d)
            if edge_idx is not None and d_idx is not None:
                graph_structure[u_idx].append((edge_idx, d_idx))

    iter_count = 0
    max_iter = 1000

    while iter_count < max_iter:
        V_old = V.copy()
        delta = 0

        for u_idx in range(num_nodes):
            transitions = graph_structure[u_idx]
            if not transitions:
                continue

            q_values = [
                rewards[edge_idx] + gamma * V_old[d_idx]
                for edge_idx, d_idx in transitions
            ]

            if q_values:
                V[u_idx] = logsumexp(q_values)
            else:
                V[u_idx] = -1e10

            diff = abs(V[u_idx] - V_old[u_idx])
            if diff > delta:
                delta = diff

        iter_count += 1
        if delta < tolerance:
            break

    # Compute final directed-edge action values once using the converged V.
    Q = {}
    for u in nodes:
        for s, d in adj.get(u, []):
            edge_idx = edge_to_idx.get((s, d))
            d_idx = node_to_idx.get(d)

            if edge_idx is not None and d_idx is not None:
                Q[(s, d)] = rewards[edge_idx] + gamma * V[d_idx]

    return V, Q
