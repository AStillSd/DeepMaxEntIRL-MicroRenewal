"""Core Deep Maximum-Entropy IRL training used in the formal experiments.

This file is extracted from the training path used by the parameter- and
feature-robustness experiments. It intentionally excludes grid-search loops,
benchmark orchestration, result-table generation, plotting, and case-specific
data preprocessing.

Training procedure
------------------
1. Map directed-edge features to rewards with a neural reward network.
2. Solve a soft policy using log-sum-exp value iteration.
3. Preserve the empirical joint distribution of expert trajectory origins and
   trajectory lengths.
4. Propagate expected directed-edge visitation deterministically (no Monte
   Carlo rollout during training).
5. Update the reward network using the MaxEnt occupancy-difference gradient.

Visitation MSE/RMSE/max error are monitoring metrics; the actual optimization
objective is the occupancy-gradient surrogate.
"""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from value_iteration import classic_value_iteration_action

Edge = Tuple[str, str]


class RewardNet(nn.Module):
    """Neural reward function used in the formal Deep MaxEnt IRL experiments."""

    def __init__(
        self,
        input_size,
        hidden_sizes,
        hidden_activation=nn.ELU,
        output_activation=nn.Tanh,
        output_scale=5.0,
    ):
        super().__init__()
        self.output_scale = output_scale

        self.config = {
            "input_size": input_size,
            "hidden_sizes": list(hidden_sizes),
            "output_scale": output_scale,
        }

        layers = []
        prev = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(hidden_activation())
            prev = h

        layers.append(nn.Linear(prev, 1))

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.net(x) * self.output_scale


def seed_everything(seed: int) -> None:
    """Set the same random seeds/deterministic flags used in the experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def architecture_hidden_sizes(architecture: str, d: int) -> List[int]:
    """Translate manuscript architecture notation into hidden-layer widths."""
    architectures = {
        "shallow": [2 * d],                       # [d, 2d, 1]
        "reference": [2 * d, d],                 # [d, 2d, d, 1]
        "deep": [2 * d, 2 * d, d],               # [d, 2d, 2d, d, 1]
        "deep_wide": [2 * d, 4 * d, 2 * d, d],   # [d, 2d, 4d, 2d, d, 1]
    }
    if architecture not in architectures:
        raise ValueError(
            f"Unknown architecture={architecture!r}; choose {sorted(architectures)}"
        )
    return architectures[architecture]


def _normalize_env(env: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = [str(node) for node in env["nodes"]]
    node_to_idx = {
        str(node): int(idx) for node, idx in env["node_to_idx"].items()
    }
    edges = [(str(u), str(v)) for u, v in env["edges"]]
    edge_to_idx = {
        (str(u), str(v)): int(idx)
        for (u, v), idx in env["edge_to_idx"].items()
    }
    adj = {
        str(node): [(str(u), str(v)) for u, v in outgoing]
        for node, outgoing in env["adj"].items()
    }
    return {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "edges": edges,
        "edge_to_idx": edge_to_idx,
        "adj": adj,
        "features": env["features"],
    }


def normalize_edge_trajectories(
    trajectories: Sequence[Sequence[Sequence[Any]]],
) -> List[List[Edge]]:
    """Normalize edge IDs to strings and reject malformed edges."""
    normalized: List[List[Edge]] = []
    for traj_idx, trajectory in enumerate(trajectories):
        current: List[Edge] = []
        for edge_idx, edge in enumerate(trajectory):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError(
                    f"Malformed edge at trajectory {traj_idx}, "
                    f"position {edge_idx}: {edge!r}"
                )
            current.append((str(edge[0]), str(edge[1])))
        if current:
            normalized.append(current)

    if not normalized:
        raise ValueError("No non-empty edge trajectories were supplied.")

    return normalized


def validate_edge_trajectories(
    edge_trajectories: Sequence[Sequence[Edge]],
    edge_to_idx: Mapping[Edge, int],
    max_examples: int = 10,
) -> None:
    """Fail early if prepared trajectories contain edges outside the subgraph."""
    valid_edges = {(str(u), str(v)) for u, v in edge_to_idx}
    missing: List[Tuple[int, Edge]] = []

    for traj_idx, trajectory in enumerate(edge_trajectories):
        for raw_edge in trajectory:
            edge = (str(raw_edge[0]), str(raw_edge[1]))
            if edge not in valid_edges:
                missing.append((traj_idx, edge))
                if len(missing) >= max_examples:
                    break
        if len(missing) >= max_examples:
            break

    if missing:
        raise ValueError(
            "Some expert transitions are not present in the training subgraph. "
            f"First examples: {missing}"
        )


def train_configurable_maxent_irl(
    *,
    features_tensor: torch.Tensor,
    expert_edge_trajectories: Sequence[Sequence[Edge]],
    env: Mapping[str, Any],
    learning_rate: float,
    gamma: float,
    l2_regularization: float,
    hidden_sizes: Sequence[int],
    output_scale: float = 1.0,
    epochs: int = 3000,
    warmup_epochs: int = 200,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-5,
    seed: int = 42,
    checkpoint_path: Optional[os.PathLike] = None,
    device: Optional[str] = None,
    verbose_every: int = 10,
) -> Tuple[RewardNet, Dict[str, Any]]:
    """Train Deep MaxEnt IRL using directed-edge occupancy differences.

    This is the training function used by the formal parameter/feature
    robustness experiments. In the grid-search notebook, the experiment
    settings are passed explicitly (e.g. gamma, learning rate, L2, network
    width, epoch cap, warmup, patience, and output scale).

    The optimization signal is

        mu_E(e) - mu_model(e)

    applied through the surrogate objective

        loss = - sum_e [(mu_E(e) - mu_model(e)) * R_theta(e)].

    Value iteration and expected-visitation propagation are policy-solving
    steps and are intentionally kept outside autograd. Monitoring and best-model
    selection use directed-edge visitation RMSE.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")

    if learning_rate <= 0 or l2_regularization < 0:
        raise ValueError(
            "learning_rate must be positive and L2 must be non-negative."
        )

    seed_everything(seed)

    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    env_n = _normalize_env(env)

    edge_trajs = normalize_edge_trajectories(
        expert_edge_trajectories
    )

    validate_edge_trajectories(
        edge_trajs,
        env_n["edge_to_idx"],
    )

    features = (
        features_tensor
        .detach()
        .clone()
        .float()
        .to(device_obj)
    )

    if features.ndim != 2:
        raise ValueError(
            f"features_tensor must be 2-D, got shape={tuple(features.shape)}"
        )

    num_edges = len(env_n["edge_to_idx"])

    if num_edges != features.shape[0]:
        raise ValueError(
            "Feature rows must match the number of subgraph directed edges."
        )

    edge_indices = sorted(
        int(idx) for idx in env_n["edge_to_idx"].values()
    )

    if edge_indices != list(range(num_edges)):
        raise ValueError(
            "edge_to_idx indices must be continuous and start from 0."
        )

    node_indices = [
        int(idx) for idx in env_n["node_to_idx"].values()
    ]

    if not node_indices:
        raise ValueError("No nodes in environment.")

    num_nodes = max(node_indices) + 1

    # 1. Edge source/destination node indices.
    src_node_idx = torch.empty(
        num_edges,
        dtype=torch.long,
        device=device_obj,
    )

    dst_node_idx = torch.empty(
        num_edges,
        dtype=torch.long,
        device=device_obj,
    )

    for (u, v), edge_idx in env_n["edge_to_idx"].items():
        if u not in env_n["node_to_idx"]:
            raise ValueError(f"Unknown source node {u}.")
        if v not in env_n["node_to_idx"]:
            raise ValueError(f"Unknown destination node {v}.")

        src_node_idx[edge_idx] = env_n["node_to_idx"][u]
        dst_node_idx[edge_idx] = env_n["node_to_idx"][v]

    # 2. Outgoing-edge groups for stochastic-policy calculation.
    outgoing_edge_groups: List[torch.Tensor] = []

    for state in env_n["nodes"]:
        valid_indices: List[int] = []

        for edge in env_n["adj"].get(state, []):
            if edge not in env_n["edge_to_idx"]:
                continue

            destination = edge[1]
            if destination not in env_n["node_to_idx"]:
                continue

            valid_indices.append(env_n["edge_to_idx"][edge])

        if valid_indices:
            outgoing_edge_groups.append(
                torch.tensor(
                    valid_indices,
                    dtype=torch.long,
                    device=device_obj,
                )
            )

    if not outgoing_edge_groups:
        raise ValueError("No valid outgoing actions in environment.")

    # 3. Expert directed-edge visitation and start-length information.
    expert_edge_counts = torch.zeros(
        num_edges,
        dtype=features.dtype,
        device=device_obj,
    )

    lengths_by_start: Dict[int, List[int]] = defaultdict(list)
    valid_trajectory_count = 0

    for trajectory_index, trajectory in enumerate(edge_trajs):
        if not trajectory:
            continue

        # Check trajectory continuity.
        for step_index in range(1, len(trajectory)):
            previous_destination = trajectory[step_index - 1][1]
            current_source = trajectory[step_index][0]

            if previous_destination != current_source:
                raise ValueError(
                    f"Trajectory {trajectory_index} is not continuous "
                    f"between steps {step_index - 1} and {step_index}: "
                    f"{previous_destination} != {current_source}"
                )

        for edge in trajectory:
            edge_idx = env_n["edge_to_idx"][edge]
            expert_edge_counts[edge_idx] += 1.0

        start_state = trajectory[0][0]

        if start_state not in env_n["node_to_idx"]:
            raise ValueError(
                f"Unknown trajectory start state: {start_state}"
            )

        start_idx = env_n["node_to_idx"][start_state]
        lengths_by_start[start_idx].append(len(trajectory))
        valid_trajectory_count += 1

    if valid_trajectory_count == 0:
        raise ValueError(
            "No valid expert trajectories remain after validation."
        )

    expert_total_transitions = expert_edge_counts.sum()

    if expert_total_transitions.item() <= 0:
        raise ValueError(
            "No valid expert transitions remain after validation."
        )

    expert_edge_share = (
        expert_edge_counts / expert_total_transitions
    )

    # 4. Preserve empirical joint distribution of start state and trajectory length.
    start_indices = sorted(lengths_by_start.keys())
    num_starts = len(start_indices)

    max_trajectory_length = max(
        max(lengths) for lengths in lengths_by_start.values()
    )

    start_state_matrix = torch.zeros(
        (num_starts, num_nodes),
        dtype=features.dtype,
        device=device_obj,
    )

    active_counts = torch.zeros(
        (num_starts, max_trajectory_length),
        dtype=features.dtype,
        device=device_obj,
    )

    for row_idx, start_idx in enumerate(start_indices):
        start_state_matrix[row_idx, start_idx] = 1.0
        lengths = lengths_by_start[start_idx]

        for t in range(max_trajectory_length):
            active_counts[row_idx, t] = float(
                sum(
                    trajectory_length > t
                    for trajectory_length in lengths
                )
            )

    scatter_destination_idx = (
        dst_node_idx
        .unsqueeze(0)
        .expand(num_starts, -1)
    )

    # 5. Model and optimizer.
    model = RewardNet(
        input_size=int(features.shape[1]),
        hidden_sizes=list(hidden_sizes),
        output_scale=float(output_scale),
    ).to(device_obj)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(l2_regularization),
    )

    loss_history: List[float] = []
    surrogate_loss_history: List[float] = []
    visitation_mse_history: List[float] = []
    visitation_max_abs_error_history: List[float] = []
    visitation_rmse_history: List[float] = []
    model_mass_history: List[float] = []

    best_visitation_mse = float("inf")
    best_monitor_rmse = float("inf")
    best_surrogate_loss = float("nan")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    started = time.perf_counter()

    # 6. Training.
    for epoch in range(int(epochs)):
        optimizer.zero_grad()

        # A. Reward network (keeps gradient).
        rewards = model(features).squeeze(-1)

        # B-D. Policy solving and expected visitation are outside autograd.
        with torch.no_grad():
            rewards_for_policy = rewards.detach()

            # B. Soft value iteration.
            values_np, _ = classic_value_iteration_action(
                rewards_for_policy.cpu().numpy(),
                gamma=float(gamma),
                nodes=env_n["nodes"],
                adj=env_n["adj"],
                edge_to_idx=env_n["edge_to_idx"],
                node_to_idx=env_n["node_to_idx"],
            )

            values = torch.as_tensor(
                values_np,
                dtype=rewards.dtype,
                device=device_obj,
            )

            # C. Stochastic soft policy:
            # pi(a|s) proportional to exp[R(s,a) + gamma * V(s')].
            edge_policy_probs = torch.zeros_like(
                rewards_for_policy
            )

            for edge_idx_tensor in outgoing_edge_groups:
                destination_indices = dst_node_idx[edge_idx_tensor]

                q_values = (
                    rewards_for_policy[edge_idx_tensor]
                    + float(gamma) * values[destination_indices]
                )

                probs = torch.softmax(q_values, dim=0)

                edge_policy_probs.index_copy_(
                    0,
                    edge_idx_tensor,
                    probs,
                )

            # D. Exact expected forward visitation propagation.
            state_probs = start_state_matrix.clone()
            model_edge_counts = torch.zeros_like(rewards_for_policy)

            for t in range(max_trajectory_length):
                edge_flow_probs = (
                    state_probs[:, src_node_idx]
                    * edge_policy_probs.unsqueeze(0)
                )

                step_active_counts = (
                    active_counts[:, t].unsqueeze(1)
                )

                model_edge_counts += torch.sum(
                    step_active_counts * edge_flow_probs,
                    dim=0,
                )

                next_state_probs = torch.zeros(
                    (num_starts, num_nodes),
                    dtype=rewards.dtype,
                    device=device_obj,
                )

                next_state_probs.scatter_add_(
                    1,
                    scatter_destination_idx,
                    edge_flow_probs,
                )

                state_probs = next_state_probs

            # E1. Same denominator for expert and model visitation.
            model_edge_share = (
                model_edge_counts / expert_total_transitions
            )

            # E2. Occupancy gap: positive means expert > model.
            occupancy_gap = (
                expert_edge_share - model_edge_share
            )

            # Monitoring metrics.
            visitation_difference = (
                model_edge_share - expert_edge_share
            )

            visitation_mse = torch.mean(
                visitation_difference.square()
            )

            current_visitation_mse = float(
                visitation_mse.item()
            )

            current_rmse = float(
                torch.sqrt(visitation_mse).item()
            )

            current_max_error = float(
                torch.max(
                    torch.abs(visitation_difference)
                ).item()
            )

            model_mass = float(
                model_edge_share.sum().item()
            )

        # E3. MaxEnt IRL occupancy-gradient surrogate.
        occupancy_gap_detached = occupancy_gap.detach()

        loss = -torch.sum(
            occupancy_gap_detached * rewards
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite surrogate loss at epoch {epoch}: "
                f"{loss.item()}"
            )

        current_surrogate_loss = float(loss.item())

        # F. Store diagnostics.
        # Legacy behavior: loss_history stores visitation MSE.
        loss_history.append(current_visitation_mse)
        surrogate_loss_history.append(current_surrogate_loss)
        visitation_mse_history.append(current_visitation_mse)
        visitation_rmse_history.append(current_rmse)
        visitation_max_abs_error_history.append(current_max_error)
        model_mass_history.append(model_mass)

        # G. Best model / early stopping monitor: visitation RMSE.
        if current_rmse < best_monitor_rmse - float(min_delta):
            best_monitor_rmse = current_rmse
            best_visitation_mse = current_visitation_mse
            best_surrogate_loss = current_surrogate_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0

        elif epoch >= int(warmup_epochs):
            patience_counter += 1

        # H. Backpropagation.
        loss.backward()
        optimizer.step()

        # I. Logging.
        if verbose_every and (epoch + 1) % int(verbose_every) == 0:
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"surrogate={current_surrogate_loss:.8f} "
                f"vis_mse={current_visitation_mse:.8f} "
                f"rmse={current_rmse:.6f} "
                f"max_error={current_max_error:.6f} "
                f"mass={model_mass:.6f} "
                f"best_rmse={best_monitor_rmse:.6f} "
                f"patience={patience_counter}/{early_stopping_patience}"
            )

        # J. Early stopping.
        if (
            epoch >= int(warmup_epochs)
            and patience_counter >= int(early_stopping_patience)
        ):
            break

    training_seconds = time.perf_counter() - started

    if best_state is None:
        raise RuntimeError(
            "Training ended without a valid best model."
        )

    model.load_state_dict(best_state)
    model.eval()

    result: Dict[str, Any] = {
        "best_epoch": int(best_epoch),
        # Kept for compatibility: best_loss is best visitation MSE.
        "best_loss": float(best_visitation_mse),
        "best_visitation_mse": float(best_visitation_mse),
        "best_visitation_rmse": float(best_monitor_rmse),
        "best_surrogate_loss": float(best_surrogate_loss),
        "epochs_run": len(loss_history),
        "training_seconds": float(training_seconds),
        "loss_history": loss_history,
        "surrogate_loss_history": surrogate_loss_history,
        "visitation_mse_history": visitation_mse_history,
        "visitation_rmse_history": visitation_rmse_history,
        "visitation_max_abs_error_history": visitation_max_abs_error_history,
        "model_mass_history": model_mass_history,
        "train_final_visitation_mse": float(
            visitation_mse_history[-1]
        ),
        "train_final_visitation_rmse": float(
            visitation_rmse_history[-1]
        ),
        "train_final_visitation_max_abs_error": float(
            visitation_max_abs_error_history[-1]
        ),
        "train_final_model_mass": float(
            model_mass_history[-1]
        ),
        # Legacy aliases retained by the formal experiment code.
        "local_mse_history": visitation_mse_history,
        "local_max_abs_error_history": visitation_max_abs_error_history,
        "train_final_local_mse": float(
            visitation_mse_history[-1]
        ),
        "train_final_local_max_abs_error": float(
            visitation_max_abs_error_history[-1]
        ),
        "from_checkpoint": False,
    }

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "state_dict": model.state_dict(),
            "config": model.config,
            "experiment_config": {
                "gamma": float(gamma),
                "learning_rate": float(learning_rate),
                "l2_regularization": float(l2_regularization),
                "hidden_sizes": list(hidden_sizes),
                "output_scale": float(output_scale),
                "seed": int(seed),
                "epochs": int(epochs),
                "warmup_epochs": int(warmup_epochs),
                "early_stopping_patience": int(
                    early_stopping_patience
                ),
                "min_delta": float(min_delta),
                "training_target": (
                    "maxent_directed_edge_occupancy_gradient"
                ),
                "monitoring_target": (
                    "directed_edge_visitation_rmse"
                ),
                "trajectory_constraint": (
                    "empirical_start_length_joint_distribution"
                ),
                "forward_method": (
                    "deterministic_expected_probability_propagation"
                ),
                "value_iteration": "soft_logsumexp",
            },
            **result,
        }

        torch.save(payload, checkpoint_path)

    return model, result
