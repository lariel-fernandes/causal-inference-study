"""Structure identification using multivariate trace criteria."""
import logging
import math
from enum import Enum
from typing import Sequence, Dict, Any, Type, Iterable, Callable, Tuple, Set, List

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

NodeId = int | str


class Relation(Enum):
    """Possible resolutions of ambiguous edges in the CPDAG."""
    CAUSAL = "->"
    ANTI_CAUSAL = "<-"
    MUTUAL = "<->"
    HIDDEN_COMMON_CAUSE = "<-?->"


def build_graph(edges: Iterable[Tuple[NodeId, NodeId, Relation | None]]) -> nx.DiGraph:
    """Build graph from edges."""
    graph = nx.DiGraph()
    for src, dst, rel in edges:
        graph.add_edge(src, dst, relation=rel)
    return graph


def draw_graph(
    graph: nx.DiGraph,
    pos: dict | None = None,
    edge_attr: str | None = None,
    node_opts: dict | None = None,
    edge_opts: dict | None = None,
    node_label_opts: dict | None = None,
    edge_label_opts: dict | None = None,
    map_edge_labels: Callable[[str], str] = lambda x: x,
) -> None:
    """Draw graph."""
    pos = nx.planar_layout(graph) if pos is None else pos
    nx.draw_networkx_nodes(graph, pos, **(node_opts or {}))
    nx.draw_networkx_labels(graph, pos, **(node_label_opts or {}))
    relation = nx.get_edge_attributes(graph, name="relation")

    if edge_attr:
        nx.draw_networkx_edge_labels(
            graph,
            pos=pos,
            **(edge_label_opts or {}),
            edge_labels={k: map_edge_labels(v) for k, v in nx.get_edge_attributes(graph, edge_attr).items()},
        )

    for edge in graph.edges():
        style = "solid"

        match relation[edge]:
            case Relation.CAUSAL:
                arrow_style = "-|>"
            case Relation.ANTI_CAUSAL:
                arrow_style = "<|-"
            case Relation.MUTUAL:
                arrow_style = "<|-|>"
            case Relation.HIDDEN_COMMON_CAUSE:
                style = "dotted"
                arrow_style = "<|-|>"
            case _:
                arrow_style = "-"

        nx.draw_networkx_edges(
            graph,
            pos=pos,
            arrows=True,
            style=style,
            edgelist=[edge],
            **(edge_opts or {}),
            arrowstyle=arrow_style,
        )


def trace(m: np.ndarray) -> float:
    """Get matrix trace."""
    return np.linalg.eigvals(m).sum()


def tracial_dependency_ratio(x: np.ndarray, w: np.ndarray) -> float:
    """Get tracial dependency ratio between predictor variable and weights."""
    cov_x = np.cov(x.T)
    return trace(w.dot(cov_x).dot(w.T)) / (trace(w.dot(w.T)) * trace(cov_x))


class NodeRegression(torch.nn.Module):
    """Regression of a node on its parents."""

    def __init__(
        self,
        P: int,
        D: int,
        H: int,
        dtype: torch.dtype = torch.float64,
        metadata: dict | None = None,
    ):
        super().__init__()
        self.metadata = metadata or {}
        self.A = torch.nn.Parameter(torch.rand((P, H, D), dtype=dtype))
        self.B = torch.nn.Parameter(torch.rand((D, H * P), dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.A @ torch.as_tensor(x).unsqueeze(2).transpose(-1, -2)
        return h.reshape(len(x), -1) @ self.B.T

    def get_parent_weights(self, parend_id: int) -> np.ndarray:
        return self.A[parend_id].detach().numpy()

    def fit(
        self,
        target: torch.Tensor,
        parents: Sequence[torch.Tensor],
        epochs: int = 100,
        batch_size: int = 100,
        optimizer_type: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        prog_bar_factory: Callable[[Iterable[int]], tqdm] | None = None,
        register_hooks: Callable[[torch.nn.Module], None] = None,
        n_epoch_logs: int = 10,
    ) -> float | None:
        criterion = torch.nn.MSELoss()
        optimizer_kwargs = optimizer_kwargs or {"lr": 0.01}
        optimizer = optimizer_type(self.parameters(), **optimizer_kwargs)

        x = torch.stack([torch.as_tensor(p) for p in parents], dim=1)
        y = torch.as_tensor(target)
        n_batches = math.ceil(len(target) / batch_size)
        loss = None

        if register_hooks:
            register_hooks(self)

        for epoch in prog_bar_factory(range(epochs)) if prog_bar_factory else range(epochs):
            for batch_idx in range(n_batches):
                start = batch_size * batch_idx
                end = start + batch_size
                loss = criterion(self.forward(x[start:end]), y[start:end])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if loss is not None and n_epoch_logs > 0 and epoch % int(epochs / n_epoch_logs) == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        return loss


def get_scores(
    graph: nx.DiGraph,
    node_samples: Dict[NodeId, np.ndarray],
    H: int | None = None,
    regress_opts: dict | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Dict[Tuple[NodeId, NodeId], List[float]]:
    """Get causal and anticausal scores for each ambiguous edge in graph."""
    for sample in node_samples.values():
        D = sample.shape[1]
        break
    else:
        return {}

    H = H or D
    assert set(graph.nodes) == set(node_samples.keys())
    relations = nx.get_edge_attributes(graph, name="relation")
    assert all(r in (Relation.CAUSAL, None) for r in relations.values())
    ambiguous_edges = {edge: [0.0, 0.0] for edge, relation in relations.items() if relation is None}

    ambiguous_nodes = {}
    for edge in ambiguous_edges:
        for node in edge:
            ambiguous_nodes.setdefault(node, set())

    for (src, dst), relation in relations.items():
        if dst in ambiguous_nodes:
            ambiguous_nodes[dst].add(src)
        if src in ambiguous_nodes and relation is None:
            ambiguous_nodes[src].add(dst)

    for node, predictors in ambiguous_nodes.items():
        scores = _get_node_scores(
            node,
            predictors=predictors,
            node_samples=node_samples,
            D=D,
            H=H,
            dtype=dtype,
            device=device,
            regress_opts=regress_opts,
        )

        for neighbor, score in scores.items():
            if (edge := (neighbor, node)) in ambiguous_edges:
                ambiguous_edges[edge][0] = float(score)
            elif (edge := (node, neighbor)) in ambiguous_edges:
                ambiguous_edges[edge][1] = float(score)

    return ambiguous_edges


def _get_node_scores(
    node: NodeId,
    predictors: Set[NodeId],
    node_samples: Dict[NodeId, np.ndarray],
    D: int,
    H: int,
    dtype: torch.dtype,
    device: torch.device,
    regress_opts: dict | None = None,
) -> Dict[NodeId, float]:
    """Regress node on its predictors and calculate tracial dependency ratio per perdictor."""
    r = NodeRegression(P=len(predictors), D=D, H=H, dtype=dtype, metadata={"node": node}).to(device)

    r.fit(
        target=torch.as_tensor(node_samples[node], dtype=dtype, device=device),
        parents=[torch.as_tensor(node_samples[n], dtype=dtype, device=device) for n in predictors],
        **regress_opts,
    )

    return {
        n: tracial_dependency_ratio(node_samples[n], r.get_parent_weights(i))
        for i, n in enumerate(predictors)
    }


def adjusted_graph(
    graph: nx.DiGraph,
    node_samples: Dict[NodeId, np.ndarray],  # noqa
    scores: Dict[Tuple[NodeId, NodeId], Tuple[float, float]],
) -> nx.DiGraph:
    """Get adjusted graph according to causal and anticausal scores of ambiguous edges in graph."""
    adjusted = nx.DiGraph()
    adjusted.add_nodes_from(graph.nodes)
    relations = nx.get_edge_attributes(graph, name="relation")

    for edge in graph.edges():
        src, dst = edge

        if relation := relations[edge]:
            adjusted.add_edge(src, dst, relation=relation)
        else:
            causal, anti_causal = scores[edge]
            if causal > anti_causal:
                adjusted.add_edge(src, dst, relation=Relation.CAUSAL, score=causal)
            else:
                adjusted.add_edge(dst, src, relation=Relation.CAUSAL, score=anti_causal)

    return adjusted
