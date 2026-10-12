from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateGraph:
    units: tuple[str, ...]
    mask: np.ndarray
    edge_features: np.ndarray
    fixed_weights: np.ndarray


def build_candidate_graph(
    units: tuple[str, ...],
    centroids_path: str | Path,
    adjacency_path: str | Path,
    *,
    k_nearest: int,
    maximum_distance_km: float,
    include_administrative_edges: bool,
) -> CandidateGraph:
    centroids = pd.read_csv(centroids_path, dtype={"fcode": str}).set_index("fcode")
    missing = sorted(set(units).difference(centroids.index))
    if missing:
        raise ValueError(f"Missing centroids for communes: {missing[:10]}")
    coords = centroids.loc[list(units), ["latitude", "longitude"]].to_numpy(float)
    distance = _haversine(coords)
    node_count = len(units)
    k = min(max(int(k_nearest), 1), node_count - 1)
    nearest = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    knn = np.zeros((node_count, node_count), dtype=bool)
    knn[np.arange(node_count)[:, None], nearest] = True
    knn &= distance <= float(maximum_distance_km)
    mask = knn | knn.T
    administrative = np.zeros_like(mask)
    if include_administrative_edges:
        edges = pd.read_csv(adjacency_path, dtype=str)
        aliases = [
            ("from_fcode", "to_fcode"),
            ("from", "to"),
            ("source", "target"),
        ]
        columns = next(
            ((left, right) for left, right in aliases if {left, right} <= set(edges)),
            None,
        )
        if columns is None:
            raise ValueError(
                "Adjacency must contain from_fcode/to_fcode, from/to, "
                "or source/target"
            )
        lookup = {unit: index for index, unit in enumerate(units)}
        for source, target in edges[list(columns)].itertuples(index=False, name=None):
            if source in lookup and target in lookup:
                i, j = lookup[source], lookup[target]
                administrative[i, j] = administrative[j, i] = True
        mask |= administrative
    np.fill_diagonal(mask, True)

    finite = distance[np.isfinite(distance)]
    scale = max(float(np.median(finite)), 1.0)
    distance_feature = np.zeros_like(distance)
    distance_feature[np.isfinite(distance)] = np.exp(
        -distance[np.isfinite(distance)] / scale
    )
    edge_features = np.stack(
        [distance_feature, knn.astype(float), administrative.astype(float)],
        axis=-1,
    ).astype(np.float32)
    edge_features[~mask] = 0.0
    fixed = distance_feature * mask
    fixed[np.diag_indices_from(fixed)] = 1.0
    fixed /= fixed.sum(axis=1, keepdims=True).clip(min=1e-8)
    return CandidateGraph(
        units=units,
        mask=mask,
        edge_features=edge_features,
        fixed_weights=fixed.astype(np.float32),
    )


def _haversine(latitude_longitude: np.ndarray) -> np.ndarray:
    radians = np.radians(latitude_longitude)
    lat = radians[:, 0]
    lon = radians[:, 1]
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    output = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
    np.fill_diagonal(output, np.inf)
    return output
