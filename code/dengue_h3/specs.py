from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    temporal_encoder: str = "tcn"
    seasonal_memory: bool = False
    climate_context: bool = False
    commune_conditioning: bool = False
    graph_mode: str = "none"
    dynamic_gate: bool = False


MODEL_SPECS = {
    "CNN": ModelSpec(temporal_encoder="cnn", seasonal_memory=True),
    "LSTM": ModelSpec(temporal_encoder="lstm", seasonal_memory=True),
    "CNN-LSTM": ModelSpec(temporal_encoder="cnn_lstm", seasonal_memory=True),
    "GNN": ModelSpec(
        temporal_encoder="last",
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="learned",
        dynamic_gate=True,
    ),
    "LSTM-FULL": ModelSpec(
        temporal_encoder="lstm",
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
    ),
    "LSTM-GNN": ModelSpec(
        temporal_encoder="lstm",
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="learned",
        dynamic_gate=True,
    ),
    "CNN-LSTM-FULL": ModelSpec(
        temporal_encoder="cnn_lstm",
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
    ),
    "CNN-LSTM-GNN": ModelSpec(
        temporal_encoder="cnn_lstm",
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="learned",
        dynamic_gate=True,
    ),
    "M1": ModelSpec(seasonal_memory=True),
    "M2": ModelSpec(seasonal_memory=True, climate_context=True),
    "M3": ModelSpec(
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
    ),
    "M4": ModelSpec(
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="fixed",
    ),
    "M5": ModelSpec(
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="learned",
    ),
    "M6": ModelSpec(
        seasonal_memory=True,
        climate_context=True,
        commune_conditioning=True,
        graph_mode="learned",
        dynamic_gate=True,
    ),
}


def get_spec(name: str) -> ModelSpec:
    key = name.upper()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose one of {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[key]
