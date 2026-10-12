from __future__ import annotations

import math

import torch
from torch import nn

from .specs import ModelSpec


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(values)
        if self.padding:
            convolved = convolved[..., :-self.padding]
        return torch.relu(values + self.dropout(self.norm(convolved)))


class MultiHeadGraphLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        candidate_mask: torch.Tensor,
        edge_features: torch.Tensor,
        dropout: float,
        heads: int,
    ):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by graph attention heads")
        self.heads = heads
        self.head_size = hidden_size // heads
        destination, source = torch.where(candidate_mask)
        self.register_buffer("destination", destination)
        self.register_buffer("source", source)
        self.register_buffer("edge_features", edge_features[destination, source])
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.edge_bias = nn.Linear(edge_features.shape[-1], heads, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = math.sqrt(self.head_size)

    def forward(
        self, nodes: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, node_count, hidden_size = nodes.shape
        query = self.query(nodes).reshape(batch, node_count, self.heads, self.head_size)
        key = self.key(nodes).reshape(batch, node_count, self.heads, self.head_size)
        value = self.value(nodes).reshape(batch, node_count, self.heads, self.head_size)
        scores = (
            query[:, self.destination] * key[:, self.source]
        ).sum(-1) / self.scale
        scores = scores + self.edge_bias(self.edge_features)[None]
        head_ids = torch.arange(self.heads, device=nodes.device)
        grouped_destination = (
            self.destination[:, None] * self.heads + head_ids[None]
        ).reshape(-1)
        flat_destinations = grouped_destination[None].expand(batch, -1)
        flat_scores = scores.reshape(batch, -1)
        maxima = torch.full(
            (batch, node_count * self.heads),
            torch.finfo(scores.dtype).min,
            device=scores.device,
            dtype=scores.dtype,
        )
        maxima.scatter_reduce_(
            1, flat_destinations, flat_scores, reduce="amax", include_self=True
        )
        flat_weights = torch.exp(
            flat_scores - maxima.gather(1, flat_destinations)
        )
        denominators = torch.zeros_like(maxima)
        denominators.scatter_add_(1, flat_destinations, flat_weights)
        flat_weights = flat_weights / denominators.gather(
            1, flat_destinations
        ).clamp_min(1e-8)
        weights = flat_weights.reshape(batch, -1, self.heads)
        messages = value[:, self.source] * self.dropout(weights)[..., None]
        aggregate = torch.zeros(
            batch, node_count, self.heads, self.head_size,
            device=nodes.device, dtype=nodes.dtype
        )
        aggregate.scatter_add_(
            1,
            self.destination[None, :, None, None].expand(
                batch, -1, self.heads, self.head_size
            ),
            messages,
        )
        output = self.norm(self.output(aggregate.reshape(batch, node_count, hidden_size)))
        if not return_attention:
            return output
        dense = torch.zeros(
            batch, node_count, node_count, device=nodes.device, dtype=nodes.dtype
        )
        dense[:, self.destination, self.source] = weights.mean(dim=-1)
        return output, dense


class DengueForecastNet(nn.Module):
    """One implementation controlled by prespecified M1--M6 component flags."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        local_feature_size: int,
        climate_feature_size: int,
        seasonal_feature_size: int,
        node_count: int,
        horizon_count: int,
        hidden_size: int,
        commune_embedding_size: int,
        dilations: tuple[int, ...],
        kernel_size: int,
        dropout: float,
        minimum_dispersion: float,
        likelihood: str = "negative_binomial",
        maximum_log_sigma: float = 3.0,
        candidate_mask: torch.Tensor,
        edge_features: torch.Tensor,
        fixed_weights: torch.Tensor,
        graph_heads: int = 4,
    ):
        super().__init__()
        self.spec = spec
        self.node_count = node_count
        self.horizon_count = horizon_count
        self.minimum_dispersion = minimum_dispersion
        self.likelihood = likelihood
        self.maximum_log_sigma = maximum_log_sigma
        input_size = local_feature_size + (
            climate_feature_size if spec.climate_context else 0
        )
        self.temporal_encoder = spec.temporal_encoder
        if self.temporal_encoder == "tcn":
            self.input_projection = nn.Conv1d(input_size, hidden_size, 1)
            self.temporal_network = nn.Sequential(
                *[
                    CausalBlock(hidden_size, kernel_size, dilation, dropout)
                    for dilation in dilations
                ]
            )
        elif self.temporal_encoder == "cnn":
            self.input_projection = nn.Conv1d(input_size, hidden_size, 1)
            self.temporal_network = nn.Sequential(
                *[
                    CausalBlock(hidden_size, kernel_size, 1, dropout)
                    for _ in dilations
                ]
            )
        elif self.temporal_encoder == "lstm":
            self.temporal_network = nn.LSTM(
                input_size, hidden_size, num_layers=2, batch_first=True,
                dropout=dropout,
            )
        elif self.temporal_encoder == "cnn_lstm":
            self.input_projection = nn.Conv1d(input_size, hidden_size, 1)
            self.cnn_frontend = CausalBlock(hidden_size, kernel_size, 1, dropout)
            self.temporal_network = nn.LSTM(
                hidden_size, hidden_size, num_layers=1, batch_first=True
            )
        elif self.temporal_encoder == "last":
            self.last_projection = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unsupported temporal encoder: {self.temporal_encoder}")
        if spec.seasonal_memory:
            self.seasonal_encoder = nn.Sequential(
                nn.Linear(seasonal_feature_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        if spec.commune_conditioning:
            self.commune_embedding = nn.Embedding(node_count, commune_embedding_size)
            self.condition = nn.Linear(commune_embedding_size, hidden_size * 2)
        if spec.graph_mode == "fixed":
            self.register_buffer("fixed_weights", fixed_weights)
            self.fixed_projection = nn.Linear(hidden_size, hidden_size)
        elif spec.graph_mode == "learned":
            self.graph = MultiHeadGraphLayer(
                hidden_size, candidate_mask, edge_features, dropout, graph_heads
            )
        if spec.dynamic_gate:
            self.spatial_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid(),
            )
        self.horizon_embedding = nn.Embedding(horizon_count, 8)
        representation_size = hidden_size * (
            2 if spec.seasonal_memory else 1
        ) + 8
        self.representation = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(horizon_count)]
        )
        self.dispersion_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(horizon_count)]
        )
        # Optional auxiliary early-warning head. It does not replace the
        # probabilistic count output; the network still predicts distribution
        # parameters and predictive case samples exactly as before.
        self.outbreak_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(horizon_count)]
        )

    def forward(
        self,
        local: torch.Tensor,
        climate: torch.Tensor,
        seasonal: torch.Tensor,
        *,
        return_representation: bool = False,
        return_attention: bool = False,
        return_outbreak_logits: bool = False,
    ):
        batch, nodes, lookback, _ = local.shape
        inputs = (
            torch.cat([local, climate], dim=-1)
            if self.spec.climate_context
            else local
        )
        temporal = inputs.reshape(batch * nodes, lookback, -1)
        if self.temporal_encoder in {"tcn", "cnn"}:
            convolutional = self.input_projection(temporal.transpose(1, 2))
            local_state = self.temporal_network(convolutional)[..., -1]
        elif self.temporal_encoder == "lstm":
            encoded, _ = self.temporal_network(temporal)
            local_state = encoded[:, -1]
        elif self.temporal_encoder == "last":
            local_state = self.last_projection(temporal[:, -1])
        else:
            convolutional = self.cnn_frontend(
                self.input_projection(temporal.transpose(1, 2))
            ).transpose(1, 2)
            encoded, _ = self.temporal_network(convolutional)
            local_state = encoded[:, -1]
        local_state = local_state.reshape(batch, nodes, -1)
        if self.spec.commune_conditioning:
            node_ids = torch.arange(nodes, device=local.device)
            scale_shift = self.condition(self.commune_embedding(node_ids))
            scale, shift = scale_shift.chunk(2, dim=-1)
            local_state = (1.0 + 0.1 * torch.tanh(scale))[None] * local_state + shift[None]

        attention = None
        if self.spec.graph_mode == "fixed":
            spatial = torch.einsum("ij,bjh->bih", self.fixed_weights, local_state)
            spatial = self.fixed_projection(spatial)
            local_state = local_state + spatial
        elif self.spec.graph_mode == "learned":
            graph_output = self.graph(
                local_state, return_attention=return_attention
            )
            if return_attention:
                spatial, attention = graph_output
            else:
                spatial = graph_output
            if self.spec.dynamic_gate:
                gate = self.spatial_gate(torch.cat([local_state, spatial], dim=-1))
                local_state = local_state + gate * spatial
            else:
                local_state = local_state + spatial

        representations, means, dispersions, outbreak_logits = [], [], [], []
        for index in range(self.horizon_count):
            pieces = [local_state]
            if self.spec.seasonal_memory:
                pieces.append(self.seasonal_encoder(seasonal[:, :, index]))
            horizon = self.horizon_embedding(
                torch.full((nodes,), index, device=local.device, dtype=torch.long)
            )
            pieces.append(horizon[None].expand(batch, -1, -1))
            hidden = self.representation(torch.cat(pieces, dim=-1))
            representations.append(hidden)
            outbreak_logits.append(self.outbreak_heads[index](hidden))
            location = self.mean_heads[index](hidden)
            if self.likelihood == "lognormal":
                means.append(location)
                scale = (
                    torch.nn.functional.softplus(self.dispersion_heads[index](hidden))
                    + self.minimum_dispersion
                ).clamp_max(self.maximum_log_sigma)
                dispersions.append(scale)
            else:
                means.append(torch.nn.functional.softplus(location))
                dispersions.append(
                    torch.nn.functional.softplus(self.dispersion_heads[index](hidden))
                    + self.minimum_dispersion
                )
        mean = torch.cat(means, dim=-1)
        dispersion = torch.cat(dispersions, dim=-1)
        representation = torch.stack(representations, dim=2)
        outbreak_logit = torch.cat(outbreak_logits, dim=-1)
        if return_outbreak_logits:
            return mean, dispersion, representation, attention, outbreak_logit
        if return_representation or return_attention:
            return mean, dispersion, representation, attention
        return mean, dispersion


def negative_binomial_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    horizon_weights: torch.Tensor,
    observation_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mean = mean.clamp_min(1e-6)
    dispersion = dispersion.clamp_min(1e-6)
    probabilities = mean / (dispersion + mean)
    distribution = torch.distributions.NegativeBinomial(
        total_count=dispersion,
        probs=probabilities,
    )
    losses = -distribution.log_prob(target)
    if observation_weights is not None:
        losses = losses * observation_weights
    return (losses * horizon_weights[None, None]).mean()


def lognormal_loss(
    target: torch.Tensor,
    log_location: torch.Tensor,
    log_scale: torch.Tensor,
    horizon_weights: torch.Tensor,
    observation_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL for log1p counts, matching the original TCN pipeline."""
    transformed = torch.log1p(target.clamp_min(0))
    scale = log_scale.clamp_min(1e-6)
    losses = torch.log(scale) + 0.5 * ((transformed - log_location) / scale) ** 2
    if observation_weights is not None:
        losses = losses * observation_weights
    return (losses * horizon_weights[None, None]).mean()
