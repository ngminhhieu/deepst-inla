import torch

from dengue_h3.models import DengueForecastNet
from dengue_h3.specs import get_spec


def _network(name: str) -> DengueForecastNet:
    nodes = 4
    mask = torch.eye(nodes, dtype=torch.bool)
    mask[0, 1] = mask[1, 0] = True
    edge_features = torch.zeros(nodes, nodes, 3)
    fixed = mask.float()
    fixed /= fixed.sum(dim=1, keepdim=True)
    return DengueForecastNet(
        spec=get_spec(name),
        local_feature_size=5,
        climate_feature_size=3,
        seasonal_feature_size=14,
        node_count=nodes,
        horizon_count=3,
        hidden_size=8,
        commune_embedding_size=4,
        dilations=(1, 2),
        kernel_size=3,
        dropout=0.0,
        minimum_dispersion=0.05,
        likelihood="lognormal",
        candidate_mask=mask,
        edge_features=edge_features,
        fixed_weights=fixed,
        graph_heads=2,
    )


def test_all_temporal_encoders_produce_three_horizon_distributions():
    local = torch.randn(2, 4, 12, 5)
    climate = torch.randn(2, 4, 12, 3)
    seasonal = torch.randn(2, 4, 3, 14)
    for name in (
        "CNN", "LSTM", "CNN-LSTM", "GNN", "LSTM-FULL", "LSTM-GNN",
        "CNN-LSTM-FULL", "CNN-LSTM-GNN", "M1",
    ):
        location, scale = _network(name)(local, climate, seasonal)
        assert location.shape == scale.shape == (2, 4, 3)
        assert torch.all(scale > 0)


def test_multihead_graph_returns_mean_attention_map():
    local = torch.randn(2, 4, 12, 5)
    climate = torch.randn(2, 4, 12, 3)
    seasonal = torch.randn(2, 4, 3, 14)
    location, scale, representation, attention = _network("M6")(
        local, climate, seasonal, return_attention=True
    )
    assert location.shape == scale.shape == (2, 4, 3)
    assert representation.shape == (2, 4, 3, 8)
    assert attention.shape == (2, 4, 4)
