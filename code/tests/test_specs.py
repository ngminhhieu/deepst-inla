from dengue_h3.specs import MODEL_SPECS


def test_ablation_sequence_is_monotonic():
    assert [name for name in MODEL_SPECS if name.startswith("M")] == [
        "M1", "M2", "M3", "M4", "M5", "M6"
    ]
    assert MODEL_SPECS["M1"].graph_mode == "none"
    assert MODEL_SPECS["M4"].graph_mode == "fixed"
    assert MODEL_SPECS["M5"].graph_mode == "learned"
    assert MODEL_SPECS["M6"].dynamic_gate


def test_temporal_benchmarks_share_information_but_change_encoder():
    assert MODEL_SPECS["CNN"].temporal_encoder == "cnn"
    assert MODEL_SPECS["LSTM"].temporal_encoder == "lstm"
    assert MODEL_SPECS["CNN-LSTM"].temporal_encoder == "cnn_lstm"
    for name in ("CNN", "LSTM", "CNN-LSTM"):
        assert MODEL_SPECS[name].seasonal_memory
        assert MODEL_SPECS[name].graph_mode == "none"


def test_external_pattern_components_are_absent():
    fields = " ".join(str(value) for value in MODEL_SPECS.values()).lower()
    assert "opendengue" not in fields
    assert "pattern_bank" not in fields


def test_graph_benchmarks_are_matched_within_encoder_family():
    for base, graph in (
        ("LSTM-FULL", "LSTM-GNN"),
        ("CNN-LSTM-FULL", "CNN-LSTM-GNN"),
        ("M3", "M6"),
    ):
        assert MODEL_SPECS[base].temporal_encoder == MODEL_SPECS[graph].temporal_encoder
        assert MODEL_SPECS[base].seasonal_memory == MODEL_SPECS[graph].seasonal_memory
        assert MODEL_SPECS[base].climate_context == MODEL_SPECS[graph].climate_context
        assert MODEL_SPECS[base].commune_conditioning == MODEL_SPECS[graph].commune_conditioning
        assert MODEL_SPECS[base].graph_mode == "none"
        assert MODEL_SPECS[graph].graph_mode == "learned"
