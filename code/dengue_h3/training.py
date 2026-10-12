from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import SnapshotData
from .distributions import parameter_names, predictive_sample
from .graph import CandidateGraph
from .models import DengueForecastNet, lognormal_loss, negative_binomial_loss
from .specs import get_spec


@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    sd: np.ndarray


@dataclass
class FittedModel:
    name: str
    network: DengueForecastNet
    local_scale: Scale
    climate_scale: Scale
    seasonal_scale: Scale
    history: list[dict[str, float]]
    likelihood: str
    outbreak_loss_weight: float = 0.0
    outbreak_positive_weight: list[float] | None = None
    outbreak_focal_gamma: float = 0.0
    alert_probability_source: str = "samples"
    selection_history: list[dict[str, float]] | None = None


def fit_model(
    name: str,
    data: SnapshotData,
    graph: CandidateGraph,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: dict,
) -> FittedModel:
    if bool(config["training"].get("refit_on_train_validation", False)):
        selection_config = deepcopy(config)
        selection_config["training"]["refit_on_train_validation"] = False
        selected = fit_model(
            name, data, graph, train_mask, validation_mask, selection_config
        )
        best_epoch = int(
            min(selected.history, key=lambda row: row["validation_loss"])["epoch"]
        )
        refit_config = deepcopy(config)
        refit_config["training"]["refit_on_train_validation"] = False
        refit_config["training"]["fixed_epochs"] = True
        refit_config["training"]["epochs"] = best_epoch
        development_mask = train_mask | validation_mask
        refitted = fit_model(
            name,
            data,
            graph,
            development_mask,
            np.zeros_like(development_mask, dtype=bool),
            refit_config,
        )
        refitted.selection_history = selected.history
        return refitted

    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    excluded_units = {str(value) for value in config["training"].get("excluded_units", [])}
    train_node_mask = np.asarray(
        [unit not in excluded_units for unit in data.units], dtype=bool
    )
    if not train_node_mask.any():
        raise ValueError("training.excluded_units removes every commune")
    local_scale = _fit_scale(
        data.local[train_mask][:, train_node_mask], axes=(0, 1, 2)
    )
    climate_scale = _fit_scale(
        data.climate[train_mask][:, train_node_mask], axes=(0, 1, 2)
    )
    seasonal_scale = _fit_scale(
        data.seasonal[train_mask][:, train_node_mask], axes=(0, 1, 2)
    )
    local = _apply_scale(data.local, local_scale)
    climate = _apply_scale(data.climate, climate_scale)
    seasonal = _apply_scale(data.seasonal, seasonal_scale)

    model_config = config["model"]
    network = DengueForecastNet(
        spec=get_spec(name),
        local_feature_size=local.shape[-1],
        climate_feature_size=climate.shape[-1],
        seasonal_feature_size=seasonal.shape[-1],
        node_count=len(data.units),
        horizon_count=len(data.horizons),
        hidden_size=int(model_config["hidden_size"]),
        commune_embedding_size=int(model_config["commune_embedding_size"]),
        dilations=tuple(int(v) for v in model_config["dilations"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
        minimum_dispersion=float(model_config["minimum_dispersion"]),
        likelihood=str(model_config.get("likelihood", "negative_binomial")),
        maximum_log_sigma=float(model_config.get("maximum_log_sigma", 3.0)),
        candidate_mask=torch.from_numpy(graph.mask),
        edge_features=torch.from_numpy(graph.edge_features),
        fixed_weights=torch.from_numpy(graph.fixed_weights),
        graph_heads=int(model_config.get("graph_heads", 4)),
    )
    device = _device(config["training"]["device"])
    network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    horizon_weights = torch.tensor(
        config["model"]["horizon_loss_weights"],
        dtype=torch.float32,
        device=device,
    )
    peak_loss_weight = float(model_config.get("peak_loss_weight", 0.0))
    peak_quantile = float(model_config.get("peak_loss_quantile", 0.90))
    peak_thresholds = torch.from_numpy(
        np.quantile(
            data.targets[train_mask][:, train_node_mask],
            peak_quantile,
            axis=(0, 1),
        ).astype(
            np.float32
        )
    )
    outbreak_loss_weight = float(model_config.get("outbreak_loss_weight", 0.0))
    outbreak_focal_gamma = float(model_config.get("outbreak_focal_gamma", 0.0))
    configured_positive_weight = model_config.get("outbreak_positive_weight", 1.0)
    if configured_positive_weight == "auto":
        train_targets = data.targets[train_mask][:, train_node_mask]
        train_thresholds = data.outbreak_thresholds[train_mask][:, train_node_mask]
        finite_thresholds = np.isfinite(train_thresholds)
        positive = ((train_targets > train_thresholds) & finite_thresholds).sum(
            axis=(0, 1)
        )
        negative = (
            (train_targets <= train_thresholds) & finite_thresholds
        ).sum(axis=(0, 1))
        outbreak_positive_weight = torch.from_numpy(
            np.clip(negative / np.maximum(positive, 1), 1.0, 8.0).astype(
                np.float32
            )
        ).to(device)
    else:
        outbreak_positive_weight = torch.full(
            (len(data.horizons),),
            float(configured_positive_weight),
            dtype=torch.float32,
            device=device,
        )
    alert_probability_source = str(
        model_config.get("alert_probability_source", "samples")
    ).lower()
    if alert_probability_source not in {"samples", "auxiliary"}:
        raise ValueError("model.alert_probability_source must be samples or auxiliary")
    tensors = tuple(
        torch.from_numpy(value)
        for value in (
            local,
            climate,
            seasonal,
            data.targets,
            data.outbreak_thresholds,
        )
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    rng = np.random.default_rng(seed)
    best_state = deepcopy(network.state_dict())
    best_score = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    fixed_epochs = bool(config["training"].get("fixed_epochs", False))

    for epoch in range(int(config["training"]["epochs"])):
        network.train()
        batch_losses = []
        shuffled = rng.permutation(train_indices)
        batch_size = int(config["training"]["batch_size"])
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = network(
                tensors[0][indices].to(device),
                tensors[1][indices].to(device),
                tensors[2][indices].to(device),
                return_outbreak_logits=outbreak_loss_weight > 0,
            )
            if outbreak_loss_weight > 0:
                location, scale, _, _, outbreak_logits = output
            else:
                location, scale = output
                outbreak_logits = None
            loss = _combined_loss(
                tensors[3][indices][:, train_node_mask].to(device),
                tensors[4][indices][:, train_node_mask].to(device),
                location[:, train_node_mask],
                scale[:, train_node_mask],
                (
                    outbreak_logits[:, train_node_mask]
                    if outbreak_logits is not None
                    else None
                ),
                horizon_weights,
                likelihood=network.likelihood,
                peak_thresholds=peak_thresholds.to(device),
                peak_loss_weight=peak_loss_weight,
                outbreak_loss_weight=outbreak_loss_weight,
                outbreak_positive_weight=outbreak_positive_weight,
                outbreak_focal_gamma=outbreak_focal_gamma,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), float(config["training"]["gradient_clip"])
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_loss = (
            float("nan")
            if fixed_epochs
            else _score(
                network,
                tensors,
                validation_indices,
                horizon_weights,
                device,
                network.likelihood,
                peak_thresholds,
                peak_loss_weight,
                outbreak_loss_weight,
                outbreak_positive_weight,
                outbreak_focal_gamma,
                train_node_mask,
            )
        )
        training_loss = float(np.mean(batch_losses))
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if fixed_epochs:
            best_state = deepcopy(network.state_dict())
        elif validation_loss < best_score - 1e-4:
            best_score = validation_loss
            best_state = deepcopy(network.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    network.load_state_dict(best_state)
    # Held-out commune embeddings receive no training signal. Replacing their
    # random initialization by the training-commune mean yields a genuine
    # unseen-area prediction without encoding held-out identities.
    if excluded_units and hasattr(network, "commune_embedding"):
        with torch.no_grad():
            weights = network.commune_embedding.weight
            mask = torch.as_tensor(train_node_mask, device=weights.device)
            weights[~mask] = weights[mask].mean(dim=0)
    network.to("cpu").eval()
    return FittedModel(
        name=name.upper(),
        network=network,
        local_scale=local_scale,
        climate_scale=climate_scale,
        seasonal_scale=seasonal_scale,
        history=history,
        likelihood=network.likelihood,
        outbreak_loss_weight=outbreak_loss_weight,
        outbreak_positive_weight=outbreak_positive_weight.detach().cpu().tolist(),
        outbreak_focal_gamma=outbreak_focal_gamma,
        alert_probability_source=alert_probability_source,
    )


def predict(
    fitted: FittedModel,
    data: SnapshotData,
    mask: np.ndarray,
    *,
    draws: int,
    seed: int,
    export_representations: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    local = torch.from_numpy(_apply_scale(data.local[mask], fitted.local_scale))
    climate = torch.from_numpy(_apply_scale(data.climate[mask], fitted.climate_scale))
    seasonal = torch.from_numpy(_apply_scale(data.seasonal[mask], fitted.seasonal_scale))
    fitted.network.eval()
    with torch.no_grad():
        location, scale, representation, _, outbreak_logits = fitted.network(
            local,
            climate,
            seasonal,
            return_outbreak_logits=True,
        )
    parameter_1 = location.numpy()
    parameter_2 = scale.numpy()
    auxiliary_probability = torch.sigmoid(outbreak_logits).numpy()
    rng = np.random.default_rng(seed)
    samples = predictive_sample(
        parameter_1,
        parameter_2,
        likelihood=fitted.likelihood,
        draws=draws,
        rng=rng,
    )
    parameter_1_name, parameter_2_name = parameter_names(fitted.likelihood)
    selected_origins = data.origins[mask]
    selected_target_dates = data.target_dates[mask]
    selected_targets = data.targets[mask]
    selected_population = data.population[mask]
    selected_thresholds = data.outbreak_thresholds[mask]
    rows = []
    representation_rows = []
    for origin_index, origin in enumerate(selected_origins):
        for node_index, unit in enumerate(data.units):
            for horizon_index, horizon in enumerate(data.horizons):
                values = samples[origin_index, node_index, horizon_index]
                threshold = float(
                    selected_thresholds[origin_index, node_index, horizon_index]
                )
                sample_probability = float((values > threshold).mean())
                aux_probability = float(
                    auxiliary_probability[origin_index, node_index, horizon_index]
                )
                outbreak_probability = (
                    aux_probability
                    if fitted.alert_probability_source == "auxiliary"
                    else sample_probability
                )
                rows.append(
                    {
                        "model": fitted.name,
                        "fcode": unit,
                        "issue_date": origin,
                        "target_date": selected_target_dates[origin_index, horizon_index],
                        "horizon": horizon,
                        "target_cases": selected_targets[
                            origin_index, node_index, horizon_index
                        ],
                        "population": selected_population[
                            origin_index, node_index, horizon_index
                        ],
                        parameter_1_name: float(
                            parameter_1[origin_index, node_index, horizon_index]
                        ),
                        parameter_2_name: float(
                            parameter_2[origin_index, node_index, horizon_index]
                        ),
                        "predicted_mean": float(values.mean()),
                        "predicted_median": float(np.median(values)),
                        "lower_50": float(np.quantile(values, 0.25)),
                        "upper_50": float(np.quantile(values, 0.75)),
                        "lower_80": float(np.quantile(values, 0.10)),
                        "upper_80": float(np.quantile(values, 0.90)),
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "outbreak_threshold": threshold,
                        "outbreak_probability": outbreak_probability,
                        "outbreak_probability_samples": sample_probability,
                        "outbreak_probability_auxiliary": aux_probability,
                        "alert_probability_source": fitted.alert_probability_source,
                    }
                )
                if export_representations:
                    latent = representation[
                        origin_index, node_index, horizon_index
                    ].numpy()
                    representation_rows.append(
                        {
                            "model": fitted.name,
                            "fcode": unit,
                            "issue_date": origin,
                            "target_date": selected_target_dates[
                                origin_index, horizon_index
                            ],
                            "horizon": horizon,
                            "target_cases": selected_targets[
                                origin_index, node_index, horizon_index
                            ],
                            "population": selected_population[
                                origin_index, node_index, horizon_index
                            ],
                            "outbreak_threshold": threshold,
                            **{
                                f"z_{index + 1:02d}": float(value)
                                for index, value in enumerate(latent)
                            },
                        }
                    )
    return pd.DataFrame(rows), samples, pd.DataFrame(representation_rows)


def save_checkpoint(fitted: FittedModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": fitted.name,
            "state_dict": fitted.network.state_dict(),
            "local_scale": fitted.local_scale,
            "climate_scale": fitted.climate_scale,
            "seasonal_scale": fitted.seasonal_scale,
            "history": fitted.history,
            "likelihood": fitted.likelihood,
            "outbreak_loss_weight": fitted.outbreak_loss_weight,
            "outbreak_positive_weight": fitted.outbreak_positive_weight,
            "outbreak_focal_gamma": fitted.outbreak_focal_gamma,
            "alert_probability_source": fitted.alert_probability_source,
        },
        path,
    )


def _fit_scale(values: np.ndarray, axes: tuple[int, ...]) -> Scale:
    mean = values.mean(axis=axes, keepdims=True)
    sd = values.std(axis=axes, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return Scale(mean=mean.astype(np.float32), sd=sd.astype(np.float32))


def _apply_scale(values: np.ndarray, scale: Scale) -> np.ndarray:
    return ((values - scale.mean) / scale.sd).astype(np.float32)


def _score(
    network,
    tensors,
    indices,
    horizon_weights,
    device,
    likelihood,
    peak_thresholds=None,
    peak_loss_weight=0.0,
    outbreak_loss_weight=0.0,
    outbreak_positive_weight=None,
    outbreak_focal_gamma=0.0,
    node_mask=None,
) -> float:
    if not len(indices):
        return float("nan")
    network.eval()
    with torch.no_grad():
        output = network(
            tensors[0][indices].to(device),
            tensors[1][indices].to(device),
            tensors[2][indices].to(device),
            return_outbreak_logits=outbreak_loss_weight > 0,
        )
        if outbreak_loss_weight > 0:
            location, scale, _, _, outbreak_logits = output
        else:
            location, scale = output
            outbreak_logits = None
        if node_mask is None:
            node_mask = slice(None)
        loss = _combined_loss(
            tensors[3][indices][:, node_mask].to(device),
            tensors[4][indices][:, node_mask].to(device),
            location[:, node_mask],
            scale[:, node_mask],
            outbreak_logits[:, node_mask] if outbreak_logits is not None else None,
            horizon_weights,
            likelihood=likelihood,
            peak_thresholds=(
                peak_thresholds.to(device) if peak_thresholds is not None else None
            ),
            peak_loss_weight=peak_loss_weight,
            outbreak_loss_weight=outbreak_loss_weight,
            outbreak_positive_weight=outbreak_positive_weight,
            outbreak_focal_gamma=outbreak_focal_gamma,
        )
    return float(loss.cpu())


def _combined_loss(
    target,
    outbreak_threshold,
    location,
    scale,
    outbreak_logits,
    horizon_weights,
    *,
    likelihood,
    peak_thresholds=None,
    peak_loss_weight=0.0,
    outbreak_loss_weight=0.0,
    outbreak_positive_weight=None,
    outbreak_focal_gamma=0.0,
):
    count_loss = _count_loss(
        target,
        location,
        scale,
        horizon_weights,
        likelihood=likelihood,
        peak_thresholds=peak_thresholds,
        peak_loss_weight=peak_loss_weight,
    )
    if outbreak_loss_weight <= 0 or outbreak_logits is None:
        return count_loss
    finite = torch.isfinite(outbreak_threshold)
    if not bool(finite.any()):
        return count_loss
    labels = (target > outbreak_threshold).to(target.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        outbreak_logits, labels, reduction="none"
    )
    if outbreak_positive_weight is not None:
        positive_weight = outbreak_positive_weight[None, None].expand_as(bce)
        bce = bce * torch.where(labels > 0, positive_weight, 1.0)
    if outbreak_focal_gamma > 0:
        probability = torch.sigmoid(outbreak_logits)
        correct_probability = torch.where(labels > 0, probability, 1.0 - probability)
        bce = bce * (1.0 - correct_probability).pow(outbreak_focal_gamma)
    weights = horizon_weights[None, None].expand_as(bce)
    weighted = (bce * weights)[finite]
    alert_loss = weighted.mean() if weighted.numel() else count_loss.new_tensor(0.0)
    return count_loss + outbreak_loss_weight * alert_loss


def _count_loss(
    target,
    location,
    scale,
    horizon_weights,
    *,
    likelihood,
    peak_thresholds=None,
    peak_loss_weight=0.0,
):
    observation_weights = None
    if peak_thresholds is not None and peak_loss_weight > 0:
        observation_weights = 1.0 + peak_loss_weight * (
            target > peak_thresholds[None, None]
        ).to(target.dtype)
    if likelihood == "lognormal":
        return lognormal_loss(
            target, location, scale, horizon_weights, observation_weights
        )
    if likelihood == "negative_binomial":
        return negative_binomial_loss(
            target, location, scale, horizon_weights, observation_weights
        )
    raise ValueError(f"Unsupported likelihood: {likelihood}")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
