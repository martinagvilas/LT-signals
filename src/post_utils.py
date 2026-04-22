from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch

from pre_utils import load_eureka_report


def merge_df(df, prefixes=("acc", "entropy", "logit_margin", "perplexity", "think_n_tokens")):
    """Melt a wide early-exit DataFrame (columns acc_0, acc_10, ...) to long format.

    Returns a DataFrame with one row per (data_point_id, data_repeat_id, percentage),
    with one column per prefix. Prefixes absent from df are silently skipped.
    """
    id_cols = ["data_point_id", "data_repeat_id", "ground_truth"]
    melted_dfs = []

    for prefix in prefixes:
        value_vars = [col for col in df.columns if col.startswith(f"{prefix}_")]
        if not value_vars:
            continue

        long_df = df.melt(
            id_vars=id_cols,
            value_vars=value_vars,
            var_name=f"{prefix}_position",
            value_name=prefix,
        )
        long_df["percentage"] = long_df[f"{prefix}_position"].str.extract(rf"{prefix}_(\d+)").astype(int)
        long_df = long_df.drop(columns=[f"{prefix}_position"])
        melted_dfs.append(long_df)

    from functools import reduce
    merged = reduce(
        lambda l, r: pd.merge(l, r, on=id_cols + ["percentage"], how="inner"),
        melted_dfs,
    )
    merged["percentage"] = merged["percentage"].replace({69: 70, 89: 90})
    return merged.sort_values(by=id_cols + ["percentage"])


def load_early_exit(model_name, dataset, path):
    file = path / f"results/early_exit/{dataset}/{model_name}_output_properties.json"
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return merge_df(pd.DataFrame(data))


def load_internal_metrics(model_name, dataset, path, average_type="ntokens_500"):
    """Load internal representation metrics produced by compute_internal_representations.py."""
    f = path / f"results/internals/{dataset}/{model_name}_{average_type}_internals.pt"
    return torch.load(f)


def create_internals_df(
    model_name,
    dataset,
    path,
    average_type="ntokens_500",
    remove_no_thinking=False,
    exit_last_n_layers=1,
    mean_layerwise=True,
):
    """Build analysis DataFrames from internal-representation metrics and early-exit results.

    Returns
    -------
    internal_results_df : DataFrame
        One row per (data point, repeat, metric, layer). Metrics:
          LT signals:       net_change, cumulative_change, aligned_change
          Output dist (internals): logit_margin_mean, entropy_mean
          Trajectory baselines:   layerwise_mag, layerwise_ang, seqwise_mag, seqwise_ang
    early_exit_results_df : DataFrame
        One row per (data point, repeat, metric). Metrics:
          Output dist (early exit): logit_margin, entropy, perplexity
          All evaluated at 100% thinking, averaged over the last exit_last_n_layers layers.
    """
    # Paper-reported LT signals and per-layer baselines
    layer_metrics = [
        "net_change",
        "cumulative_change",
        "aligned_change",
        "logit_margin_mean",
        "entropy_mean",
    ]
    # Cross-layer trajectory signals (averaged over layers/segments)
    trajectory_metrics = ["layerwise_mag", "layerwise_ang", "seqwise_mag", "seqwise_ang"]
    # Output-distribution baselines from the early exit script
    exit_metrics = ["logit_margin", "entropy", "perplexity"]

    early_exit_df = load_early_exit(model_name, dataset, path)
    internals = load_internal_metrics(model_name, dataset, path, average_type=average_type)
    try:
        report = load_eureka_report(model_name, dataset, path)
    except FileNotFoundError:
        report = None  # custom-JSON datasets (TSP non-phi4) have no Eureka report

    internal_results = []
    early_exit_results = []

    for i in internals:
        datapoint = i["data_point_id"]
        data_repeat = i["data_repeat_id"]

        if report is not None:
            report_entry = report.loc[
                (report["data_point_id"] == datapoint) & (report["data_repeat_id"] == data_repeat)
            ]
            try:
                output = report_entry["raw_model_output"].item()
            except Exception:
                output = report_entry["model_output"].item()
            if output is None:
                continue
            if remove_no_thinking and "</think>" not in output:
                continue

        d_exit = early_exit_df.loc[
            (early_exit_df["data_point_id"] == datapoint)
            & (early_exit_df["data_repeat_id"] == data_repeat)
        ]

        acc = d_exit.loc[d_exit["percentage"] == 100, "acc"].item()
        early_acc = d_exit.loc[d_exit["percentage"] == 0, "acc"].item()
        acc_sum = d_exit["acc"].sum()
        n_think_tokens = i["n_think_tokens"]

        base = {
            "datapoint": datapoint,
            "data_repeat": data_repeat,
            "accuracy": acc,
            "early_acc": early_acc,
            "acc_sum": acc_sum,
            "n_think_tokens": n_think_tokens,
        }

        # Per-layer LT signals and output-distribution internals
        for metric in layer_metrics:
            for layer_idx, val in enumerate(i[metric]):
                internal_results.append({**base, "metric_name": metric, "layer": layer_idx, "value": val})

        # Trajectory signals (scalar or array, depending on mean_layerwise)
        for metric in trajectory_metrics:
            val = torch.mean(i[metric]).item() if mean_layerwise else i[metric].cpu().numpy()
            internal_results.append({**base, "metric_name": metric, "layer": "all", "value": val})

        # Output-distribution baselines from early exit at 100% thinking
        for metric in exit_metrics:
            if metric not in d_exit.columns:
                continue
            metric_data = np.mean(
                np.stack(d_exit[metric].values)[:, -exit_last_n_layers:], axis=-1
            )[-1].item()
            early_exit_results.append({**base, "metric_name": metric, "value": metric_data, "layer": -exit_last_n_layers})

    return pd.DataFrame(internal_results), pd.DataFrame(early_exit_results)
