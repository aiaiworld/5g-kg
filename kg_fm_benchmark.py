#!/usr/bin/env python3
"""KG-augmented Fault Management benchmark.

This script intentionally uses only the Python standard library. It can run a
smoke benchmark with the built-in fixture, or it can load the open DejaVu A1
dataset after the user places A1.zip or an extracted A1 directory locally.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics
import time
import urllib.request
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_A1_URL = "https://zenodo.org/records/6955909/files/A1.zip?download=1"


@dataclass
class Fault:
    timestamp: int
    roots: list[str]
    fault_type: str = ""


@dataclass
class Dataset:
    name: str
    metric_values: dict[str, dict[int, float]]
    faults: list[Fault]
    node_metrics: dict[str, set[str]]
    edges: dict[str, set[str]]
    node_types: dict[str, str] = field(default_factory=dict)

    @property
    def timestamps(self) -> list[int]:
        values: set[int] = set()
        for series in self.metric_values.values():
            values.update(series.keys())
        return sorted(values)

    @property
    def nodes(self) -> set[str]:
        out = set(self.node_metrics)
        out.update(self.edges)
        for dsts in self.edges.values():
            out.update(dsts)
        for fault in self.faults:
            out.update(fault.roots)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KG-augmented KPI prediction, anomaly detection, and RCA."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to an extracted DejaVu A1 directory, or to A1.zip.",
    )
    parser.add_argument(
        "--download-a1",
        action="store_true",
        help="Download DejaVu A1.zip into data/open if the network allows it.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_A1_URL,
        help="Open dataset URL used with --download-a1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/kg_fm/results"),
        help="Directory for results.json, results.md, and kg_triples.nt.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit metric rows for a quick pass. 0 means read all rows.",
    )
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--fault-window-steps", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--kg-alpha", type=float, default=0.2)
    parser.add_argument("--rca-propagation-weight", type=float, default=0.5)
    parser.add_argument("--rca-max-depth", type=int, default=3)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--min-threshold", type=float, default=3.0)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Force the built-in tiny fixture even when --data-dir is supplied.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    if args.download_a1:
        data_dir = download_a1(args.download_url, Path("data/open"))

    if args.fixture or data_dir is None:
        dataset = make_fixture()
    else:
        dataset = load_dataset(data_dir, args.output_dir, args.max_rows)

    summary, triples = run_benchmark(dataset, args)
    summary["runtime_sec"] = round(time.perf_counter() - started, 3)
    summary["dataset"] = {
        "name": dataset.name,
        "metrics": len(dataset.metric_values),
        "nodes": len(dataset.nodes),
        "edges": sum(len(v) for v in dataset.edges.values()),
        "faults": len(dataset.faults),
        "timestamps": len(dataset.timestamps),
    }

    write_outputs(summary, triples, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print(render_markdown(summary))
    return 0


def download_a1(url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "A1.zip"
    if not zip_path.exists():
        print(f"Downloading {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def load_dataset(data_path: Path, output_dir: Path, max_rows: int) -> Dataset:
    data_dir = resolve_data_dir(data_path, output_dir)
    metrics_path = find_file(data_dir, ["metrics.norm.csv", "metrics.csv"])
    faults_path = find_file(data_dir, ["faults.csv"])
    if metrics_path is None:
        raise FileNotFoundError(f"Cannot find metrics.csv under {data_dir}")
    if faults_path is None:
        raise FileNotFoundError(f"Cannot find faults.csv under {data_dir}")

    metric_values = read_metrics(metrics_path, max_rows=max_rows)
    faults = read_faults(faults_path)
    graph_path = find_file(data_dir, ["graph.yml", "graph.yaml", "graph.csv"])
    node_metrics, edges, node_types = load_graph(graph_path, metric_values)
    add_metric_nodes(node_metrics, metric_values)
    add_fault_nodes(node_metrics, faults)
    return Dataset(
        name=data_dir.name,
        metric_values=metric_values,
        faults=faults,
        node_metrics=node_metrics,
        edges=edges,
        node_types=node_types,
    )


def resolve_data_dir(data_path: Path, output_dir: Path) -> Path:
    data_path = data_path.resolve()
    if data_path.is_dir():
        return select_dataset_dir(data_path)
    if data_path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"{data_path} is neither a directory nor a zip file")
    extract_root = output_dir / "_extracted" / data_path.stem
    marker = extract_root / ".complete"
    if not marker.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(extract_root)
        marker.write_text("ok\n", encoding="utf-8")
    return select_dataset_dir(extract_root)


def select_dataset_dir(root: Path) -> Path:
    if (root / "metrics.csv").exists() or (root / "metrics.norm.csv").exists():
        return root
    candidates: list[Path] = []
    for path, _dirs, files in os.walk(root):
        file_set = set(files)
        if "metrics.csv" in file_set or "metrics.norm.csv" in file_set:
            candidates.append(Path(path))
    if not candidates:
        return root
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def find_file(root: Path, names: list[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    wanted = set(names)
    for path, _dirs, files in os.walk(root):
        for filename in files:
            if filename in wanted:
                return Path(path) / filename
    return None


def read_metrics(path: Path, max_rows: int = 0) -> dict[str, dict[int, float]]:
    metric_values: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if max_rows and index >= max_rows:
                break
            timestamp = parse_int(first_present(row, ["timestamp", "time", "ts"]))
            name = first_present(row, ["name", "metric", "kpi_name", "metric_name"])
            value_raw = first_present(row, ["value", "val", "metric_value", "kpi_value"])
            if timestamp is None or not name or value_raw is None:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue
            metric_values[name][timestamp] = value
    if not metric_values:
        raise ValueError(f"No metric rows loaded from {path}")
    return dict(metric_values)


def read_faults(path: Path) -> list[Fault]:
    faults: list[Fault] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = parse_int(first_present(row, ["timestamp", "time", "ts"]))
            root_raw = first_present(
                row,
                [
                    "root_cause_node",
                    "root_cause",
                    "root_cause_instance",
                    "fault_loc",
                    "node",
                ],
            )
            if timestamp is None or not root_raw:
                continue
            roots = [part.strip() for part in str(root_raw).split(";") if part.strip()]
            fault_type = first_present(row, ["fault_type", "type", "failure_type"]) or ""
            faults.append(Fault(timestamp=timestamp, roots=roots, fault_type=str(fault_type)))
    if not faults:
        raise ValueError(f"No faults loaded from {path}")
    return sorted(faults, key=lambda f: f.timestamp)


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
        if key.lower() in lower and lower[key.lower()] not in ("", None):
            return lower[key.lower()]
    return None


def parse_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def load_graph(
    graph_path: Path | None,
    metric_values: dict[str, dict[int, float]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    if graph_path is None:
        node_metrics: dict[str, set[str]] = defaultdict(set)
        add_metric_nodes(node_metrics, metric_values)
        return dict(node_metrics), {}, {}
    if graph_path.suffix.lower() == ".csv":
        return load_graph_csv(graph_path)
    try:
        return load_graph_yaml_subset(graph_path)
    except Exception as exc:
        print(f"Warning: failed to parse {graph_path}: {exc}. Inferring nodes from metrics.")
        node_metrics = defaultdict(set)
        add_metric_nodes(node_metrics, metric_values)
        return dict(node_metrics), {}, {}


def load_graph_csv(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    node_metrics: dict[str, set[str]] = defaultdict(set)
    edges: dict[str, set[str]] = defaultdict(set)
    node_types: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = first_present(row, ["src", "source", "from"])
            dst = first_present(row, ["dst", "target", "to"])
            node = first_present(row, ["node", "id"])
            metric = first_present(row, ["metric", "name"])
            node_type = first_present(row, ["type", "node_type"])
            if src and dst:
                edges[str(src)].add(str(dst))
            if node:
                if metric:
                    node_metrics[str(node)].add(str(metric))
                if node_type:
                    node_types[str(node)] = str(node_type)
    return dict(node_metrics), dict(edges), node_types


def load_graph_yaml_subset(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    objects = parse_yaml_subset(path)
    global_params: dict[str, Any] = {}
    for obj in objects:
        if obj.get("class") == "global_params":
            global_params = {k: v for k, v in obj.items() if k != "class"}
            break

    node_metrics: dict[str, set[str]] = defaultdict(set)
    node_types: dict[str, str] = {}
    edges: dict[str, set[str]] = defaultdict(set)

    for obj in objects:
        if obj.get("class") != "node":
            continue
        for params in expanded_params(obj, global_params):
            node = format_template(str(obj["id"]), params)
            metrics = [format_template(str(metric), params) for metric in obj.get("metrics", [])]
            node_metrics[node].update(metrics)
            node_types[node] = str(obj.get("type", "FailureUnit"))

    for obj in objects:
        if obj.get("class") != "edge":
            continue
        for params in expanded_params(obj, global_params):
            src = format_template(str(obj["src"]), params)
            dst = format_template(str(obj["dst"]), params)
            if src in node_metrics and dst in node_metrics:
                edges[src].add(dst)

    return dict(node_metrics), dict(edges), node_types


def parse_yaml_subset(path: Path) -> list[dict[str, Any]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if stripped.startswith("- ") and len(line) == len(stripped):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return [parse_yaml_object(block) for block in blocks]


def parse_yaml_object(lines: list[str]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    object_indent = 0
    for raw in lines[1:]:
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if ":" in stripped and not stripped.startswith("- "):
            object_indent = indent
            break
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if index == 0 and stripped.startswith("- "):
            stripped = stripped[2:].strip()
            indent = 0
        expected_indent = 0 if index == 0 else object_indent
        if indent != expected_indent or ":" not in stripped:
            index += 1
            continue
        key, value = split_key_value(stripped)
        if value != "":
            obj[key] = parse_yaml_value(value)
            index += 1
            continue
        if key == "params":
            parsed, index = parse_yaml_mapping_section(lines, index + 1, indent)
            obj[key] = parsed
        elif key in {"metrics", "global_params"}:
            parsed_list, index = parse_yaml_list_section(lines, index + 1, indent)
            obj[key] = parsed_list
        else:
            parsed_map, index = parse_yaml_mapping_section(lines, index + 1, indent)
            obj[key] = parsed_map
    return obj


def parse_yaml_mapping_section(
    lines: list[str],
    index: int,
    parent_indent: int,
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(lines):
        raw = lines[index]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if indent <= parent_indent:
            break
        if stripped.startswith("- ") or ":" not in stripped:
            index += 1
            continue
        key, value = split_key_value(stripped)
        if value != "":
            out[key] = parse_yaml_value(value)
            index += 1
            continue
        values, index = parse_yaml_list_section(lines, index + 1, indent)
        out[key] = values
    return out, index


def parse_yaml_list_section(
    lines: list[str],
    index: int,
    parent_indent: int,
) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(lines):
        raw = lines[index]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if indent <= parent_indent:
            break
        if stripped.startswith("- "):
            out.append(parse_yaml_value(stripped[2:].strip()))
        index += 1
    return out, index


def split_key_value(text: str) -> tuple[str, str]:
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def parse_yaml_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [strip_quotes(part.strip()) for part in inner.split(",")]
    return strip_quotes(value)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def expanded_params(obj: dict[str, Any], global_params: dict[str, Any]) -> list[dict[str, Any]]:
    params = dict(obj.get("params") or {})
    for key in obj.get("global_params", []) or []:
        if key in global_params:
            params[key] = global_params[key]
    if not params:
        return [{}]
    keys = list(params)
    value_lists = [ensure_list(params[key]) for key in keys]
    if obj.get("product", False):
        return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]
    max_len = max(len(values) for values in value_lists)
    expanded: list[dict[str, Any]] = []
    for index in range(max_len):
        row = {}
        for key, values in zip(keys, value_lists):
            row[key] = values[index if len(values) > 1 else 0]
        expanded.append(row)
    return expanded


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def format_template(template: str, params: dict[str, Any]) -> str:
    try:
        return template.format(**params)
    except Exception:
        return template


def add_metric_nodes(
    node_metrics: dict[str, set[str]],
    metric_values: dict[str, dict[int, float]],
) -> None:
    mapped_metrics = set().union(*node_metrics.values()) if node_metrics else set()
    for metric in metric_values:
        if metric in mapped_metrics:
            continue
        node, _metric_type = split_metric_name(metric)
        node_metrics.setdefault(node, set()).add(metric)


def add_fault_nodes(node_metrics: dict[str, set[str]], faults: list[Fault]) -> None:
    for fault in faults:
        for root in fault.roots:
            node_metrics.setdefault(root, set())


def split_metric_name(name: str) -> tuple[str, str]:
    if "##" in name:
        node, metric_type = name.split("##", 1)
        return node, metric_type
    for sep in ("::", "/", "."):
        if sep in name:
            node, metric_type = name.rsplit(sep, 1)
            return node, metric_type
    return name, name


def make_fixture() -> Dataset:
    nodes = ["gnb1", "amf1", "smf1", "upf1", "nrf1"]
    metric_types = ["latency_ms", "error_rate", "cpu"]
    timestamps = [i * 60 for i in range(30)]
    metric_values: dict[str, dict[int, float]] = {}

    for node_index, node in enumerate(nodes):
        for metric_type in metric_types:
            series: dict[int, float] = {}
            for i, ts in enumerate(timestamps):
                base = 10.0 + node_index * 1.3 + math.sin(i / 3.0)
                if metric_type == "error_rate":
                    base = 0.01 + node_index * 0.002 + 0.001 * math.sin(i / 4.0)
                elif metric_type == "cpu":
                    base = 40.0 + node_index * 4.0 + 2.0 * math.sin(i / 5.0)

                if node == "amf1" and 15 <= i <= 17:
                    base += {"latency_ms": 18.0, "error_rate": 0.12, "cpu": 28.0}[metric_type]
                if node in {"smf1", "upf1"} and 16 <= i <= 18:
                    base += {"latency_ms": 14.0, "error_rate": 0.08, "cpu": 18.0}[metric_type]
                if node == "gnb1" and 17 <= i <= 18:
                    base += {"latency_ms": 5.0, "error_rate": 0.02, "cpu": 5.0}[metric_type]
                series[ts] = round(base, 6)
            metric_values[f"{node}##{metric_type}"] = series

    node_metrics: dict[str, set[str]] = defaultdict(set)
    add_metric_nodes(node_metrics, metric_values)
    edges = {
        "gnb1": {"amf1"},
        "amf1": {"smf1"},
        "smf1": {"upf1"},
        "nrf1": {"amf1"},
    }
    node_types = {
        "gnb1": "RAN",
        "amf1": "5GC_NF",
        "smf1": "5GC_NF",
        "upf1": "5GC_NF",
        "nrf1": "5GC_NF",
    }
    faults = [Fault(timestamp=15 * 60, roots=["amf1"], fault_type="amf_latency_storm")]
    return Dataset("fixture_tiny_5gc", metric_values, faults, dict(node_metrics), edges, node_types)


def run_benchmark(dataset: Dataset, args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    timestamps = dataset.timestamps
    labels = build_fault_labels(timestamps, dataset.faults, args.granularity, args.fault_window_steps)
    train_ts = choose_train_timestamps(timestamps, labels, dataset.faults, args.train_ratio)
    eval_ts = [ts for ts in timestamps if ts not in train_ts]
    stats = compute_metric_stats(dataset.metric_values, train_ts)
    neighbors = undirected_neighbors(dataset.edges)

    residuals = compute_residuals(
        dataset=dataset,
        stats=stats,
        neighbors=neighbors,
        eval_ts=set(eval_ts),
        kg_alpha=args.kg_alpha,
    )
    anomaly = evaluate_anomaly_detection(
        residuals=residuals,
        labels=labels,
        train_ts=train_ts,
        eval_ts=eval_ts,
        quantile=args.threshold_quantile,
        min_threshold=args.min_threshold,
    )
    prediction = evaluate_prediction(residuals)
    rca = evaluate_rca(
        dataset=dataset,
        residuals=residuals,
        args=args,
    )
    triples = build_kg_triples(dataset)
    summary: dict[str, Any] = {
        "train": {
            "timestamps": len(train_ts),
            "eval_timestamps": len(eval_ts),
            "fault_window_steps": args.fault_window_steps,
        },
        "prediction": prediction,
        "anomaly_detection": anomaly,
        "root_cause_analysis": rca,
        "methods": {
            "baseline": "Last-value forecast + robust residual + local node residual RCA",
            "kg": "Proposed: KG neighbor-context forecast + robust residual + topology propagation RCA",
        },
        "hyperparameters": {
            "kg_alpha": args.kg_alpha,
            "rca_propagation_weight": args.rca_propagation_weight,
            "rca_max_depth": args.rca_max_depth,
            "threshold_quantile": args.threshold_quantile,
            "min_threshold": args.min_threshold,
        },
    }
    return summary, triples


def build_fault_labels(
    timestamps: list[int],
    faults: list[Fault],
    granularity: int,
    fault_window_steps: int,
) -> dict[int, int]:
    timestamp_set = set(timestamps)
    labels = {ts: 0 for ts in timestamps}
    for fault in faults:
        for step in range(fault_window_steps + 1):
            ts = fault.timestamp + step * granularity
            if ts in timestamp_set:
                labels[ts] = 1
    return labels


def choose_train_timestamps(
    timestamps: list[int],
    labels: dict[int, int],
    faults: list[Fault],
    train_ratio: float,
) -> set[int]:
    if not timestamps:
        return set()
    min_fault = min((fault.timestamp for fault in faults), default=None)
    if min_fault is not None:
        before_fault = [ts for ts in timestamps if ts < min_fault and labels.get(ts, 0) == 0]
        if len(before_fault) >= max(5, int(0.2 * len(timestamps))):
            return set(before_fault)
    cutoff = max(1, int(len(timestamps) * train_ratio))
    return {ts for ts in timestamps[:cutoff] if labels.get(ts, 0) == 0}


def compute_metric_stats(
    metric_values: dict[str, dict[int, float]],
    train_ts: set[int],
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for metric, values in metric_values.items():
        train_values = [v for ts, v in values.items() if ts in train_ts and is_finite(v)]
        if len(train_values) < 3:
            train_values = [v for v in values.values() if is_finite(v)]
        center = median(train_values) if train_values else 0.0
        deviations = [abs(v - center) for v in train_values]
        scale = 1.4826 * median(deviations) if deviations else 0.0
        if scale <= 1e-12 and len(train_values) >= 2:
            scale = statistics.pstdev(train_values)
        if scale <= 1e-12:
            scale = 1.0
        stats[metric] = (center, scale)
    return stats


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def undirected_neighbors(edges: dict[str, set[str]]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for src, dsts in edges.items():
        for dst in dsts:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return dict(neighbors)


def compute_residuals(
    dataset: Dataset,
    stats: dict[str, tuple[float, float]],
    neighbors: dict[str, set[str]],
    eval_ts: set[int],
    kg_alpha: float,
) -> dict[str, Any]:
    metric_to_nodes = build_metric_owner_map(dataset)
    node_metric_by_type = index_metrics_by_node_type(dataset.metric_values, metric_to_nodes)
    result = {
        "timestamp_scores": {"baseline": defaultdict(float), "kg": defaultdict(float)},
        "node_scores": {"baseline": defaultdict(lambda: defaultdict(float)), "kg": defaultdict(lambda: defaultdict(float))},
        "errors": {"baseline": {"abs": [], "sq": []}, "kg": {"abs": [], "sq": []}},
    }

    for metric, values_by_ts in dataset.metric_values.items():
        ordered = sorted(values_by_ts.items())
        if len(ordered) < 2:
            continue
        owner_nodes = metric_to_nodes.get(metric, [split_metric_name(metric)[0]])
        node = choose_primary_owner(metric, owner_nodes)
        _node, metric_type = split_metric_name(metric)
        center, scale = stats[metric]
        for index in range(1, len(ordered)):
            prev_ts, prev_value = ordered[index - 1]
            ts, actual = ordered[index]
            baseline_pred = prev_value
            neighbor_z = neighbor_context_z(
                node=node,
                metric_type=metric_type,
                prev_ts=prev_ts,
                metric_values=dataset.metric_values,
                stats=stats,
                neighbors=neighbors,
                node_metric_by_type=node_metric_by_type,
            )
            own_prev_z = (prev_value - center) / scale
            if neighbor_z is None:
                kg_pred = baseline_pred
            else:
                kg_z = (1.0 - kg_alpha) * own_prev_z + kg_alpha * neighbor_z
                kg_pred = center + scale * kg_z

            for method, prediction in (("baseline", baseline_pred), ("kg", kg_pred)):
                abs_error = abs(actual - prediction)
                score = abs_error / scale
                result["timestamp_scores"][method][ts] = max(
                    result["timestamp_scores"][method][ts], score
                )
                for owner in owner_nodes:
                    result["node_scores"][method][ts][owner] = max(
                        result["node_scores"][method][ts][owner], score
                    )
                if ts in eval_ts:
                    result["errors"][method]["abs"].append(abs_error)
                    result["errors"][method]["sq"].append(abs_error * abs_error)
    return result


def build_metric_owner_map(dataset: Dataset) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for node, metrics in dataset.node_metrics.items():
        for metric in metrics:
            if metric in dataset.metric_values:
                owners[metric].append(node)
    for metric in dataset.metric_values:
        if metric not in owners:
            owners[metric].append(split_metric_name(metric)[0])
    return {metric: sorted(nodes) for metric, nodes in owners.items()}


def choose_primary_owner(metric: str, owner_nodes: list[str]) -> str:
    base_node, _metric_type = split_metric_name(metric)
    return sorted(owner_nodes, key=lambda node: (node == base_node, -len(node), node))[0]


def index_metrics_by_node_type(
    metric_values: dict[str, dict[int, float]],
    metric_to_nodes: dict[str, list[str]],
) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for metric in metric_values:
        _node, metric_type = split_metric_name(metric)
        for node in metric_to_nodes.get(metric, [split_metric_name(metric)[0]]):
            out[(node, metric_type)] = metric
    return out


def neighbor_context_z(
    node: str,
    metric_type: str,
    prev_ts: int,
    metric_values: dict[str, dict[int, float]],
    stats: dict[str, tuple[float, float]],
    neighbors: dict[str, set[str]],
    node_metric_by_type: dict[tuple[str, str], str],
) -> float | None:
    z_values: list[float] = []
    for neighbor in neighbors.get(node, set()):
        metric = node_metric_by_type.get((neighbor, metric_type))
        if metric is not None and prev_ts in metric_values.get(metric, {}):
            center, scale = stats[metric]
            z_values.append((metric_values[metric][prev_ts] - center) / scale)
    if not z_values:
        return None
    return sum(z_values) / len(z_values)


def evaluate_prediction(residuals: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method, errors in residuals["errors"].items():
        abs_errors = errors["abs"]
        sq_errors = errors["sq"]
        out[method] = {
            "mae": round(sum(abs_errors) / len(abs_errors), 6) if abs_errors else 0.0,
            "rmse": round(math.sqrt(sum(sq_errors) / len(sq_errors)), 6) if sq_errors else 0.0,
            "n": len(abs_errors),
        }
    return out


def evaluate_anomaly_detection(
    residuals: dict[str, Any],
    labels: dict[int, int],
    train_ts: set[int],
    eval_ts: list[int],
    quantile: float,
    min_threshold: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method, scores_by_ts in residuals["timestamp_scores"].items():
        train_scores = [scores_by_ts.get(ts, 0.0) for ts in train_ts]
        threshold = max(min_threshold, percentile(train_scores, quantile))
        tp = fp = fn = tn = 0
        for ts in eval_ts:
            pred = int(scores_by_ts.get(ts, 0.0) >= threshold)
            label = labels.get(ts, 0)
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[method] = {
            "threshold": round(threshold, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return out


def percentile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if is_finite(v))
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - pos) + values[upper] * (pos - lower)


def evaluate_rca(
    dataset: Dataset,
    residuals: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    methods = ["baseline", "kg"]
    ranks_by_method: dict[str, list[int]] = {method: [] for method in methods}
    all_nodes = sorted(dataset.nodes)

    for fault in dataset.faults:
        window_ts = [
            fault.timestamp + step * args.granularity
            for step in range(args.fault_window_steps + 1)
        ]
        for method in methods:
            local_scores = aggregate_node_scores(residuals["node_scores"][method], window_ts, all_nodes)
            if method == "kg":
                scores = topology_propagation_scores(
                    local_scores=local_scores,
                    edges=dataset.edges,
                    weight=args.rca_propagation_weight,
                    max_depth=args.rca_max_depth,
                )
            else:
                scores = local_scores
            rank = best_root_rank(scores, fault.roots, all_nodes)
            ranks_by_method[method].append(rank)

    out: dict[str, dict[str, float]] = {}
    for method, ranks in ranks_by_method.items():
        out[method] = {
            "hit_at_1": round(sum(1 for r in ranks if r <= 1) / len(ranks), 6) if ranks else 0.0,
            "hit_at_3": round(sum(1 for r in ranks if r <= 3) / len(ranks), 6) if ranks else 0.0,
            "hit_at_5": round(sum(1 for r in ranks if r <= 5) / len(ranks), 6) if ranks else 0.0,
            "mrr": round(sum(1.0 / r for r in ranks) / len(ranks), 6) if ranks else 0.0,
            "mean_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
            "n_faults": len(ranks),
        }
    return out


def aggregate_node_scores(
    node_scores_by_ts: dict[int, dict[str, float]],
    window_ts: list[int],
    all_nodes: list[str],
) -> dict[str, float]:
    out = {node: 0.0 for node in all_nodes}
    for ts in window_ts:
        for node, score in node_scores_by_ts.get(ts, {}).items():
            out[node] = max(out.get(node, 0.0), score)
    return out


def topology_propagation_scores(
    local_scores: dict[str, float],
    edges: dict[str, set[str]],
    weight: float,
    max_depth: int,
) -> dict[str, float]:
    scores = dict(local_scores)
    for node in local_scores:
        for downstream, depth in downstream_nodes(node, edges, max_depth):
            scores[node] += weight * local_scores.get(downstream, 0.0) / (depth + 1)
    return scores


def downstream_nodes(
    start: str,
    edges: dict[str, set[str]],
    max_depth: int,
) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    seen = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for dst in edges.get(node, set()):
            if dst in seen:
                continue
            seen.add(dst)
            found.append((dst, depth + 1))
            queue.append((dst, depth + 1))
    return found


def best_root_rank(scores: dict[str, float], roots: list[str], all_nodes: list[str]) -> int:
    ranked = sorted(all_nodes, key=lambda node: (-scores.get(node, 0.0), node))
    root_set = set(roots)
    for index, node in enumerate(ranked, start=1):
        if node in root_set:
            return index
    return len(ranked) + 1


def build_kg_triples(dataset: Dataset) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for node in sorted(dataset.nodes):
        node_id = iri("fi", node)
        node_type = dataset.node_types.get(node, "FailureUnit")
        triples.append((node_id, iri("rdf", "type"), iri("fm", node_type)))
        for metric in sorted(dataset.node_metrics.get(node, set())):
            metric_id = iri("metric", metric)
            _node, metric_type = split_metric_name(metric)
            triples.append((node_id, iri("fm", "hasMetric"), metric_id))
            triples.append((metric_id, iri("fm", "metricType"), literal(metric_type)))
    for src, dsts in dataset.edges.items():
        for dst in sorted(dsts):
            triples.append((iri("fi", src), iri("fm", "dependsOn"), iri("fi", dst)))
    for fault in dataset.faults:
        fault_id = iri("fault", str(fault.timestamp))
        triples.append((fault_id, iri("rdf", "type"), iri("fm", "FaultEvent")))
        for root in fault.roots:
            triples.append((fault_id, iri("fm", "hasRootCause"), iri("fi", root)))
    return triples


def iri(namespace: str, value: str) -> str:
    safe = str(value).replace(" ", "_").replace("<", "").replace(">", "")
    return f"<urn:{namespace}:{safe}>"


def literal(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_outputs(summary: dict[str, Any], triples: list[tuple[str, str, str]], output_dir: Path) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.md").write_text(render_markdown(summary), encoding="utf-8")
    with (output_dir / "kg_triples.nt").open("w", encoding="utf-8") as handle:
        for subject, predicate, obj in triples:
            handle.write(f"{subject} {predicate} {obj} .\n")


def render_markdown(summary: dict[str, Any]) -> str:
    dataset = summary.get("dataset", {})
    def best_pair(base: float, kg: float, higher_is_better: bool) -> tuple[str, str]:
        base_best = base >= kg if higher_is_better else base <= kg
        kg_best = kg >= base if higher_is_better else kg <= base
        base_text = f"{base:.6f}"
        kg_text = f"{kg:.6f}"
        return (
            f"**{base_text}**" if base_best else base_text,
            f"**{kg_text}**" if kg_best else kg_text,
        )

    rows = [
        "# KG-FM Benchmark Results",
        "",
        f"Dataset: `{dataset.get('name', 'unknown')}`",
        "",
        "| Task | Metric | Baseline | Proposed | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    prediction = summary["prediction"]
    for metric in ["mae", "rmse"]:
        base = prediction["baseline"][metric]
        kg = prediction["kg"][metric]
        base_cell, kg_cell = best_pair(base, kg, higher_is_better=False)
        rows.append(f"| KPI prediction | {metric.upper()} | {base_cell} | {kg_cell} | {kg - base:.6f} |")
    anomaly = summary["anomaly_detection"]
    for metric in ["precision", "recall", "f1"]:
        base = anomaly["baseline"][metric]
        kg = anomaly["kg"][metric]
        base_cell, kg_cell = best_pair(base, kg, higher_is_better=True)
        rows.append(f"| Anomaly detection | {metric} | {base_cell} | {kg_cell} | {kg - base:.6f} |")
    rca = summary["root_cause_analysis"]
    for metric in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_rank"]:
        base = rca["baseline"][metric]
        kg = rca["kg"][metric]
        base_cell, kg_cell = best_pair(base, kg, higher_is_better=(metric != "mean_rank"))
        rows.append(f"| RCA | {metric} | {base_cell} | {kg_cell} | {kg - base:.6f} |")
    rows.append("")
    rows.append("```json")
    rows.append(json.dumps(dataset, indent=2, sort_keys=True))
    rows.append("```")
    rows.append("")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
