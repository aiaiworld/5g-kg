# KG-FM Benchmark

This benchmark evaluates one lightweight idea for Fault Management in an Autonomous Network:
use a Knowledge Graph as the shared context for KPI prediction, anomaly detection, and
root-cause analysis.

The script has no third-party Python dependency. It can run immediately with a built-in
tiny fixture. For paper results, run it on the open DejaVu A1 dataset.

## Open Dataset

Primary dataset: DejaVu A1, published with "Actionable and Interpretable Fault
Localization for Recurring Failures in Online Service Systems".

Expected files after extraction:

- `graph.yml`: failure dependency graph
- `metrics.csv` or `metrics.norm.csv`: long-form KPI table with `timestamp,name,value`
- `faults.csv`: fault events with `timestamp,root_cause_node`

The loader uses `metrics.norm.csv` when available, then falls back to
`metrics.csv`.

Official dataset page:

```text
https://zenodo.org/records/6955909
```

Direct A1 file:

```text
https://zenodo.org/records/6955909/files/A1.zip?download=1
```

Place `A1.zip` under `data/open/A1.zip`.

## Run

Smoke test:

```powershell
python benchmarks/kg_fm/kg_fm_benchmark.py --fixture
```

Run with an extracted A1 directory:

```powershell
python benchmarks/kg_fm/kg_fm_benchmark.py --data-dir data/open/A1
```

Run with A1 zip:

```powershell
python benchmarks/kg_fm/kg_fm_benchmark.py --data-dir data/open/A1.zip
```

Try the built-in downloader if the network allows Zenodo:

```powershell
python benchmarks/kg_fm/kg_fm_benchmark.py --download-a1
```

Quick pass on a large file:

```powershell
python benchmarks/kg_fm/kg_fm_benchmark.py --data-dir data/open/A1.zip --max-rows 500000
```

## Outputs

The script writes:

- `benchmarks/kg_fm/results/results.json`
- `benchmarks/kg_fm/results/results.md`
- `benchmarks/kg_fm/results/kg_triples.nt`

`results.md` is formatted as a compact benchmark table for the poster.

## Methods

Baseline:

- KPI prediction: last-value one-step forecast.
- Anomaly detection: robust residual score using train-set median and MAD.
- RCA: rank nodes by maximum local residual score in the fault window.

Proposed KG-augmented method:

- KPI prediction: last-value forecast adjusted by normalized previous-step neighbor KPI
  context from the dependency graph.
- Anomaly detection: robust residual score from the KG-context forecast.
- RCA: topology-aware propagation score. A candidate root gets local residual evidence
  plus attenuated downstream anomaly evidence reachable in the KG.

Default hyperparameters used for the poster table:

- `kg_alpha=0.2`
- `rca_propagation_weight=0.5`
- `rca_max_depth=3`

## Metrics

- KPI prediction: MAE, RMSE.
- Anomaly detection: precision, recall, F1 on fault windows.
- RCA: Hit@1, Hit@3, Hit@5, MRR, mean rank.

Use `results.md` directly as the result table after running the full DejaVu A1 benchmark.
