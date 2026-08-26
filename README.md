# DEST — Deterministic Collatz Sampling for Reproducible Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](#citing)
[![Results](https://img.shields.io/badge/results-176%20runs-brightgreen.svg)](#results)

**Deterministic samplers based on Collatz dynamics that match or beat stochastic training in accuracy while cutting seed-to-seed variance by up to 80%.**

> Same seed → same weights, same split, same dropout. The *only* difference between methods is the order in which training samples are presented.

---

## Abstract

Stochastic data ordering is the default in deep learning but introduces run-to-run variance that harms reproducibility. DEST (*Deterministic Sampling for Enhanced Training*) replaces random permutations with deterministic orders derived from Collatz dynamics. We evaluate five samplers — `stochastic`, `sobol`, `collatz_v1/v2/v3` — on CIFAR-10 (n=20 seeds/method, 100 runs) and CIFAR-100 (n=10, 40 runs) under identical conditions (15 epochs, SGD+cosine, deterministic dropout). On CIFAR-10, `collatz_v3` achieves **85.87% ± 0.83**, significantly outperforming stochastic (+0.76 pp, *p*≈0.006, *d*=0.87, paired *t*=3.47, wins 15/20). On CIFAR-100 it matches stochastic accuracy while reducing variance by **≈80%** (σ 0.19 vs 0.43). The sampler overhead is ≈4% of training time.

---

## Repository Structure

```
DEST/
├── src/dest/            # installable package (config, samplers, runner, metrics, …)
├── notebooks/            # Colab-ready notebooks (all standalone, no repo checkout needed)
├── scripts/              # generators & analysis (generate_*_nb.py, analyze_paper_results.py)
├── results/
│   ├── paper/            # 140 JSON RunResults — CIFAR-10/100 paper experiments
│   └── debug/            # 36 JSON — MNIST sanity checks
├── docs/
│   └── resumen_operativo_DEST.md  # long operational report (Spanish)
├── tests/
├── data/                 # gitignored — CIFAR downloads land here
├── pyproject.toml
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/starlyn2010/DEST.git
cd DEST
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### Reproduce a Single Seed (local)

```python
from dest.config import get_config
from dest.runner import ExperimentRunner

config = get_config("PAPER")           # 15 epochs, batch 128, cosine
config["output_dir"] = "./results/paper"
runner = ExperimentRunner(config)

res = runner.run_single_seed(
    exp_id="CIFAR10_collatz_v3",
    sampler_name="collatz_v3",
    seed=42,
    dataset="CIFAR10",
    dropout_mode="deterministic",
)
print(f"acc={res.final_test_acc:.2f}%  best={res.best_test_acc:.2f}%")
```

### Colab — Complete Remaining Seeds

All notebooks in `notebooks/` are **standalone**: they embed `src/dest/` as base64 and need no `git clone`.

| Notebook | Purpose | Runs |
|---|---|---|
| `CIFAR100_Final_6_Seeds_standalone.ipynb` | the 6-seed patch (46–51) | 6 |
| `DEST_Todas_Semillas_Faltantes.ipynb` | paper gaps (CIFAR100 v3 42–45, CIFAR10 v3/v2, …) | 22 |
| `DEST_CIFAR100_Pareado_Semillas200.ipynb` | **clean paired study — 10 seeds 200–209 × 5 methods, seed-major, resume-safe** | 50 |

> **Reproducibility guarantee:** `seed_everything(seed)` + fixed `split_seed=0` + local RNGs in samplers ⇒ same numeric seed gives bit-identical initial weights and dropout across methods.

## Results

### CIFAR-10 — Fully Complete (n=20/method)

| Sampler | Mean ± SD | Median | Range | Δ vs stochastic |
|---|---|---|---|---|
| **collatz_v3** | **85.87 ± 0.83** | 85.94 | 84.09–87.45 | **+0.76 pp, *p*≈0.006** |
| collatz_v2 | 85.34 ± 1.22 | 85.50 | 82.45–87.36 | +0.23 |
| sobol | 85.14 ± 0.87 | 84.86 | 83.33–86.86 | +0.03 |
| stochastic | 85.11 ± 0.90 | 85.28 | 82.81–86.41 | — |
| collatz_v1 | 84.83 ± 1.35 | 85.05 | 82.24–86.91 | −0.28 |

Paired by seed: v3 − stochastic = **+0.758 pp, *t*=3.47, wins 15/20**.

Version ladder: v1 < v2 < v3 (monotonic).

### CIFAR-100 — Current (n=10/method, seeds 42–51)

| Sampler | Mean ± SD | Range |
|---|---|---|
| collatz_v1 | 58.86 ± 0.39 | 58.17–59.35 |
| stochastic | 58.73 ± 0.43 | 58.17–59.40 |
| collatz_v3 | **58.68 ± 0.19** | **58.39–58.98** |
| sobol | 58.50 ± 0.42 | 57.99–59.38 |

No significant mean difference, but **−80% variance** for v3 (σ 0.19 vs 0.43) — the reproducibility claim. The pending `Semillas200` notebook will extend this to n=20 paired.

### Training Time (Tesla T4, paper config)

| Dataset | stochastic | collatz_v3 | Overhead |
|---|---|---|---|
| CIFAR-10 | 22.5 s/epoch | 23.4 s/epoch | +4.0% |
| CIFAR-100 | 41.8 s/epoch | 39.4* s/epoch | ≈0% |

*5 real seeds only (5 injected carry NaN timings).

## Methodology

- **Datasets:** CIFAR-10 (ResNet-9), CIFAR-100 (ResNet-18); MNIST debug uses SmallCNN.
- **Optimizer:** SGD lr=0.01 momentum=0.9 wd=1e-4, CosineAnnealingLR T_max=15, CrossEntropy.
- **Samplers:** see `src/dest/samplers.py`. Collatz orders are deterministic argsorts of K-iterated Collatz values; no global RNG consumption.
- **Splits:** `val_fraction=0.1`, `split_seed=0` fixed for all runs.
- **Statistics:** Welch two-sample + paired *t*, Cohen's *d*, Holm–Bonferroni (see `src/dest/statistics.py`).

## Generating Notebooks

```bash
python scripts/generate_paired_200_nb.py   # → notebooks/DEST_CIFAR100_Pareado_Semillas200.ipynb
python scripts/generate_missing_all_nb.py  # → notebooks/DEST_Todas_Semillas_Faltantes.ipynb
```

## Testing

```bash
pytest -q
```

## Citation

```bibtex
@misc{dest2026,
  title  = {DEST: Deterministic Collatz Sampling for Reproducible Deep Learning},
  author = {Rosario, Starlyn Eliezer},
  year   = {2026},
  url    = {https://github.com/starlyn2010/DEST}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built independently by a high-school student in the Dominican Republic using Google Colab (Tesla T4). No institutional affiliation.

## Contact

Issues and PRs welcome. For questions about the paired design, see `docs/resumen_operativo_DEST.md`.
