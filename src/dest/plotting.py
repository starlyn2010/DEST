"""
plotting.py — Publication-quality figures for DEST experiments.

New in v2:
  - Variance bands (mean ± std across seeds) on accuracy/loss curves.
  - plot_dataset_scaling: the central Phase-2 figure.
  - plot_final_comparison: grouped bar chart across methods per dataset.
  - plot_convergence_speed: bar chart of convergence epoch.
  - All figures saved as high-res PNG (300 dpi) + vector PDF.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from typing import Dict, List, Any, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── style constants ─────────────────────────────────
PALETTE = {
    "stochastic":  "#607D8B",
    "sobol":       "#2196F3",
    "collatz_v1":  "#FF9800",
    "collatz_v2":  "#9C27B0",
    "collatz_v3":  "#F44336",
}
LABELS = {
    "stochastic":  "Random Shuffle (baseline)",
    "sobol":       "Sobol Scrambled",
    "collatz_v1":  "Collatz V1",
    "collatz_v2":  "Collatz V2",
    "collatz_v3":  "Collatz V3 (DEST)",
}
DATASET_LABELS = {
    "MNIST":        "MNIST\n(easy)",
    "FASHIONMNIST": "Fashion-MNIST\n(moderate)",
    "CIFAR10":      "CIFAR-10\n(hard)",
    "CIFAR100":     "CIFAR-100\n(harder)",
    "TINYIMAGENET": "TinyImageNet\n(hardest)",
}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":    100,
})


# ─────────────────────────── helper ──────────────────────────────────────────
def _save(fig, paths):
    """Save figure to every path in `paths`."""
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _runs_to_matrix(runs, key="test_accs"):
    """Stack per-epoch curves from a list of RunResult dicts -> (seeds, epochs) array."""
    arr = [getattr(r, key, None) or r.get(key, []) for r in runs]
    # normalise length
    L = min(len(a) for a in arr) if arr else 0
    return np.array([a[:L] for a in arr]) if L else np.empty((0, 0))


# ─────────────────────────── Plotter class ───────────────────────────────────
class Plotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir  = os.path.join(output_dir, "plots")
        self.paper_dir  = os.path.join(output_dir, "paper", "figures")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.paper_dir, exist_ok=True)

    # ── variance-band accuracy curves ────────────────────────────────────────
    def plot_accuracy_curves(self, results_by_sampler: Dict, tag: str = ""):
        fig, ax = plt.subplots(figsize=(9, 5))
        for sampler, runs in results_by_sampler.items():
            mat = _runs_to_matrix(runs, "test_accs")
            if mat.size == 0:
                continue
            mu  = mat.mean(0)
            sd  = mat.std(0)
            ep  = np.arange(1, len(mu) + 1)
            c   = PALETTE.get(sampler, "#333333")
            ax.plot(ep, mu, color=c, label=LABELS.get(sampler, sampler), linewidth=2)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=c)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Test Accuracy per Epoch — {tag}")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        stem = f"{tag}_accuracy_curves" if tag else "accuracy_curves"
        _save(fig, [f"{self.plots_dir}/{stem}.png", f"{self.paper_dir}/{stem}.pdf"])

    # ── variance-band loss curves ─────────────────────────────────────────────
    def plot_loss_curves(self, results_by_sampler: Dict, tag: str = ""):
        fig, ax = plt.subplots(figsize=(9, 5))
        for sampler, runs in results_by_sampler.items():
            mat = _runs_to_matrix(runs, "train_losses")
            if mat.size == 0:
                continue
            mu  = mat.mean(0); sd = mat.std(0)
            ep  = np.arange(1, len(mu) + 1)
            c   = PALETTE.get(sampler, "#333333")
            ax.plot(ep, mu, color=c, label=LABELS.get(sampler, sampler), linewidth=2)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=c)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss")
        ax.set_title(f"Train Loss per Epoch — {tag}")
        ax.legend(); ax.grid(alpha=0.3)
        stem = f"{tag}_loss_curves" if tag else "loss_curves"
        _save(fig, [f"{self.plots_dir}/{stem}.png", f"{self.paper_dir}/{stem}.pdf"])

    # ── box-plots of final accuracy ──────────────────────────────────────────
    def plot_boxplots(self, results_by_sampler: Dict, tag: str = ""):
        samplers = list(results_by_sampler.keys())
        data = [[r.final_test_acc if hasattr(r, "final_test_acc") else r.get("final_test_acc", 0)
                 for r in results_by_sampler[s]] for s in samplers]

        fig, ax = plt.subplots(figsize=(max(6, len(samplers) * 1.8), 5))
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        for patch, s in zip(bp["boxes"], samplers):
            patch.set_facecolor(PALETTE.get(s, "#90A4AE"))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(samplers) + 1))
        ax.set_xticklabels([LABELS.get(s, s) for s in samplers], rotation=20, ha="right")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Final Test Accuracy Distribution — {tag}")
        ax.grid(alpha=0.3, axis="y")
        stem = f"{tag}_boxplots" if tag else "boxplots"
        _save(fig, [f"{self.plots_dir}/{stem}.png", f"{self.paper_dir}/{stem}.pdf"])

    # ── randomness sweep (Exp J) ─────────────────────────────────────────────
    def plot_randomness_sweep(self, sweep_results: Dict):
        alphas = sorted(sweep_results.keys())
        means  = [np.mean([r.final_test_acc for r in sweep_results[a]]) for a in alphas]
        stds   = [np.std( [r.final_test_acc for r in sweep_results[a]]) for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(alphas, means, yerr=stds, marker="o", linewidth=2, capsize=5, color="#E91E63")
        ax.set_xlabel("α (0 = pure Collatz, 1 = pure random)"); ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Randomness Sweep — Effect of α on Accuracy")
        ax.grid(alpha=0.3)
        _save(fig, [f"{self.plots_dir}/exp_J_randomness_sweep.png",
                    f"{self.paper_dir}/exp_J_randomness_sweep.pdf"])

    # ── ★ DATASET SCALING PLOT (Phase 2 key figure) ──────────────────────────
    def plot_dataset_scaling(
        self,
        all_results: Dict,           # keyed by (dataset, sampler_name)
        dataset_order: List[str],
        output_tag: str = "DEST_scaling_validation",
    ):
        """
        X-axis : Dataset difficulty (MNIST → TinyImageNet)
        Y-axis : Test accuracy improvement over stochastic baseline (Δ pp)

        One line per non-baseline sampler, with ±1σ band across seeds.
        The horizontal reference line at y=0 is the stochastic baseline.
        """
        comparators = ["sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        available_datasets = [
            d for d in dataset_order if any((d, s) in all_results for s in comparators)
        ]
        if not available_datasets:
            print("⚠️  No data available for scaling plot."); return

        fig, ax = plt.subplots(figsize=(11, 6))

        for sampler in comparators:
            x_vals, y_means, y_stds = [], [], []
            for i, ds in enumerate(available_datasets):
                baseline_runs  = all_results.get((ds, "stochastic"), [])
                treatment_runs = all_results.get((ds, sampler), [])
                if not baseline_runs or not treatment_runs:
                    continue
                base_acc = np.mean([r.final_test_acc for r in baseline_runs])
                trt_accs = np.array([r.final_test_acc for r in treatment_runs])
                x_vals.append(i)
                y_means.append(float(np.mean(trt_accs) - base_acc))
                y_stds.append(float(np.std(trt_accs)))

            if not x_vals:
                continue
            c  = PALETTE[sampler]
            lbl = LABELS[sampler]
            ax.plot(x_vals, y_means, "o-", color=c, label=lbl, linewidth=2.5, markersize=9)
            ax.fill_between(
                x_vals,
                np.array(y_means) - np.array(y_stds),
                np.array(y_means) + np.array(y_stds),
                alpha=0.15, color=c,
            )

        ax.axhline(0, color="#455A64", linestyle="--", linewidth=1.5,
                   label="Stochastic baseline (Δ = 0)")
        ax.set_xticks(range(len(available_datasets)))
        ax.set_xticklabels(
            [DATASET_LABELS.get(d, d) for d in available_datasets], fontsize=11
        )
        ax.set_xlabel("Dataset Difficulty →", fontsize=13)
        ax.set_ylabel("Test Accuracy Improvement over Random (pp)", fontsize=13)
        ax.set_title(
            "DEST Phase 2 — Does improvement scale with dataset difficulty?",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)

        _save(fig, [f"{self.plots_dir}/{output_tag}.png",
                    f"{self.paper_dir}/{output_tag}.pdf"])
        print(f"✅ Scaling plot saved → {self.plots_dir}/{output_tag}.png")

    # ── final accuracy comparison (grouped bar) ───────────────────────────────
    def plot_final_comparison(self, all_results: Dict, dataset_order: List[str]):
        samplers = ["stochastic", "sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        avail_ds = [d for d in dataset_order if any((d, s) in all_results for s in samplers)]
        if not avail_ds:
            return

        fig, axes = plt.subplots(1, len(avail_ds), figsize=(5 * len(avail_ds), 6), sharey=False)
        if len(avail_ds) == 1:
            axes = [axes]

        for ax, ds in zip(axes, avail_ds):
            labels, means, errors, colors = [], [], [], []
            for s in samplers:
                runs = all_results.get((ds, s), [])
                if not runs: continue
                accs = [r.final_test_acc for r in runs]
                labels.append(s.replace("_", "\n")); means.append(np.mean(accs))
                errors.append(np.std(accs)); colors.append(PALETTE[s])
            if not means: continue
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(ds); ax.set_ylabel("Test Acc (%)"); ax.grid(alpha=0.3, axis="y")
            ymin = max(0, min(means) - 2); ymax = max(means) + 1
            ax.set_ylim(ymin, ymax)

        fig.suptitle("Final Test Accuracy by Method and Dataset", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, [f"{self.plots_dir}/final_comparison.png",
                    f"{self.paper_dir}/final_comparison.pdf"])

    # ── convergence speed ─────────────────────────────────────────────────────
    def plot_convergence_speed(self, all_results: Dict, dataset_order: List[str]):
        samplers = ["stochastic", "sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        avail_ds = [d for d in dataset_order if any((d, s) in all_results for s in samplers)]
        if not avail_ds:
            return

        fig, ax = plt.subplots(figsize=(max(8, len(avail_ds) * 3), 5))
        x = np.arange(len(avail_ds))
        w = 0.15
        for i, s in enumerate(samplers):
            speeds = []
            for ds in avail_ds:
                runs = all_results.get((ds, s), [])
                conv = [r.convergence_epoch_90 for r in runs if r.convergence_epoch_90]
                speeds.append(np.mean(conv) if conv else None)
            y = [v if v is not None else 0 for v in speeds]
            ax.bar(x + i * w, y, w, label=LABELS[s], color=PALETTE[s], alpha=0.8)

        ax.set_xticks(x + w * 2); ax.set_xticklabels(avail_ds)
        ax.set_ylabel("Epoch to reach 90% test accuracy")
        ax.set_title("Convergence Speed by Method and Dataset")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        _save(fig, [f"{self.plots_dir}/convergence_speed.png",
                    f"{self.paper_dir}/convergence_speed.pdf"])

    # ── seed variance heatmap ─────────────────────────────────────────────────
    def plot_seed_variance(self, all_results: Dict, dataset_order: List[str]):
        samplers = ["stochastic", "sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        avail_ds = [d for d in dataset_order if any((d, s) in all_results for s in samplers)]
        if not avail_ds:
            return

        matrix = np.zeros((len(samplers), len(avail_ds)))
        for i, s in enumerate(samplers):
            for j, ds in enumerate(avail_ds):
                runs = all_results.get((ds, s), [])
                accs = [r.final_test_acc for r in runs]
                matrix[i, j] = np.std(accs) if len(accs) > 1 else 0.0

        fig, ax = plt.subplots(figsize=(max(6, len(avail_ds) * 2), 4))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Std of test accuracy (pp)")
        ax.set_xticks(range(len(avail_ds))); ax.set_xticklabels(avail_ds)
        ax.set_yticks(range(len(samplers))); ax.set_yticklabels([LABELS[s] for s in samplers])
        ax.set_title("Cross-seed Variance (lower = more stable)")
        for i in range(len(samplers)):
            for j in range(len(avail_ds)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        _save(fig, [f"{self.plots_dir}/seed_variance_heatmap.png",
                    f"{self.paper_dir}/seed_variance_heatmap.pdf"])
