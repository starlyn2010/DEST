"""
statistics.py — Statistical analysis utilities for DEST experiments.

Changes vs v1:
  - NaN-safe p-value handling (occurs when comparing baseline to itself).
  - Cleaner Holm-Bonferroni with graceful statsmodels fallback.
  - compare_groups returns all fields even with minimal data (n=1).
"""

import warnings
import numpy as np
from scipy import stats
from typing import Any, Dict, List, Tuple


class Statistics:

    # ── bootstrap confidence intervals ────────────────────────────────────────
    @staticmethod
    def compute_ci_95_bootstrap(
        data: List[float], n_bootstrap: int = 10_000, seed: int = 42
    ) -> Tuple[float, float]:
        if len(data) < 2:
            mu = float(np.mean(data)) if data else 0.0
            return mu, mu
        rng   = np.random.RandomState(seed)
        means = [np.mean(rng.choice(data, size=len(data), replace=True))
                 for _ in range(n_bootstrap)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # ── Cohen's d ─────────────────────────────────────────────────────────────
    @staticmethod
    def cohens_d(group_a: List[float], group_b: List[float]) -> float:
        a, b = np.array(group_a), np.array(group_b)
        if len(a) < 2 or len(b) < 2:
            return 0.0
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        n1, n2 = len(a), len(b)
        s_p = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        return 0.0 if s_p == 0 else float((np.mean(a) - np.mean(b)) / s_p)

    # ── full group comparison ─────────────────────────────────────────────────
    @staticmethod
    def compare_groups(
        baseline: List[float], treatment: List[float]
    ) -> Dict[str, Any]:
        b, t = np.array(baseline), np.array(treatment)

        # -- t-test -----------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(b) == len(t) and len(b) >= 2:
                t_stat, p_ttest = stats.ttest_rel(t, b)
            elif len(b) >= 2 and len(t) >= 2:
                t_stat, p_ttest = stats.ttest_ind(t, b)
            else:
                t_stat, p_ttest = 0.0, 1.0

        # Guard NaN (e.g. identical arrays)
        p_ttest = float(p_ttest) if np.isfinite(p_ttest) else 1.0

        # -- Wilcoxon / Mann-Whitney ------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if len(b) == len(t) and len(b) >= 4:
                    w_stat, p_wil = stats.wilcoxon(t, b)
                elif len(b) >= 4 and len(t) >= 4:
                    w_stat, p_wil = stats.mannwhitneyu(t, b, alternative="two-sided")
                else:
                    w_stat, p_wil = 0.0, 1.0
            except Exception:
                w_stat, p_wil = 0.0, 1.0
        p_wil = float(p_wil) if np.isfinite(p_wil) else 1.0

        d = Statistics.cohens_d(list(treatment), list(baseline))
        ci_lo, ci_hi = Statistics.compute_ci_95_bootstrap(list(treatment))
        delta = float(np.mean(t) - np.mean(b)) if len(b) else 0.0

        return {
            "mean_treatment":      float(np.mean(t))  if len(t) else 0.0,
            "std_treatment":       float(np.std(t))   if len(t) else 0.0,
            "ci95_low":            ci_lo,
            "ci95_high":           ci_hi,
            "mean_baseline":       float(np.mean(b))  if len(b) else 0.0,
            "std_baseline":        float(np.std(b))   if len(b) else 0.0,
            "delta":               delta,
            "p_val_ttest":         p_ttest,
            "p_val_wilcoxon":      p_wil,
            "cohens_d":            float(d),
            "is_significant_ttest":    bool(p_ttest < 0.05),
            "is_significant_wilcoxon": bool(p_wil   < 0.05),
        }

    # ── Holm-Bonferroni ───────────────────────────────────────────────────────
    @staticmethod
    def apply_holm_bonferroni(
        p_values: List[float], alpha: float = 0.05
    ) -> Tuple[List[bool], List[float]]:
        p = [v if np.isfinite(v) else 1.0 for v in p_values]   # sanitize NaN

        try:
            from statsmodels.stats.multitest import multipletests
            rej, p_corr, _, _ = multipletests(p, alpha=alpha, method="holm")
            return rej.tolist(), p_corr.tolist()
        except ImportError:
            pass

        # manual fallback
        n = len(p)
        order = np.argsort(p)
        p_corr = np.ones(n)
        rej    = np.zeros(n, dtype=bool)
        for rank, idx in enumerate(order):
            adj = min(1.0, p[idx] * (n - rank))
            p_corr[idx] = adj
            if adj < alpha:
                rej[idx] = True
        return rej.tolist(), p_corr.tolist()

    # ── variance comparison summary ───────────────────────────────────────────
    @staticmethod
    def variance_reduction(baseline: List[float], treatment: List[float]) -> Dict[str, float]:
        """Return variance ratio and percentage reduction."""
        var_b = float(np.var(baseline)) if len(baseline) > 1 else 0.0
        var_t = float(np.var(treatment)) if len(treatment) > 1 else 0.0
        ratio    = var_t / var_b if var_b > 1e-12 else 1.0
        pct_red  = (1.0 - ratio) * 100.0
        return {"var_baseline": var_b, "var_treatment": var_t,
                "ratio": ratio, "pct_reduction": pct_red}
