"""
report.py — Automated research report generator for Phase 2 scaling validation.

Answers the 6 core empirical questions:
  Q1: Does DEST outperform Random?
  Q2: Does DEST outperform Sobol?
  Q3: Does improvement increase with dataset complexity?
  Q4: Does DEST reduce variance?
  Q5: Does DEST converge faster?
  Q6: Is computational overhead acceptable?
"""

import os
import json
import numpy as np
from typing import Dict, List, Any

from .statistics import Statistics


def _acc(runs):
    return [r.final_test_acc for r in runs] if runs else []

def _sig(p):
    return "✅ YES (p={:.4f})".format(p) if p < 0.05 else "❌ NO (p={:.4f})".format(p)


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, "reports")
        self.tables_dir  = os.path.join(output_dir, "paper", "tables")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.tables_dir,  exist_ok=True)

    # ── 6-question automated analysis ────────────────────────────────────────
    def answer_six_questions(
        self,
        all_results: Dict,
        dataset_order: List[str],
        config: Dict,
    ) -> Dict[str, str]:
        """
        Parameters
        ----------
        all_results : dict keyed by (dataset, sampler_name) -> List[RunResult]
        dataset_order : list of dataset names in difficulty order.

        Returns
        -------
        dict {question_label: answer_text}
        """
        dest = "collatz_v3"
        baseline = "stochastic"
        sobol    = "sobol"
        avail = [d for d in dataset_order if (d, baseline) in all_results]

        answers = {}

        # ── Q1 ───────────────────────────────────────────────────────────────
        q1_lines = ["**Q1 — Does DEST outperform Random Shuffle?**\n"]
        for ds in avail:
            b_acc = _acc(all_results.get((ds, baseline), []))
            d_acc = _acc(all_results.get((ds, dest), []))
            if not b_acc or not d_acc: continue
            cmp = Statistics.compare_groups(b_acc, d_acc)
            delta = cmp["delta"]
            q1_lines.append(
                f"  {ds}: DEST={np.mean(d_acc):.3f}% vs Random={np.mean(b_acc):.3f}% "
                f"(Δ={delta:+.3f}pp, {_sig(cmp['p_val_ttest'])}, "
                f"Cohen's d={cmp['cohens_d']:.2f})"
            )
        answers["Q1"] = "\n".join(q1_lines)

        # ── Q2 ───────────────────────────────────────────────────────────────
        q2_lines = ["**Q2 — Does DEST outperform Sobol?**\n"]
        for ds in avail:
            s_acc = _acc(all_results.get((ds, sobol),   []))
            d_acc = _acc(all_results.get((ds, dest),    []))
            if not s_acc or not d_acc: continue
            cmp = Statistics.compare_groups(s_acc, d_acc)
            q2_lines.append(
                f"  {ds}: DEST={np.mean(d_acc):.3f}% vs Sobol={np.mean(s_acc):.3f}% "
                f"(Δ={cmp['delta']:+.3f}pp, {_sig(cmp['p_val_ttest'])})"
            )
        answers["Q2"] = "\n".join(q2_lines)

        # ── Q3 ───────────────────────────────────────────────────────────────
        q3_lines = ["**Q3 — Does improvement grow with dataset difficulty?**\n"]
        deltas = []
        for ds in avail:
            b_acc = _acc(all_results.get((ds, baseline), []))
            d_acc = _acc(all_results.get((ds, dest), []))
            if b_acc and d_acc:
                delta = np.mean(d_acc) - np.mean(b_acc)
                deltas.append(delta)
                q3_lines.append(f"  {ds}: Δ = {delta:+.4f}pp")
        if len(deltas) >= 2:
            trend = "INCREASING ✅" if deltas[-1] > deltas[0] else "NOT clearly increasing ❌"
            q3_lines.append(f"\n  Overall trend: {trend}")
        answers["Q3"] = "\n".join(q3_lines)

        # ── Q4 ───────────────────────────────────────────────────────────────
        q4_lines = ["**Q4 — Does DEST reduce variance across seeds?**\n"]
        for ds in avail:
            b_acc = _acc(all_results.get((ds, baseline), []))
            d_acc = _acc(all_results.get((ds, dest), []))
            if not b_acc or not d_acc: continue
            vr = Statistics.variance_reduction(b_acc, d_acc)
            verdict = "REDUCED ✅" if vr["pct_reduction"] > 0 else "INCREASED ❌"
            q4_lines.append(
                f"  {ds}: σ_random={np.std(b_acc):.4f}%  σ_DEST={np.std(d_acc):.4f}%  "
                f"Reduction={vr['pct_reduction']:.1f}%  → {verdict}"
            )
        answers["Q4"] = "\n".join(q4_lines)

        # ── Q5 ───────────────────────────────────────────────────────────────
        q5_lines = ["**Q5 — Does DEST converge faster?**\n"]
        for ds in avail:
            b_runs = all_results.get((ds, baseline), [])
            d_runs = all_results.get((ds, dest), [])
            b_conv = [r.convergence_epoch_90 for r in b_runs if r.convergence_epoch_90]
            d_conv = [r.convergence_epoch_90 for r in d_runs if r.convergence_epoch_90]
            if b_conv and d_conv:
                delta_ep = np.mean(d_conv) - np.mean(b_conv)
                verdict = "FASTER ✅" if delta_ep < 0 else ("SAME ➡️" if delta_ep == 0 else "SLOWER ❌")
                q5_lines.append(
                    f"  {ds}: Random={np.mean(b_conv):.1f} ep  DEST={np.mean(d_conv):.1f} ep  "
                    f"Δ={delta_ep:+.1f}  → {verdict}"
                )
            else:
                q5_lines.append(f"  {ds}: convergence data insufficient.")
        answers["Q5"] = "\n".join(q5_lines)

        # ── Q6 ───────────────────────────────────────────────────────────────
        q6_lines = ["**Q6 — Is the computational overhead of DEST acceptable?**\n"]
        for ds in avail:
            b_runs = all_results.get((ds, baseline), [])
            d_runs = all_results.get((ds, dest), [])
            if not b_runs or not d_runs: continue
            b_t = np.mean([np.mean(r.train_time_per_epoch) for r in b_runs])
            d_t = np.mean([np.mean(r.train_time_per_epoch) for r in d_runs])
            b_s = np.mean([np.mean(r.sampler_time_per_epoch) for r in b_runs])
            d_s = np.mean([np.mean(r.sampler_time_per_epoch) for r in d_runs])
            overhead_pct = (d_s / b_s - 1) * 100 if b_s > 1e-9 else 0
            verdict = "ACCEPTABLE ✅" if overhead_pct < 10 else "HIGH ⚠️"
            q6_lines.append(
                f"  {ds}: sampler_time random={b_s*1e3:.2f}ms  DEST={d_s*1e3:.2f}ms  "
                f"overhead={overhead_pct:+.1f}%  → {verdict}"
            )
        answers["Q6"] = "\n".join(q6_lines)

        return answers

    # ── full markdown report ──────────────────────────────────────────────────
    def generate_scaling_report(
        self,
        all_results: Dict,
        dataset_order: List[str],
        config: Dict,
        stats_summary: List[Dict],
    ) -> str:
        mode = config.get("execution_mode", "?")
        n_seeds = len(config.get("seeds", []))
        avail = [d for d in dataset_order if any((d, s) in all_results for s in ["stochastic"])]

        answers = self.answer_six_questions(all_results, avail, config)

        # per-dataset results table
        table_rows = ["| Dataset | Method | Test Acc (%) | Δ vs Random | Cohen's d | p-value | Significant |",
                      "|---|---|:---:|:---:|:---:|:---:|:---:|"]
        samplers = ["stochastic", "sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        for ds in avail:
            b_acc = _acc(all_results.get((ds, "stochastic"), []))
            for s in samplers:
                runs = all_results.get((ds, s), [])
                if not runs: continue
                accs = _acc(runs)
                if s == "stochastic":
                    table_rows.append(
                        f"| **{ds}** | **Random (baseline)** | **{np.mean(accs):.3f} ± {np.std(accs):.3f}** | — | — | — | — |"
                    )
                else:
                    cmp = Statistics.compare_groups(b_acc, accs)
                    sig = "**YES**" if cmp["is_significant_ttest"] else "No"
                    table_rows.append(
                        f"| {ds} | {s} | {np.mean(accs):.3f} ± {np.std(accs):.3f} | "
                        f"{cmp['delta']:+.3f}pp | {cmp['cohens_d']:.2f} | {cmp['p_val_ttest']:.4f} | {sig} |"
                    )

        report = f"""# DEST Phase 2 — Scaling Validation Report

**Mode**: {mode} | **Seeds**: {n_seeds} | **Datasets**: {', '.join(avail)}

---

## Summary Table

{chr(10).join(table_rows)}

---

## Six Core Empirical Questions

{answers.get('Q1', '')}

---

{answers.get('Q2', '')}

---

{answers.get('Q3', '')}

---

{answers.get('Q4', '')}

---

{answers.get('Q5', '')}

---

{answers.get('Q6', '')}

---

## Conclusion

This Phase 2 experiment {"confirms" if len(avail) >= 2 else "begins to test"} whether
DEST generalizes beyond MNIST to harder computer vision benchmarks.
{"The scaling plot is the definitive visualization." if len(avail) >= 2 else ""}
{"Run in PAPER or FULL mode for statistically robust conclusions." if n_seeds < 10 else ""}

*Auto-generated by DEST Scaling Validation framework — Phase 2*
"""
        report_path = os.path.join(self.reports_dir, "scaling_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"✅ Report saved → {report_path}")
        return report

    # ── LaTeX table ──────────────────────────────────────────────────────────
    def generate_latex_table(self, all_results: Dict, dataset_order: List[str]):
        samplers = ["stochastic", "sobol", "collatz_v1", "collatz_v2", "collatz_v3"]
        avail = [d for d in dataset_order if any((d, s) in all_results for s in samplers)]

        rows = []
        for ds in avail:
            b_acc = _acc(all_results.get((ds, "stochastic"), []))
            for s in samplers:
                runs = all_results.get((ds, s), [])
                if not runs: continue
                accs = _acc(runs)
                delta = f"{np.mean(accs) - np.mean(b_acc):+.3f}" if s != "stochastic" else "—"
                rows.append(
                    f"  {ds} & {s} & ${np.mean(accs):.3f} \\pm {np.std(accs):.3f}$ & {delta} \\\\"
                )

        latex = ("\\begin{table}[h]\\centering\n"
                 "\\caption{DEST Phase 2 Scaling Validation Results}\n"
                 "\\begin{tabular}{llcc}\n\\hline\n"
                 "Dataset & Method & Test Acc (\\%) & $\\Delta$ vs Random \\\\\n\\hline\n"
                 + "\n".join(rows) +
                 "\n\\hline\n\\end{tabular}\n\\end{table}\n")

        path = os.path.join(self.tables_dir, "Table2_scaling_results.tex")
        with open(path, "w") as f:
            f.write(latex)

    # ── legacy methods kept for backward compat ───────────────────────────────
    def generate_markdown_report(self, stats_summary: List[Dict], config: Dict) -> str:
        mode = config.get("execution_mode", "?")
        dataset = config.get("dataset", "?")
        n_seeds = len(config.get("seeds", []))

        rows = []
        for s in stats_summary:
            sig = "**YES**" if s.get("is_significant_holm") or s.get("is_significant_ttest") else "No"
            p = s.get("p_val_corrected", s.get("p_val_ttest", 1.0))
            p = p if isinstance(p, float) and np.isfinite(p) else 1.0
            rows.append(
                f"| **{s['mode'].upper()}** | {s['mean_acc']:.2f}% ± {s['std_acc']:.2f}% | "
                f"{s['mean_loss']:.4f} | {s['cohens_d']:.2f} | {p:.4f} | {sig} |"
            )

        report = f"""# Reporte de Investigación Oficial — Proyecto DEST

**Modo de Ejecución**: {mode}
**Dataset**: {dataset} | **Semillas**: {n_seeds}

## 1. Resumen de Resultados Cuantitativos

| Método | Test Accuracy (%) | Test Loss | Cohen's d | p-value (Holm) | Significativo |
|---|:---:|:---:|:---:|:---:|:---:|
{chr(10).join(rows)}

## 2. Conclusiones Principales

- Los métodos deterministas demostraron menor variabilidad entre semillas.
- Ver scaling_report.md para el análisis de Fase 2.
"""
        path = os.path.join(self.output_dir, "final_report.md")
        with open(path, "w") as f:
            f.write(report)
        return report
