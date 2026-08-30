# DEPRECATED RESULTS — Collatz Sampler Bug (98% Duplicates)

## ⚠️ CRITICAL: Do Not Use These Results

All results in `results/deprecated_collatz_bug_2026-08-28/` and `results/paper/` (CIFAR-10/CIFAR-100 Collatz runs) are **INVALID** due to a critical bug in the Collatz sampler.

## 🐛 The Bug: 98% Duplicate Values in Collatz Sampler

### What Happened
The Collatz samplers (`CollatzFix1Sampler`, `CollatzFix3Sampler`) generated permutation indices by computing Collatz orbit values and sorting them with `np.argsort()`. However, due to the limited range of Collatz orbit values:

- **45,000 samples → only ~715 unique values** (98.4% duplicates)
- `np.argsort()` with stable sort preserved original order for duplicates
- **Result: 98% of batches were sequential** (not low-discrepancy)

### Evidence
| Metric | Seed 42 | Seed 200 |
|--------|---------|----------|
| Unique values / 45,000 | 715 (1.6%) | 791 (1.8%) |
| Duplicates | 98.4% | 98.2% |
| Kendall τ (42 vs 200) | 0.18 (correlated) | — |
| τ vs sequential order | 0.04 (near-sequential) | — |

### Why It Mattered
- **42–61 seeds**: By chance, the `offset = seed*10000` produced a near-sequential order that happened to work well with the fixed `split_seed=0` → **+0.76pp p=0.006**
- **200–209 seeds**: Different offset produced unfavorable order → **−0.69pp p=0.0027**
- **Sobol/Halton**: No duplicates (45k unique), but tied with stochastic → QMC advantage not realized

---

## ✅ THE FIX (Commit `fed2d3c`)

### Fix Applied
```python
# src/dest/samplers.py - CollatzFix1Sampler & CollatzFix3Sampler
# BEFORE (buggy):
indices = np.argsort(values).tolist()

# AFTER (fixed):
tie_rng = np.random.RandomState(seed * 1000003 + epoch * 9176 + 777)
tie = tie_rng.rand(self.num_samples)
indices = np.lexsort((tie, norm_values)).tolist()
```

### Fix Verification
| Metric | Before | After |
|--------|--------|-------|
| Unique values / 45,000 | 715 (1.6%) | **45,000 (100%)** |
| Duplicates | 98.4% | **0%** |
| Kendall τ (42 vs 200) | 0.18 | **0.003** |
| τ vs sequential | 0.04 | **0.04** (now independent) |

Commit: [`fed2d3c`](https://github.com/starlyn2010/DEST/commit/fed2d3c) — "fix: Collatz samplers tie-breaker (lexsort) — 98% duplicates → 45k únicos, tau 0.18→0.00"

---

## 📁 DEPRECATED FOLDERS

| Folder | Contents | Status |
|--------|----------|--------|
| `results/deprecated_collatz_bug_2026-08-28/` | 90 JSONs (CIFAR-10/100 collatz_v1/v2/v3 seeds 42–61, 200–209) | **DO NOT USE** |
| `results/paper/` (Collatz runs) | Mixed with bug | **DEPRECATED** |

---

## ✅ CLEAN FIX RESULTS

### Valid Results (Post-Fix)
| Location | Description | Runs |
|----------|-------------|------|
| `results/clean_fix/` | 20 runs (seeds 300–309, fix applied) | 20 |
| `results/kaggle_clean_300_309/` | Same as clean_fix | 20 |
| `results/paper/` (Halton/Sobol/Stochastic) | Unaffected by bug | Valid |

### Clean Fix Results (Seeds 300–309, Fix Applied)
| Method | Mean ± SD | Diff vs Stochastic | p-value | Cohen's d |
|--------|-----------|-------------------|---------|-----------|
| stochastic | 85.27 ± 1.12 | — | — | — |
| collatz_v3 (fix) | 85.10 ± 0.85 | **−0.16 pp** | **0.53** | −0.21 |

> **Conclusion:** With the fix applied on fresh seeds (300–309), V3 **does not beat stochastic** (−0.16pp, p=0.53, d=−0.21). The original +0.76pp was an artifact of the 98% duplicate bug + offset luck.

---

## 📋 WHAT TO USE / WHAT TO IGNORE

| Use | Ignore |
|-----|--------|
| `results/clean_fix/` (20 runs, fix) | `results/deprecated_collatz_bug_2026-08-28/` (90 runs) |
| `results/paper/` Sobol/Halton/Stochastic | `results/paper/` Collatz runs |
| `results/kaggle_clean_300_309/` | `results/paper/` Collatz 42–61, 200–209 |
| `results/fashion/` (unaffected) | — |

---

## 📚 REFERENCES

- **Fix commit:** [`fed2d3c`](https://github.com/starlyn2010/DEST/commit/fed2d3c)
- **Full analysis:** `INFORME_OPERATIVO_COMPLETO_DEST.md`
- **Bug analysis:** `QMC_DEST_KILL.md`, `destroy_collatz_uniformity.md`
- **Skill for future:** `colab-perfecto` (in `INFORME_OPERATIVO_COMPLETO_DEST.md`)

---

> **Bottom line:** The original +0.76pp "victory" was a bug artifact. With the fix on fresh seeds, V3 ties with stochastic. No theorem, just conditional evidence.