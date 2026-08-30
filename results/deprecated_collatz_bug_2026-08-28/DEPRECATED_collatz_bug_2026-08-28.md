# DEPRECATED — Collatz sampler bug 98% duplicados

**Fecha:** 2026-08-28
**Bug:** `src/dest/samplers.py:88` `CollatzFix1/2/3Sampler` genera 45k valores con solo ~715 únicos (98% duplicados, `vals 3..2e9`), `argsort` estable 98% secuencial. `tau 42 vs 200 =0.18` (correlacionado). Afecta a todos los JSONs `collatz_*` generados antes del fix.

**Archivos afectados (NO USAR para paper):**
- `CIFAR10_collatz_v1/v2/v3` seeds 42–61 (20×3=60) — PAPER original `+0.76` fue suerte de `seed*10000` offset con `split_seed=0`
- `CIFAR10_collatz_v1/v3` seeds 200–209 Halton/Fashion inyección 200–204 (20) — también bug, inyectados sintéticamente pero con mismo vals bug
- `CIFAR100_collatz_v1/v3` seeds 42–51 — mismo bug

**No afectados:** `stochastic`, `sobol`, `halton` (45k únicos, tau~0).

**Acción:** Movidos a `deprecated_collatz_bug_2026-08-28/` para no usar en análisis futuros. El rerun `DEST_Collatz_Rerun_200_209.ipynb` debe usar sampler corregido (desempate aleatorio) cuando se implemente fix.

**Fix propuesto:** `samplers.py` añadir `+ 1e-9*rand` o `argsort((vals, rand))` para romper empates, o usar `np.argsort(values, kind='stable')` con clave secundaria.
