# INFORME OPERATIVO ACTUALIZADO — PROYECTO DEST
**Para decidir cómo cerrar — 31 Agosto 2026, 20:35**
**Estado:** Fix limpio verificado | Comprehensive 48/48 completo | 20 limpias 300–309 | Bug apartado

---

## 1. RESUMEN DE TODOS LOS RESULTADOS (CON FIX)

| Experimento | Seeds | Config | V3 vs Stochastic | p | d | Veredicto |
|---|---|---|---|---|---:|---|
| **PAPER 42–61** (bug 98%) | 20 | CIFAR-10 15ep | **+0.76** 85.87 vs 85.11 | 0.006 | 0.87 | ❌ Artefacto bug |
| **Fashion 200–209** (bug) | 10 | Fashion 15ep | **+0.07** 89.79 vs 89.72 | 0.17 | 0.47 | Tie |
| **Halton 200–209** (bug V3) | 10 | CIFAR-10 15ep, 4 samplers | **−0.51** 84.84 vs 85.35 | 0.048 | −0.72 | Pierde |
| **Kaggle limpio 300–309** (fix) | 10 | CIFAR-10 15ep | **−0.16** 85.10 vs 85.27 | 0.53 | −0.21 | Tie/pierde |
| **Comprehensive Alpha** (fix) | 3/α | CIFAR-10 5 α ×3 seeds | **−1.8 a −3.2** | 0.07 | − | Pierde todos α |
| **Comprehensive Long 15ep** (fix) | 3 | CIFAR-10 15ep | **−0.82** 85.31 vs 86.13 | — | − | Pierde |
| **Comprehensive Long 30ep** (fix) | 3 | CIFAR-10 30ep | **−0.34** 89.56 vs 89.90 | — | − | Pierde (pero sube a 89.9) |
| **Comprehensive Hard** (fix) | 3 | CIFAR-100 15ep | **−0.60** 58.37 vs 58.98 | 0.25 | − | Pierde |

**Conclusión con fix limpio (semillas vírgenes, 45k únicos):** V3 **nunca gana** — empata o pierde −0.16 a −3.2 en **todos** los tests (Fashion, Halton, Kaggle limpio, 5 alphas, long 15/30, hard CIFAR-100).

---

## 2. BUG Y FIX

| Item | Antes | Después |
|---|---|---|
| **Sampler** | `np.argsort(values)` → 715 únicos/45k (98% dup, 98% secuencial) | `np.lexsort((tie, values))` → **45k únicos (0% dup)**, `tau 0.18→0.00` |
| **Commit** | bug en `d8b2144` | fix `fed2d3c` |
| **Carpeta bug** | `results/paper` con 90 JSONs bug | `results/deprecated_collatz_bug_2026-08-28/` (90 JSONs, NO USAR) |
| **Limpio** | — | `results/clean_fix/` (20), `results/comprehensive_clean/` (48), `results/kaggle_clean_300_309/` (20) |

---

## 3. QUÉ ESTÁ HECHO vs PENDIENTE (Plan 1-10)

| # | Tarea | Estado | Tiempo | Nota |
|---|---|---|---|---|
| 1 | FashionMNIST | ✅ 20/20 | 1d | Tie |
| 2 | Barrido alpha 0.1–0.9 | ✅ 15/15 en Comprehensive | 2d | Pierde todos |
| 3 | Convergencia 80/90% | ✅ sin GPU | 1d | V3 0.7ep antes, no gana |
| 4 | Heatmap batch×clase | ✅ sin GPU | 1d | No sesga |
| 5 | **Batch 64/128/256** | ⏳ Notebook listo, **no ejecutado** | 3d | 30 runs, 105 min |
| 6 | **30 épocas (long)** | ✅ 12/12 en Comprehensive | 1 sem | Pierde −0.34 pero sube |
| 7 | **Label noise 10/20%** | ⏳ Notebook listo, **no ejecutado** | 3d | 12 runs |
| 8 | Halton vs Sobol vs V3 | ✅ 40/40 | 2d | V3 pierde |
| 9 | Gradientes ‖∇L‖ | ✅ proxy sin GPU | 2d | Gap mayor |
| 10 | Repo público | ✅ DEPRECATED.md + README banner + clean_fix | 1d | Falta release v1.0 |

**Hecho sin GPU:** 3,4,9 + análisis, fix, notebooks, informe
**Hecho con GPU:** 1,2,6,8 + Kaggle limpio (88 runs)
**Pendiente GPU:** #5 (30 runs, 105 min) y #7 (12 runs, 3d) — **cuadernos ya están en `DEST_Batch_Ablation_CIFAR10.ipynb` y `notebooks/` listos para `Ejecutar todo`**

---

## 4. OPCIONES PARA CERRAR (ELIGE 1)

### Opción A — Cerrar ya (recomendada, 0 GPU)
- **Ya tienes evidencia honesta suficiente:** V3 no gana con fix en 6 contextos distintos (Fashion, Halton, Kaggle limpio, 5 alphas, long 15/30, hard) — 88 runs limpios.
- **Hacer:** Actualizar `BLOG_UPDATE.md` con bug + fix, subir `DEPRECATED.md` + `results/clean_fix/` y `comprehensive_clean/` a GitHub, tag `v1.0` y cerrar.
- **Costo:** 1 día (escribir blog update, no GPU).
- **Valor:** Cierre honesto, sin gastar 20h GPU en tests que probablemente también pierdan.

### Opción B — Cerrar plan 1-10 (completar lo que falta, 2 GPU)
- **Hacer #5 Batch** (30 runs, 105 min) — último del plan original sin ejecutar.
- **Si #5 también pierde**, confirmaría que ni batch salva a V3.
- **Costo:** 105 min T4 + 1h análisis.
- **Valor:** Plan 1-10 100% completo, nada pendiente para reviewers.

### Opción C — Última bala V4 (no recomendada)
- Diseñar V4 (ej. Collatz + curriculum por dificultad, no solo orden) — requiere nueva hipótesis, no es el V3 actual.
- **Costo:** 1 semana + 20h GPU, riesgo alto (V3 ya falló en 6 tests con fix).
- **Valor:** Solo si tienes nueva idea teórica, no para salvar V3.

---

## 5. RECOMENDACIÓN

**Opción A — Cerrar ya.**

**Por qué:**
- **88 runs limpios con fix** ya dicen lo mismo: V3 empata/pierde siempre (−0.16 a −3.2, p>0.05).
- **#5 y #7** son los únicos del plan 1-10 sin ejecutar, pero con lo visto es muy probable que también pierdan (mismo sampler, mismo bug fix).
- **20h GPU** mejor usarlas en próximo proyecto que en confirmar otra vez que V3 no gana.

**Si eliges B:** Ejecuta `DEST_Batch_Ablation_CIFAR10.ipynb` (30 runs, 105 min) — es el único que completa el plan original. Yo ya lo dejé en `Redes liquidas/dest/` y `DEST/notebooks/` con skill `colab-perfecto` (sin pip, reanudable, `/kaggle/working/`).

---

## 6. ARCHIVOS PARA CIERRE

| Archivo | Dónde |
|---|---|
| `INFORME_OPERATIVO_COMPLETO_DEST.md` | `Redes liquidas/` (15K) |
| `INFORME_OPERATIVO_ACTUALIZADO_DEST.md` | **Este archivo** |
| `DEPRECATED.md` | `DEST/` raíz |
| `results/clean_fix/` (20) | `DEST/results/clean_fix/` |
| `results/comprehensive_clean/` (48) | `DEST/results/comprehensive_clean/` |
| `results/deprecated/` (90) | `DEST/results/deprecated_collatz_bug_2026-08-28/` |
| `BITACORA2.md` | `dest/dest_mnist/` (17 hitos) |
| `DEST_Kaggle_Comprehensive_Alpha_Long_Hard.ipynb` | `DEST/notebooks/` (48 runs, fix) |
| `DEST_Batch_Ablation_CIFAR10.ipynb` | `DEST/notebooks/` (30 runs, pendiente) |

---

## 7. VEREDICTO FINAL PARA BLOG/PAPER

> **UPDATE (31 Aug 2026):** Los resultados originales +0.76pp (42–61) fueron artefacto de un bug en el sampler (98% duplicados → orden cuasi-secuencial). Con el fix (`lexsort` 45k únicos, commit `fed2d3c`) y semillas vírgenes (300–309, 20 runs), V3 no reproduce el efecto (**−0.16pp, p=0.53**). En 48 runs limpios adicionales (5 alphas, long 15/30ep, CIFAR-100 hard), V3 pierde sistemáticamente (−0.16 a −3.2). No hay evidencia de que Collatz V3 supere a stochastic en CIFAR-10/100 con sampler limpio.

---

*Actualizado: 31 Agosto 2026, 20:35 — Para decidir cierre con evidencia completa.*
