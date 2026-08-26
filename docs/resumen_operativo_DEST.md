# RESUMEN OPERATIVO — Proyecto DEST (Deterministic Sampling for Enhanced Training)

**Fecha de corte:** 25 de agosto de 2026
**Datos:** 176 runs persistidos (`dest/dest_results_paper`: 140 · `dest/dest_results_debug`: 36)

---

## 1. Resumen ejecutivo

El proyecto compara **5 estrategias de ordenamiento de datos** (samplers) para entrenar redes
neuronales bajo configuración idéntica. La hipótesis de DEST es que un orden determinista basado
en la dinámica de Collatz puede igualar o superar el accuracy del orden aleatorio, reduciendo al
mismo tiempo la **varianza entre semillas** (mayor reproducibilidad).

Hallazgos principales hasta la fecha:

1. **CIFAR10 (n=20 semillas por método, completo):** `collatz_v3` es el mejor método en media
   (**85.87% ± 0.83**) superando a stochastic (+0.76 pp, t=2.76, **p≈0.006**, d=0.87) y a sobol
   (+0.72 pp, p≈0.007). Con análisis pareado por semilla la evidencia es aún más sólida
   (t=+3.47, gana en 15/20 semillas).
2. **CIFAR100 (n=10 por método):** collatz_v3 empata en accuracy con stochastic (−0.05 pp,
   n.s.) pero con **80% menos varianza** (±0.19 vs ±0.43). El rango completo de v3 es
   [58.39–58.98]: es el método más predecible.
3. **Costo:** el overhead computacional del sampler es marginal (≈4% más de tiempo de
   entrenamiento en CIFAR10 frente a stochastic; dentro del ruido en CIFAR100).
4. Existe una **escalera monotónica por versión** en CIFAR10: v1 (84.83) < v2 (85.34) < v3
   (85.87), consistente con que cada iteración del diseño mejora al anterior.

---

## 2. Diseño experimental

### 2.1 Métodos comparados
| Sampler | Descripción |
|---|---|
| `stochastic` | Orden aleatorio estándar (randperm por época, PyTorch). Línea base |
| `sobol` | Permutación derivada de secuencia Sobol (cuasi-aleatoria, scipy qmc) |
| `collatz_v1` | Orden determinista: valor final de K pasos de Collatz por índice + argsort |
| `collatz_v2` | Variante con ruido controlado en la dinámica (noise_ratio) |
| `collatz_v3` | Variante con calendario de mezcla α (alpha_start → alpha_end) |

### 2.2 Control de variables (garantía de pareo)
Toda corrida usa `run_single_seed()` del módulo `dest_lib.runner`, que garantiza:
- `seed_everything(seed)` al inicio (Python/NumPy/PyTorch/CUDA).
- **Split train/val/test fijo** (`split_seed=0`) idéntico para todos los runs.
- Los samplers usan **generadores RNG locales propios**: no consumen el stream global.
- En modo `deterministic`, las máscaras de dropout dependen solo de la seed.

Consecuencia: **a igual semilla, dos métodos parten de pesos iniciales bit-idénticos y ven el
mismo dataset dividido igual; lo único que difiere es el orden de presentación de muestras.**
Esto habilita análisis pareado por semilla (más potente que el independiente).

### 2.3 Configuración PAPER (la que producen los resultados principales)
| Parámetro | Valor |
|---|---|
| Modelo | ResNet18 (CIFAR100) / ResNet9 (CIFAR10), dropout determinista |
| Épocas | 15 |
| Optimizador | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch | 128 |
| Seeds | 42–61 (20) según disponibilidad del bloque |

---

## 3. Inventario general de experimentos

| Bloque | Dónde vive | Runs | Estado |
|---|---|---|---|
| CIFAR10 × 5 samplers × 20 seeds | `dest_results_paper` | 100 | ✅ Completo |
| CIFAR100 × 4 samplers × 10 seeds (42–51) | `dest_results_paper` | 40 | ⚠️ Parcial (falta v2 y seeds 52–61) |
| Debug MNIST (Exp A–J, 3 seeds c/u) | `dest_results_debug` | 36 | ✅ Completo* |
| Scaling Validation (FASHIONMNIST) | — | 0 | ❌ Nunca corrido (notebooks placeholder vacíos) |
| Strict Validation | — | 0 | ❌ Nunca corrido |
| Ablation Study | — | 0 | ❌ Nunca corrido |
| Pareado CIFAR100 semillas 200–209 | notebook listo | (50 plan.) | 🔵 Pendiente de ejecución |

\* Exp_B y Exp_D no existen en el set debug (fueron omitidos o no se persistieron).

---

## 4. Fase DEBUG — MNIST (sanity check, 36 runs)

Configuración reducida: SmallCNN, 3 épocas, seeds {42,43,44}, batch 256.

| Experimento | Contenido | Acc medio |
|---|---|---|
| Exp_A | stochastic (baseline) | 97.29% ± 0.08 |
| Exp_C | dropout determinista | **97.51% ± 0.06** (mejor) |
| Exp_E | full stack (sampler+dropout det.) | 97.44% ± 0.11 |
| Exp_F | sobol | 97.18% ± 0.15 |
| Exp_G / H / I | collatz v1 / v2 / v3 | 97.28 / 97.10 / 97.26 |
| Exp_J | barrido α ∈ {0, .25, .5, .75, 1} | 97.22–97.29 (plano) |

**Lectura:** en un problema saturado (~97%) ningún método destaca; sirve como verificación de
que el pipeline entrena estable y sin fugas de varianza anómalas. El barrido α muestra
sensibilidad nula a ese hiperparámetro en este régimen.

---

## 5. CIFAR10 — bloque principal COMPLETO (100 runs)

### 5.1 Resultados por método (20 semillas cada uno)
| Sampler | Media ± σ | Mediana | Rango | Reducción de varianza vs stoch |
|---|---|---|---|---|
| **collatz_v3** | **85.87 ± 0.83** | 85.94 | 84.09–87.45 | **−15%** |
| collatz_v2 | 85.34 ± 1.22 | 85.50 | 82.45–87.36 | −84% peor (mayor varianza: +84%) |
| sobol | 85.14 ± 0.87 | 84.86 | 83.33–86.86 | +7% |
| stochastic | 85.11 ± 0.90 | 85.28 | 82.81–86.41 | — (referencia) |
| collatz_v1 | 84.83 ± 1.35 | 85.05 | 82.24–86.91 | −125% (peor) |

### 5.2 Significancia estadística (Welch, bilateral)
| Comparación | Δ media | t | p aprox | Cohen's d |
|---|---|---|---|---|
| v3 vs stochastic | **+0.76 pp** | 2.76 | **≈0.006** | 0.87 (grande) |
| v3 vs sobol | +0.72 pp | 2.69 | ≈0.007 | 0.85 |
| v3 vs collatz_v1 | +1.04 pp | 2.92 | ≈0.003 | 0.92 |
| v3 vs collatz_v2 | +0.53 pp | 1.61 | ≈0.108 (n.s.) | 0.51 |

### 5.3 Análisis PAREADO por semilla (n=20, más sensible)
| Comparación | Δ pareada | t pareado | Victorias v3 |
|---|---|---|---|
| v3 − stochastic | +0.758 pp | **+3.47** | **15/20** |
| v3 − sobol | +0.724 pp | **+3.45** | 15/20 |
| v3 − collatz_v1 | +1.036 pp | +3.67 | 15/20 |
| v3 − collatz_v2 | +0.530 pp | +1.76 | 14/20 |

t>3 con 19 g.l. corresponde a p<0.005. El resultado de v3 sobre los baselines
(stochastic/sobol) es robusto por ambas vías de análisis.

### 5.4 Estabilidad de collatz_v3
Sus 20 semillas viven en [84.09, 87.45]; 16 de 20 están ≥85.0%. Su peor semilla (43: 84.09%)
supera a las peores de stochastic (82.81%), sobol (83.33%) y v1 (82.24%).

---

## 6. CIFAR100 — estado actual (40/80 runs, seeds 42–51)

### 6.1 Resultados por método
| Sampler | Media ± σ | Rango | Varianza relativa |
|---|---|---|---|
| collatz_v1 | 58.86 ± 0.39 | 58.17–59.35 | 0.82× |
| stochastic | 58.73 ± 0.43 | 58.17–59.40 | 1.00× (ref) |
| **collatz_v3** | **58.68 ± 0.19** | **58.39–58.98** | **0.20×** |
| sobol | 58.50 ± 0.42 | 57.99–59.38 | 0.95× |

*(collatz_v2 no tiene corridas en CIFAR100 todavía.)*

### 6.2 Lectura
- En el dataset difícil, **nadie gana en media** de forma significativa (diferencias <0.2 pp,
  todas n.s.). Es esperable: 15 épocas es corto para CIFAR100.
- El resultado diferencial es de **consistencia**: collatz_v3 reduce la desviación estándar un
  **~55%** respecto a todos los demás (σ=0.19 vs 0.39–0.43) y su rango completo cabe en 0.6 pp.
  Para reproducción científica esto significa: "elige cualquier seed y obtendrás ~58.7%",
  mientras stochastic puede dar entre 58.2 y 59.4.
- Pareado por semilla (n=10): v3−stoch = −0.047 pp (t=−0.30, 6/10); v3−sobol = +0.184
  (t=+1.41, 7/10).

---

## 7. Eficiencia en tiempo de entrenamiento (runs reales)

Medido en Tesla T4, mismo hardware y config:

| Dataset | Método | Train s/época | Runtime total | Throughput |
|---|---|---|---|---|
| CIFAR10 | stochastic | 22.5 | 6.8 min | 2007 samp/s |
| CIFAR10 | sobol | 22.5 | 6.8 min | 2013 samp/s |
| CIFAR10 | collatz_v1 | 22.8 | 6.9 min | 1987 samp/s |
| CIFAR10 | collatz_v2 | 22.6 | 6.8 min | 2001 samp/s |
| CIFAR10 | collatz_v3 | 23.4 | 7.0 min | 1927 samp/s |
| CIFAR100 | stochastic | 41.8 | 11.8 min | 1075 samp/s |
| CIFAR100 | collatz_v3 (5 reales) | 39.4 | 11.1 min | — |

**Lectura:** collatz_v3 cuesta ≈4% más de entrenamiento en CIFAR10 (23.4 vs 22.5 s/época,
atribuible a generar la permutación Collatz + argsort por época) y es indistinguible (o más
rápido dentro del ruido) en CIFAR100. El costo del determinismo es marginal frente a su ganancia
de +0.76 pp y −16% de varianza.

---

## 8. Calidad de datos y limitaciones declaradas

1. **5 JSON inyectados desde log (CIFAR100 v3, seeds 46–50):** sus accuracies y losses por
   época son reales (copiados de la salida de entrenamiento); métricas secundarias no visibles
   en log (val_loss, F1, precision/recall, ECE, tiempos) van como NaN. Están marcados con
   `"nota": "inyectado desde log"` y `"status": "COMPLETE"`. Las medias/σ de accuracy no se
   ven afectadas.
2. **Desbalance de n:** CIFAR10 tiene 20 seeds/método; CIFAR100 10 seeds y sin collatz_v2.
   Las comparaciones cruzadas entre datasets deben respetar esa asimetría.
3. **p-valores aproximados** por vía normal; para el paper conviene exactos (scipy) + Holm-
   Bonferroni (ya implementado en `dest_lib/statistics.py`).
4. Familias Scaling_Validation / Strict / Ablation existen como notebooks pero **sin resultados**.

---

## 9. Trabajo en curso

**Notebook `DEST_CIFAR100_Pareado_Semillas200.ipynb`** (listo, pendiente de ejecutar):
- Diseño pareado limpio: **semillas 200–209 × 5 métodos = 50 runs** (rango aislado para no
  mezclarse con 42–61).
- Cada semilla corre los 5 samplers en bloque → cada bloque completado cierra un set pareado.
- Reanudable tras desconexiones (salta JSON existentes).
- Reporta al final: tabla de accuracy por semilla, diffs pareadas, t-pareado, victorias, y
  **medición de tiempos** (overhead del sampler vs entrenamiento).

Al integrarse, CIFAR100 pasará de n=10 a n=20 por método con diseño 100% pareado, homólogo a
CIFAR10.

---

## 10. Conclusiones operativas

1. **Resultado principal defendible hoy:** en CIFAR10, collatz_v3 supera significativamente al
   orden aleatorio y a sobol en accuracy (+0.7–0.8 pp, p<0.01, d≈0.86) reduciendo además la
   varianza entre semillas. Efecto grande según Cohen.
2. **Resultado complementario:** en CIFAR100 mantiene el accuracy promedio y recorta la
   variabilidad ~80% — el argumento de reproducibilidad.
3. **La escalera v1<v2<v3** sugiere que las mejoras de diseño del sampler son aditivas.
4. **Riesgos/pending:** completar CIFAR100 (v2 + seeds restantes o el bloque 200–209),
   generar p-valores exactos y figuras para el paper, y decidir si las familias nunca corridas
   (Scaling Validation, Ablation) se ejecutan o se descartan formalmente.

---
*Métricas calculadas directamente sobre los 140 JSON de `dest/dest_results_paper` y 36 de
`dest/dest_results_debug`. Scripts de soporte: `analyze_paper_results.py`,
`dest_lib/statistics.py`, generadores de notebooks en raíz.*
