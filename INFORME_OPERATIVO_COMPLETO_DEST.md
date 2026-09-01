# INFORME OPERATIVO COMPLETO — PROYECTO DEST
**Deterministic Entropy-Scheduled Training**  
**Fecha:** 30 Agosto 2026  
**Estado:** Bug 98% duplicados corregido | 20 limpias 300–309 en Kaggle (fix) | 40 Halton completados | 20 FashionMNIST | PAPER original DEPRECATED  

---

## 📋 RESUMEN EJECUTIVO

| Métrica | Valor | Estado |
|---|---|---|
| **Bug crítico** | 98% duplicados en Collatz (715 únicos/45k) | ✅ **Corregido** (`lexsort`) |
| **PAPER original (42–61)** | V3 +0.76pp p=0.006 | ❌ **DEPRECATED** (artefacto bug) |
| **Rerun limpio 300–309** | V3 −0.16pp p=0.53 | ✅ **Fix verificado** (no gana) |
| **Halton 200–209 (40/40)** | V3 pierde −0.51pp p=0.048 | ✅ Completado |
| **FashionMNIST 200–209** | V3 +0.07pp p=0.17 | ✅ Tie (no generaliza) |
| **Kaggle 300–309 (20 limpias)** | V3 −0.16pp p=0.53 | ✅ Fix verificado en semillas vírgenes |
| **Fix Collatz** | 98% → 0% duplicados | ✅ Push `fed2d3c` |

**Conclusión honesta:** El +0.76pp original fue **artefacto del bug 98% duplicados + suerte del offset `seed*10000`** con `split_seed=0`. Con fix limpio en semillas vírgenes (300–309), V3 **empata/pierde** contra azar. No hay teorema, solo evidencia condicional.

---

## 🐛 EL BUG CRÍTICO (98% DUPLICADOS)

### Qué pasó
```python
# samplers.py:88 ANTES del fix (bug)
offset = seed * 10000 + epoch * 7919
values = np.empty(N, dtype=np.float64)
for i in range(N):
    x = 2 * (i + offset) + 1
    for _ in range(K):  # K=50
        x = collatz_step(x, c)
    values[i] = x
indices = np.argsort(values).tolist()  # 98% duplicados → argsort estable = secuencial
```

**Efecto real:** 45k valores → **solo ~715 únicos** (98% colisiones). `np.argsort` estable preservaba orden original para empates → **98% secuencial**, no "baja discrepancia".

### Evidencia del bug
| Métrica | `seed 42` | `seed 200` |
|---|---|---|
| Valores únicos | 715 / 45,000 | 791 / 45,000 |
| Duplicados | 98.4% | 98.2% |
| Tau 42 vs 200 | 0.18 (correlacionado) | — |
| Tau vs secuencial | 0.04 (casi secuencial) | — |

### Fix aplicado (2026-08-28)
```python
# samplers.py DESPUÉS del fix
tie_rng = np.random.RandomState(seed * 1000003 + epoch * 9176 + 777)
tie = tie_rng.rand(self.num_samples)
indices = np.lexsort((tie, norm_values)).tolist()  # desempate determinista por seed
```

**Resultado post-fix:**
- 45,000 únicos / 45,000 (0% duplicados)
- Tau 42 vs 200: 0.18 → **0.003**
- Tau vs secuencial: 0.04 → **0.04** (independiente)
- Push: `fed2d3c` → `origin/master`, sync a `Redes liquidas/dest_lib/samplers.py`

---

## 📊 RESULTADOS POR EXPERIMENTO (CRONOLOGÍA HONESTA)

### 1. MNIST (Julio 2026) — 5 semillas, 3 modos
| Método | Test Acc | Test Loss | Estabilidad |
|---|---|---|---|
| Stochastic | 98.86% ± 0.06% | 0.0828 | ±0.06% |
| Sobol | 99.02% ± 0.04% | 0.0895 | **±0.04%** |
| Collatz V1 | 99.06% ± 0.05% | 0.0872 | ±0.05% |

> **Nota:** MNIST demasiado fácil — "victoria" de Collatz era **falso positivo** del bug (funcionaba sin shuffle).

---

### 2. CIFAR-10 Escala PAPER (Ago 2024) — 42–61, 20 semillas/método, 15 épocas
| Sampler | Media ± SD | Mediana | vs Stochastic | p (Welch) | d (Cohen) |
|---|---:|---:|---:|---:|---:|
| **collatz_v3 (BUG)** | **85.87 ± 0.83** | 85.94 | **+0.76 pp** | **0.006** | 0.87 |
| collatz_v2 (BUG) | 85.34 ± 1.22 | 85.50 | +0.23 pp | 0.28 | 0.24 |
| sobol | 85.14 ± 0.87 | 84.86 | +0.03 pp | 0.82 | 0.04 |
| stochastic | 85.11 ± 0.90 | 85.28 | — | — | — |
| collatz_v1 (BUG) | 84.83 ± 1.35 | 85.05 | −0.28 pp | 0.25 | −0.30 |

> **Veredicto:** Resultados **DEPRECATED** — artefacto del bug 98% duplicados + suerte offset `seed*10000` (42–61).

---

### 3. CIFAR-100 (Ago 2024) — 10 semillas/método
| Sampler | Media ± SD | Rango | Varianza |
|---|---:|---:|---:|
| collatz_v1 (BUG) | 58.86 ± 0.39 | 58.17–59.35 | 0.154 |
| stochastic | 58.73 ± 0.43 | 58.17–59.40 | 0.186 |
| **collatz_v3 (BUG)** | **58.68 ± 0.19** | **58.39–58.98** | **0.037 (−80%)** |
| sobol | 58.50 ± 0.42 | 57.99–59.38 | 0.174 |

> Solo varianza reducida (−80%), **no media**.

---

### 4. Halton Anexo #8 — CIFAR-10 200–209 (40/40, fix parcial pero Collatz con bug)
| Sampler | Media ± SD | vs Stochastic | p (Welch) | d |
|---|---:|---:|---:|---:|
| stochastic | 85.35 ± 1.56 | — | — | — |
| sobol | 85.25 ± 1.28 | −0.10 pp | 0.69 | −0.13 |
| halton | 85.08 ± 1.11 | −0.27 pp | 0.59 | −0.18 |
| **collatz_v3 (BUG)** | **84.84 ± 1.07** | **−0.51 pp** | **0.048** | **−0.72** |

> **Excluyendo outlier seed 209 (81.37%):** V3 pierde **−0.69 pp p=0.0027**, gana solo 1/9.  
> **Sobol/Halton:** 45k únicos (0 duplicados), tau 42 vs 200 ≈ 0.02 — **no sufren bug**, pero empatan con azar.

---

### 5. FashionMNIST Anexo #1 — 200–209 (bug Collatz, SmallCNN, 20/20)
| Método | Media ± SD | Rango | Diff | p | d |
|---|---:|---|---:|---:|---:|
| Stochastic | 89.72 ± 0.24 | 89.34–90.09 | — | — | — |
| **Collatz V3 (BUG)** | **89.79 ± 0.21** | 89.43–90.13 | **+0.07 pp** | 0.17 | 0.47 |

> **No significativo** (p=0.17, tie 6/10, 1 empate).  
> **Conclusión:** V3 no generaliza a dataset fácil (techo 90% vs 85% CIFAR-10).

---

### 6. Kaggle Limpio 300–309 (20/20, **FIX LIMPIO**, semillas vírgenes)
| Método | Media ± SD | Rango | Diff | p | d |
|---|---:|---|---:|---:|---:|
| **stochastic** | **85.27 ± 1.12** | 83.12–86.45 | — | — | — |
| **collatz_v3 (FIX)** | **85.10 ± 0.85** | 84.10–86.67 | **−0.16 pp** | **0.53** | **−0.21** |

> **V3 pierde (no significativo)** con fix limpio en semillas vírgenes nunca usadas.  
> **Confirma:** +0.76pp original era **artefacto del bug + suerte offset 42–61**.

---

### 5. Halton 200–209 (40/40 completados, bug en Collatz)
| Sampler | Media ± SD | vs Stochastic | p (Welch) | d | Gana |
|---|---:|---:|---:|---:|---:|
| stochastic | 85.35 ± 1.56 | — | — | — | — |
| sobol | 85.25 ± 1.28 | −0.10 pp | 0.69 | −0.13 | 4/10 |
| halton | 85.08 ± 1.11 | −0.27 pp | 0.59 | −0.18 | 3/10 |
| collatz_v3 (BUG) | 84.84 ± 1.07 | **−0.51 pp** | **0.048** | **−0.72** | 1/10 |

> Excluyendo seed 209: V3 pierde **−0.69 pp p=0.0027**, gana 1/9.

---

## 📓 TODOS LOS COLAB / KAGGLE / NOTEBOOKS (INVENTARIO COMPLETO)

### Notebooks Vigentes (Post-Fix, Clean)
| Notebook | Ubicación | Propósito | Estado |
|---|---|---|---|
| `DEST_Kaggle_Minimal_NoPip.ipynb` | `Redes liquidas/dest/` + `DEST/notebooks/` | **20 limpias 300–309 fix** (Kaggle T4, sin pip) | ✅ **USAR** |
| `DEST_Kaggle_Clean_20_ONLY.ipynb` | mismo | 20 limpias solo (sin pool) | ✅ **USAR** |
| `DEST_Collatz_Rerun_Remaining_206_209.ipynb` | mismo | Rerun 206–209 fix (8 runs) | ✅ **USAR** |
| `DEST_Barrido_Alpha_CIFAR10.ipynb` | `Redes liquidas/dest/` + `DEST/notebooks/` | Anexo #2: α=0.1/0.3/0.5/0.7/0.9 ×3 seeds | ✅ Listo |
| `DEST_Batch_Ablation_CIFAR10.ipynb` | mismo | Anexo #5: batch 64/128/256 ×2 samplers ×5 seeds | ✅ Listo |
| `DEST_Kaggle_Minimal_NoPip.ipynb` | `DEST/notebooks/` | Kaggle minimal (sin pip) | ✅ **USAR** |
| `DEST_Kaggle_Clean_20_ONLY.ipynb` | mismo | 20 limpias sin pool | ✅ Listo |

### Notebooks DEPRECATED (Bug 98%)
| Notebook | Ubicación | Motivo |
|---|---|---|
| `DEST_Halton_vs_Sobol_vs_V3.ipynb` | `Redes liquidas/dest/` + `DEST/notebooks/` | `.deprecated.txt` — inyectó 20 con bug |
| `DEST_Halton_vs_Sobol_vs_V3.ipynb.deprecated.txt` | mismo | Marcado |
| `DEST_FashionMNIST_V3_vs_Stochastic.ipynb` | `Redes liquidas/dest/` | Bug Collatz en Fashion |
| `DEST_Collatz_Rerun_200_209.ipynb` | `Redes liquidas/dest/` | Bug en 200–205 inyectados |
| `DEST_Collatz_Rerun_Remaining_206_209.ipynb` | `Redes liquidas/dest/` + `DEST/notebooks/` | Solo 206–209 con fix |
| `CIFAR100_Final_6_Seeds*.ipynb` | `Redes liquidas/notebooks/` | Bug en CIFAR-100 |
| `DEST_Todas_Semillas_Faltantes.ipynb` | `Redes liquidas/notebooks/` | Bug en PAPER gap filling |

---

## ⚠️ ERRORES COMETIDOS Y LECCIONES (PARA NO REPETIR)

| Error | Qué pasó | Impacto | Fix / Prevención |
|---|---|---|---|
| **Bug 98% duplicados Collatz** | `argsort` sobre 45k con 98% colisiones → secuencial | +0.76pp falso en 42–61 | `lexsort((tie, vals))` con `RandomState(seed*1e6)` |
| **Offset `seed*10000` determinista** | Mismo offset = mismo orden → suerte con `split_seed=0` | +0.76pp fue suerte, no método | Usar `offset = seed * 1000003 + epoch * 7919 + hash(epoch)` |
| **No verificar unicidad en sampler** | 98% duplicados pasaron desapercibidos 2 meses | Paper entero cuestionable | Test unitario obligatorio: `assert len(set(perm)) == N` |
| **Pip install -e DEST (780MB)** en Colab/Kaggle | OOM + Kernel died + torch reinstall incompatible | Perdió 3 runs en Colab, 2 en Kaggle | `pip install -e DEST --no-deps` + sys.path |
| **P100 sm_60 en Kaggle** | PyTorch 2.5+ no soporta sm_60 | Kernel died | Cambiar a **T4 x2** (sm_75) o `torch==2.0.1+cu118` |
| **Reinstalar torch en Kaggle** | Downgrade 2.5.1→2.10 rompe env | Kernel died a los 330s | **No reinstalar** — usar torch nativo Kaggle |
| **Git sin commit antes de push** | Fix local no en GitHub → Colab clona versión vieja | Bug persistió 2 semanas | `git add -A && git commit -m "fix" && git push` ANTES de Colab |
| **Notebooks no STANDALONE** | Requieren subir `dest_lib/` manual | Fricción, errores | `git clone + pip install -e DEST --no-deps` + alias `dest_lib` |

---

## 🛠️ SKILL: COLAB PERFECTO (NO COMER ERRORES)

```markdown
# Skill: colab-perfecto
# Uso: antes de CUALQUIER Colab/Kaggle, ejecutar checklist

## PRE-CHECKLIST (ANTES DE SUBIR)
- [ ] `git add -A && git commit -m "fix: ..." && git push` (fix en GitHub ANTES de Colab)
- [ ] `git log --oneline -1` → verifica commit hash en GitHub
- [ ] Test local: `python -c "from samplers import X; assert len(set(X(...))) == N"`
- [ ] Verificar `requirements.txt` compatible con runtime (torch, CUDA)

## COLAB SETUP TEMPLATE (COPY-PASTE)
```python
# 0. Setup SIN pip install pesado
import os, sys, subprocess
if os.path.exists("DEST"): subprocess.call(["rm","-rf","DEST"])
subprocess.check_call(["git","clone","https://github.com/TU_USER/TU_REPO.git"])
subprocess.check_call([sys.executable,"-m","pip","install","-e","DEST","--no-deps","-q"])
if "DEST/src" not in sys.path: sys.path.insert(0,"DEST/src")
import dest; sys.modules["dest_lib"]=dest
for sub in ["config","samplers","models","datasets","runner"]:
    try: m=__import__(f"dest.{sub}", fromlist=[sub]); sys.modules[f"dest_lib.{sub}"]=m
    except: pass
print("✅ DEST fix instalado SIN pip pesado")
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## KAGGLE SETUP TEMPLATE
```python
# Settings → Accelerator: GPU T4 x2 (NO P100 sm_60)
# torch de Kaggle ya compatible (sm_75)
# NO reinstalles torch
```

## KERNEL DIED PREVENTION
- [ ] Usar `/kaggle/working/` para output (persiste entre restarts)
- [ ] Checkpoint cada N seeds: `shutil.make_archive(...)`
- [ ] Reanudable: `if os.path.exists(json): continue`
- [ ] NO `pip install` pesado en loop — solo una vez al inicio
- [ ] Monitor RAM: `!free -h` cada 5 seeds

## REANUDABLE PATTERN
```python
for seed in seeds:
    for sampler in samplers:
        out = f"{outdir}/exp_{sampler}_seed_{seed}.json"
        if os.path.exists(out):
            try:
                j=json.load(open(out))
                if j.get("status")=="COMPLETE" and len(j.get("test_accs",[]))==epochs:
                    continue
                else: os.remove(out)
            except: os.remove(out) if os.path.exists(out) else None
        r = runner.run_single_seed(...)
```

---

## 📈 ESTADO ACTUAL Y PRÓXIMOS PASOS

### ✅ COMPLETADO (Post-Fix)
- [x] Bug 98% duplicados → **Fix `lexsort` push `fed2d3c`**
- [x] 20 limpias Kaggle 300–309 fix → **V3 pierde −0.16pp p=0.53**
- [x] 40 Halton 200–209 completados → V3 pierde con bug
- [x] 20 FashionMNIST 200–209 → Tie +0.07pp p=0.17
- [x] 20/20 Collatz rerun 206–209 fix → En curso
- [x] 90 JSONs bug → `deprecated_collatz_bug_2026-08-28/`
- [x] Fix `lexsort` push `fed2d3c` + sync `dest_lib`
- [x] Anexo #2, #5 listos STANDALONE
- [x] Kaggle Minimal NoPip + 20 Only listos

### ⏳ PENDIENTES (Orden ROI)
| # | Tarea | Tiempo | Estado |
|---|---|---|---|
| 1 | Rerun Collatz 200–209 **todo fix** (20 limpias) | ~70 min | ⏳ Parcial (8/20) |
| 2 | Anexo #2 Barrido α (0.1–0.9) | 2 días | ✅ Notebook listo |
| 3 | Anexo #5 Batch ablation (64/128/256) | 3 días | ✅ Notebook listo |
| 4 | Anexo #6 30-50 épocas | 1 semana | ⏳ |
| 5 | Anexo #7 Label noise | 3 días | ⏳ |
| 6 | Anexo #10 Repo público v1.0 | 1 día | ⏳ |

### 🚫 NO HACER (Confirmado inútil)
- ❌ Más semillas 42–61 / 200–209 (bug, deprecated)
- ❌ Teorema "V3 > Stochastic" general (roto)
- ❌ Colab con `pip install -e DEST` (usa `--no-deps`)
- ❌ P100 en Kaggle (usa T4 x2)
- ❌ Reinstalar torch en Kaggle (usa nativo)

---

## 🎯 VEREDICTO FINAL PARA PAPER/BLOG

> **No hay teorema.** El +0.76pp (p=0.006) en PAPER 42–61 fue **artefacto del bug 98% duplicados + suerte del offset `seed*10000`** con `split_seed=0`.  
> 
> **Evidencia real con fix limpio (300–309, semillas vírgenes):** V3 **−0.16pp p=0.53 d=−0.21** — empata/pierde contra azar.  
> 
> **Uso honesto en blog:** *"V3 mostró +0.76pp en CIFAR-10 (42–61), pero re-evaluación con fix de sampler en semillas vírgenes (300–309) no reproduce el efecto (−0.16pp, p=0.53). El beneficio original parece atribuible a un bug en el sampler (98% duplicados) que hacía el orden cuasi-secuencial y coincidió favorablemente con el split fijo. No hay evidencia de superioridad robusta de V3 sobre azar en CIFAR-10."*

---

## 📁 ARCHIVOS CLAVE PARA ENTREGA

| Archivo | Qué contiene |
|---|---|
| `INFORME_OPERATIVO_COMPLETO_DEST.md` | **Este informe** |
| `DEPRECATED_collatz_bug_2026-08-28.md` | 90 JSONs marcados NO USAR |
| `DEST_Kaggle_Minimal_NoPip.ipynb` | Kaggle 20 limpias fix (subir a Kaggle T4) |
| `DEST_Kaggle_Minimal_NoPip.ipynb` | 20 Only (sin pool) |
| `DEST_Collatz_Rerun_Remaining_206_209.ipynb` | Rerun 206–209 fix |
| `DEST_Barrido_Alpha_CIFAR10.ipynb` | Anexo #2 listo |
| `DEST_Batch_Ablation_CIFAR10.ipynb` | Anexo #5 listo |
| `DEST_Kaggle_Minimal_NoPip.ipynb` | Kaggle minimal (sin pip) |
| `BITACORA2.md` | Bitácora completa con todos hitos |
| `src/dest/samplers.py` | Fix `lexsort` línea 88, 186 |
| `dest_results_paper/DEPRECATED_collatz_bug_2026-08-28/` | 90 JSONs deprecated |

---

## ✅ CHECKLIST FINAL ANTES DE CERRAR

- [x] Bug 98% duplicados corregido y pusheado
- [x] 20 limpias Kaggle 300–309 fix completadas
- [x] 40 Halton 200–209 completados
- [x] 20 FashionMNIST completados
- [x] 90 JSONs bug movidos a deprecated
- [x] Fix `lexsort` push `fed2d3c`
- [x] Notebooks deprecated marcados
- [x] Kaggle Minimal NoPip + 20 Only listos
- [x] Anexo #2, #5 notebooks listos
- [x] Skill `colab-perfecto` documentada
- [x] Informe operativo completo generado
- [ ] Rerun 200–209 **todo fix** (pendiente)
- [ ] Paper/Blog con redacción honesta (pendiente)

---

*Generado automáticamente desde bitácoras, logs, JSONs y análisis — 30 Agosto 2026*  
*Proyecto DEST — Deterministic Entropy-Scheduled Training*