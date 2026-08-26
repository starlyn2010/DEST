import os
import json

def main():
    lib_modules = {}
    lib_path = 'dest_lib'
    if os.path.exists(lib_path):
        for fname in sorted(os.listdir(lib_path)):
            if fname.endswith('.py'):
                with open(os.path.join(lib_path, fname), 'r') as fh:
                    lib_modules[fname] = fh.read()

    # Diseno PAREADO limpio: 10 semillas nuevas (200-209) x 5 metodos.
    # Rango 200+ para que no se mezclen con los runs viejos (42-61).
    METODOS = ["stochastic", "sobol", "collatz_v1", "collatz_v3", "collatz_v2"]
    FALTANTES = []
    for s in range(200, 210):
        for m in METODOS:
            FALTANTES.append(["CIFAR100", m, s])

    lib_json = json.dumps(lib_modules, indent=4)
    faltantes_json = json.dumps(FALTANTES, indent=4)

    # -- Celda 1: deps + dest_lib --
    cell1 = "\n".join([
        "import os, sys, json, zipfile, torch, subprocess, importlib",
        "",
        "try:",
        "    import seaborn, sklearn, tqdm",
        "except ImportError:",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',",
        "                           'seaborn', 'scikit-learn', 'tqdm', 'matplotlib', '-q'])",
        "",
        f"lib_files = {lib_json}",
        "",
        "os.makedirs('dest_lib', exist_ok=True)",
        "for filename, content in lib_files.items():",
        "    fpath = os.path.join('dest_lib', filename)",
        "    with open(fpath, 'w', encoding='utf-8') as fh:",
        "        fh.write(content)",
        "",
        "created = sorted(f for f in os.listdir('dest_lib') if f.endswith('.py'))",
        "print('Archivos en dest_lib:', created)",
        "assert 'config.py' in created and 'runner.py' in created",
        "",
        "importlib.invalidate_caches()",
        "for mod_name in list(sys.modules.keys()):",
        "    if 'dest_lib' in mod_name:",
        "        del sys.modules[mod_name]",
        "",
        "cwd = os.path.abspath('.')",
        "if cwd not in sys.path:",
        "    sys.path.insert(0, cwd)",
        "",
        "from dest_lib.config import get_config",
        "from dest_lib.runner import ExperimentRunner",
        "print('Imports OK')",
        "",
        "CONFIG = get_config('PAPER')",
        "CONFIG['output_dir'] = os.path.join(cwd, 'dest_results_paper')",
        "os.makedirs(CONFIG['output_dir'], exist_ok=True)",
        "runner = ExperimentRunner(CONFIG)",
        "",
        "print(f'PyTorch: {torch.__version__}')",
        "print(f'CUDA: {torch.cuda.is_available()}')",
        "if torch.cuda.is_available():",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')",
    ])

    # -- Celda 2: entrenamiento pareado 200-209 --
    cell2 = "\n".join([
        f"FALTANTES = {faltantes_json}",
        "",
        "print(f'Runs pendientes: {len(FALTANTES)}  |  semillas 200-209 x 5 metodos (orden pareado)')",
        "",
        "def _path(ds, samp, seed):",
        "    return os.path.join(CONFIG['output_dir'], f'{ds}_{samp}_{samp}_seed_{seed}.json')",
        "",
        "# Reanudable: si Colab se cae, Celda 1 + esta otra vez; lo guardado se salta.",
        "# Cada bloque de semilla son 5 runs (~2.5 h): al cerrar uno, ya tienes un set pareado completo.",
        "for idx, (ds, sampler, seed) in enumerate(FALTANTES, 1):",
        "    if os.path.exists(_path(ds, sampler, seed)):",
        "        print(f'[{idx}/{len(FALTANTES)}] {ds} {sampler} seed {seed}: YA EXISTE -> salto')",
        "        continue",
        "    print('\\n' + '='*60)",
        f"    print(f'[{{idx}}/{{len(FALTANTES)}}] DS={{ds}} | {{sampler}} | SEED={{seed}}')",
        "    print('='*60)",
        "    res = runner.run_single_seed(",
        "        exp_id=f'{ds}_{sampler}',",
        "        sampler_name=sampler,",
        "        seed=seed,",
        "        dataset=ds,",
        "        dropout_mode='deterministic',",
        "    )",
        "    print(f'Semilla {seed} ({sampler}) completada. Acc: {res.final_test_acc:.2f}%')",
        "",
        "print('\\n===== ANALISIS PAREADO (semillas 200-209, quintetos completos) =====')",
        "import glob, json, statistics as st_, math",
        "",
        "METODOS = ['stochastic', 'sobol', 'collatz_v1', 'collatz_v3', 'collatz_v2']",
        "",
        "def _cargar():",
        "    mat = {}",
        "    for f_ in glob.glob(os.path.join(CONFIG['output_dir'], 'CIFAR100_*_seed_*.json')):",
        "        with open(f_) as fh_:",
        "            d_ = json.load(fh_)",
        "        mat.setdefault(d_['seed'], {})[d_['sampler_name']] = d_['final_test_acc']",
        "    return {s_: m_ for s_, m_ in sorted(mat.items())",
        "            if len(m_) == len(METODOS) and set(m_) == set(METODOS)}",
        "",
        "matriz = _cargar()",
        "print(f'Semillas pareadas completas: {sorted(matriz.keys())}')",
        "",
        "if not matriz:",
        "    print('Todavia no hay semillas completas.')",
        "else:",
        "    print('\\nTabla de accuracy por semilla:')",
        "    print('   seed | ' + ' | '.join(f'{m_:>11s}' for m_ in METODOS))",
        "    for s_, m_ in matriz.items():",
        "        print(f'   {s_:4d} | ' + ' | '.join(f'{m_[k_]:11.2f}' for k_ in METODOS))",
        "    print('\\nComparacion pareada vs collatz_v3 (diff por semilla):')",
        "    for base_ in ['stochastic', 'sobol', 'collatz_v1', 'collatz_v2']:",
        "        ds_ = [matriz[s_]['collatz_v3'] - matriz[s_][base_] for s_ in sorted(matriz)]",
        "        n_ = len(ds_)",
        "        md_ = st_.mean(ds_)",
        "        sd_ = st_.stdev(ds_) if n_ > 1 else float('nan')",
        "        t_ = md_ / (sd_ / math.sqrt(n_)) if n_ > 1 and sd_ else float('nan')",
        "        w_ = sum(1 for x_ in ds_ if x_ > 0)",
        "        print(f'  collatz_v3 - {base_:10s}: dMedia={md_:+.3f}pp  t_pareado={t_:+.2f}  v3_gana_{w_}/{n_}')",
        "        print('      diffs:', [f'{x_:+.2f}' for x_ in ds_])",
        "",
        "    print('\\n===== MEDICION DE TIEMPOS (mismas semillas pareadas) =====')",
        "    def _cargar_tiempos():",
        "        tm_ = {}",
        "        for f_ in glob.glob(os.path.join(CONFIG['output_dir'], 'CIFAR100_*_seed_*.json')):",
        "            with open(f_) as fh_:",
        "                d_ = json.load(fh_)",
        "            if d_['seed'] not in matriz:",
        "                continue",
        "            tm_.setdefault(d_['seed'], {})[d_['sampler_name']] = d_",
        "        return tm_",
        "",
        "    tmat_ = _cargar_tiempos()",
        "    print(f'   {\"metodo\":12s} | {\"sampler s/epoca\":>15s} | {\"train s/epoca\":>14s} | {\"runtime total\":>13s} | {\"samples/s\":>10s}')",
        "    resumen_t_ = {}",
        "    for m_ in METODOS:",
        "        st_e_  = [st_.mean(tmat_[s_][m_]['sampler_time_per_epoch']) for s_ in sorted(tmat_)]",
        "        tr_e_  = [st_.mean(tmat_[s_][m_]['train_time_per_epoch'])  for s_ in sorted(tmat_)]",
        "        tot_   = [tmat_[s_][m_]['total_runtime_seconds']           for s_ in sorted(tmat_)]",
        "        sps_   = [st_.mean(tmat_[s_][m_]['samples_per_second'])    for s_ in sorted(tmat_)]",
        "        resumen_t_[m_] = (st_.mean(st_e_), st_.mean(tr_e_), st_.mean(tot_), st_.mean(sps_))",
        "        print(f'   {m_:12s} | {st_.mean(st_e_):15.4f} | {st_.mean(tr_e_):14.2f} | {st_.mean(tot_):13.1f} | {st_.mean(sps_):10.0f}')",
        "",
        "    base_t_ = resumen_t_['stochastic']",
        "    print('\\n   Overhead relativo vs stochastic (mismo numero de batches en todos):')",
        "    for m_, (se_, tr_, tt_, sp_) in resumen_t_.items():",
        "        print(f'   {m_:12s}: sampler x{se_/base_t_[0]:6.2f} | entrenamiento total x{tt_/base_t_[2]:6.3f}')",
    ])

    # -- Celda 3: zip y descarga --
    cell3 = "\n".join([
        "zip_filename = 'resultados_DEST_cifar100_seeds200.zip'",
        "print(f'Empaquetando en {zip_filename}...')",
        "with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:",
        "    for root, dirs, fs in os.walk(CONFIG['output_dir']):",
        "        for file in fs:",
        "            if not file.endswith('.zip'):",
        "                fpath = os.path.join(root, file)",
        "                arcname = os.path.relpath(fpath, CONFIG['output_dir'])",
        "                zipf.write(fpath, arcname)",
        "size_mb = os.path.getsize(zip_filename) / 1024 / 1024",
        "print(f'ZIP listo: {zip_filename} ({size_mb:.2f} MB)')",
        "try:",
        "    from google.colab import files",
        "    files.download(zip_filename)",
        "    print('Descarga iniciada.')",
        "except ImportError:",
        "    print(f'Descarga manual: {zip_filename}')",
    ])

    ok = True
    for i, src in enumerate([cell1, cell2, cell3], 1):
        try:
            compile(src, f"celda_{i}", "exec")
            print(f"Celda {i}: OK")
        except SyntaxError as e:
            print(f"Celda {i}: ERROR -> {e}")
            ok = False
    if not ok:
        print("ABORTADO.")
        return

    md_intro = (
        "# DEST - CIFAR100 Pareado (semillas 200-209, 5 metodos)\n"
        "Experimento limpio y aislado: **10 semillas nuevas (200-209) x 5 samplers**.\n"
        "Rango 200+ para que ninguna se confunda con los runs viejos (42-61).\n"
        "\n"
        "Misma seed => mismos pesos iniciales, mismo dropout determinista y mismo split\n"
        "(split_seed=0); **lo unico distinto entre metodos es el orden del sampler**.\n"
        "\n"
        "| Dato | Valor |\n"
        "|---|---|\n"
        "| Runs totales | 50 |\n"
        "| Por semilla | 5 runs (~2.5 h) |\n"
        "| Total estimado | ~25 h de T4 |\n"
        "\n"
        "**Reanudable:** Colab se cayo -> Celda 1 + Celda 2 otra vez; lo guardado se salta.\n"
        "Puedes parar tras cualquier semilla: cada bloque cerrado es un set pareado completo.\n"
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md_intro},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3},
        ]
    }

    os.makedirs('notebooks', exist_ok=True)
    out_path = os.path.join("notebooks", "DEST_CIFAR100_Pareado_Semillas200.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"\nNotebook listo: {out_path}")

if __name__ == "__main__":
    main()
