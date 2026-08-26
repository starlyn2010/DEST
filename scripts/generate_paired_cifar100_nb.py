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

    # Diseno PAREADO: para cada semilla se corren TODOS los metodos seguidos.
    # Con la misma seed hay mismos pesos iniciales, mismo dropout determinista y
    # mismo split (split_seed=0); lo unico distinto es el orden del sampler.
    # Bloque 1 (prioridad): quinteto completo para seeds 52-61 -> n=20 en 5 metodos
    # Bloque 2: collatz_v2 en seeds 42-51 (el unico metodo sin historial en CIFAR100)
    METODOS = ["stochastic", "sobol", "collatz_v1", "collatz_v3"]
    FALTANTES = []
    for s in range(52, 62):
        for m in METODOS + ["collatz_v2"]:
            FALTANTES.append(["CIFAR100", m, s])
    for s in range(42, 52):
        FALTANTES.append(["CIFAR100", "collatz_v2", s])

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

    # -- Celda 2: entrenamiento pareado con reanudacion --
    cell2 = "\n".join([
        f"FALTANTES = {faltantes_json}",
        'METODOS_BLOQUE = ["stochastic", "sobol", "collatz_v1", "collatz_v3", "collatz_v2"]',
        "",
        "print(f'Runs pendientes: {len(FALTANTES)} (orden pareado: semilla -> 5 metodos)')",
        "",
        "def _path(ds, samp, seed):",
        "    return os.path.join(CONFIG['output_dir'], f'{ds}_{samp}_{samp}_seed_{seed}.json')",
        "",
        "# Reanudable: si Colab se cae, Celda 1 + esta otra vez; lo hecho se salta.",
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
        "print('\\n===== ANALISIS PAREADO (semillas con los 5 metodos) =====')",
        "import glob, json, statistics as st_, math",
        "",
        "def _cargar():",
        "    mat = {}",
        "    g_ = glob.glob(os.path.join(CONFIG['output_dir'], 'CIFAR100_*_seed_*.json'))",
        "    for f_ in g_:",
        "        with open(f_) as fh_:",
        "            d_ = json.load(fh_)",
        "        mat.setdefault(d_['seed'], {})[d_['sampler_name']] = d_['final_test_acc']",
        "    return {s_: m_ for s_, m_ in sorted(mat.items()) if len(m_) == 5}",
        "",
        "matriz = _cargar()",
        "print(f'Semillas con quinteto completo: {sorted(matriz.keys())}')",
        "",
        "if not matriz:",
        "    print('Todavia no hay semillas con los 5 metodos completos.')",
        "else:",
        "    def _pareado(a_, b_):",
        "        ds_ = [matriz[s_][a_] - matriz[s_][b_] for s_ in sorted(matriz)]",
        "        n_ = len(ds_)",
        "        md_ = st_.mean(ds_)",
        "        sd_ = st_.stdev(ds_) if n_ > 1 else float('nan')",
        "        t_ = md_ / (sd_ / math.sqrt(n_)) if n_ > 1 and sd_ else float('nan')",
        "        wins_ = sum(1 for x_ in ds_ if x_ > 0)",
        "        return ds_, md_, t_, wins_",
        "",
        "    for base_ in ['stochastic', 'sobol']:",
        "        ds_, md_, t_, w_ = _pareado('collatz_v3', base_)",
        "        print(f'collatz_v3 - {base_:10s}: dMedia={md_:+.3f}pp  t_pareado={t_:+.2f}  v3_gana_{w_}/{len(ds_)}')",
        "        print('   diffs por seed:', [f'{x_:+.2f}' for x_ in ds_])",
    ])

    # -- Celda 3: zip y descarga --
    cell3 = "\n".join([
        "zip_filename = 'resultados_DEST_cifar100_pareado.zip'",
        "print(f'Empaquetando en {zip_filename}...')",
        "with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:",
        "    for root, dirs, files in os.walk(CONFIG['output_dir']):",
        "        for file in files:",
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
        "# DEST - CIFAR100 Pareado (seeds 52-61, 5 metodos)\n"
        "Diseno **pareado**: para cada semilla se corren los 5 samplers seguidos.\n"
        "Con la misma seed: mismos pesos iniciales, mismo dropout determinista y mismo split\n"
        "(`split_seed=0`); **lo unico distinto entre metodos es el orden del sampler**.\n"
        "\n"
        "| Bloque | Contenido | Runs |\n"
        "|---|---|---|\n"
        "| 1 | seeds 52-61 x 5 metodos | 50 |\n"
        "| 2 | collatz_v2 seeds 42-51 (catch-up) | 10 |\n"
        "\n"
        "**Reanudable:** si se cae Colab, Celda 1 + Celda 2 de nuevo; lo guardado se salta.\n"
        "Puedes detenerte al terminar cualquier bloque de semilla: cada uno cierra un set pareado completo.\n"
        "Al final la Celda 2 imprime el analisis pareado (diff por seed, t pareado, victorias).\n"
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
    out_path = os.path.join("notebooks", "DEST_CIFAR100_Pareado_52_61.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"\nNotebook listo: {out_path}")

if __name__ == "__main__":
    main()
