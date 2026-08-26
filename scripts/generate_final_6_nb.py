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

    FALTANTES = [
        ["CIFAR100", "collatz_v3", 46],
        ["CIFAR100", "collatz_v3", 47],
        ["CIFAR100", "collatz_v3", 48],
        ["CIFAR100", "collatz_v3", 49],
        ["CIFAR100", "collatz_v3", 50],
        ["CIFAR100", "collatz_v3", 51],
    ]

    lib_json = json.dumps(lib_modules, indent=4)
    faltantes_json = json.dumps(FALTANTES, indent=4)

    # ── Celda 1: instalar deps + inyectar librería ──────────────────
    cell1 = "\n".join([
        "import os, sys, json, zipfile, torch, subprocess, importlib",
        "",
        "# 1. Instalar dependencias opcionales que Colab no trae",
        "try:",
        "    import seaborn, sklearn, tqdm",
        "except ImportError:",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',",
        "                           'seaborn', 'scikit-learn', 'tqdm', 'matplotlib', '-q'])",
        "",
        "# 2. Escribir dest_lib en disco",
        f"lib_files = {lib_json}",
        "",
        "os.makedirs('dest_lib', exist_ok=True)",
        "for filename, content in lib_files.items():",
        "    fpath = os.path.join('dest_lib', filename)",
        "    with open(fpath, 'w', encoding='utf-8') as fh:",
        "        fh.write(content)",
        "",
        "# 3. Verificar que los archivos existen",
        "created = sorted(os.listdir('dest_lib'))",
        "print('Archivos en dest_lib:', created)",
        "assert 'config.py' in created, 'ERROR: config.py no fue creado!'",
        "assert 'runner.py' in created, 'ERROR: runner.py no fue creado!'",
        "",
        "# 4. Limpiar cache de Python para que detecte los archivos nuevos",
        "importlib.invalidate_caches()",
        "for mod_name in list(sys.modules.keys()):",
        "    if 'dest_lib' in mod_name:",
        "        del sys.modules[mod_name]",
        "",
        "# 5. Agregar directorio actual al path e importar",
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

    # ── Celda 2: loop de entrenamiento ──────────────────────────────
    cell2 = "\n".join([
        f"FALTANTES = {faltantes_json}",
        "",
        "print(f'Semillas a entrenar: {len(FALTANTES)} (collatz_v3 seeds 46-51)')",
        "",
        "for ds, sampler, seed in FALTANTES:",
        "    print('\\n' + '='*60)",
        "    print(f'DS={ds} | SAMPLER={sampler} | SEED={seed}')",
        "    print('='*60)",
        "    res = runner.run_single_seed(",
        "        exp_id=f'{ds}_{sampler}',",
        "        sampler_name=sampler,",
        "        seed=seed,",
        "        dataset=ds,",
        "        dropout_mode='deterministic',",
        "    )",
        "    print(f'Semilla {seed} completada. Acc: {res.final_test_acc:.2f}%')",
    ])

    # ── Celda 3: zip y descarga ─────────────────────────────────────
    cell3 = "\n".join([
        "zip_filename = 'resultados_CIFAR100_Final.zip'",
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

    # ── Verificar sintaxis ──────────────────────────────────────────
    ok = True
    for i, src in enumerate([cell1, cell2, cell3], 1):
        try:
            compile(src, f"celda_{i}", "exec")
            print(f"Celda {i}: OK")
        except SyntaxError as e:
            print(f"Celda {i}: ERROR -> {e}")
            ok = False

    if not ok:
        print("ABORTADO: hay errores de sintaxis.")
        return

    # ── Construir notebook ──────────────────────────────────────────
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# DEST — CIFAR100 collatz_v3 (6 seeds finales)\nSolo las 6 semillas que faltan (46-51). Ejecuta las 3 celdas en orden."
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3},
        ]
    }

    out_path = "CIFAR100_Final_6_Seeds.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"\nNotebook listo: {out_path}")

if __name__ == "__main__":
    main()
