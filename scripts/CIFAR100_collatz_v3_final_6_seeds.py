import os, sys, json, zipfile, torch

# ── Auto-instalar dependencias ──────────────────────────────
import subprocess
try:
    import seaborn, sklearn, tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "scikit-learn", "tqdm", "matplotlib", "-q"])

# ── Inyectar dest_lib ───────────────────────────────────────
lib_files = {
    "__init__.py": "# shim package __init__.py for backward compatibility\nfrom importlib import import_module\n_mod = import_module('dest.dest_lib')\nfor k, v in vars(_mod).items():\n    if not k.startswith('__'):\n        globals()[k] = v\n"
}

os.makedirs('dest_lib', exist_ok=True)
for filename, content in lib_files.items():
    with open(os.path.join('dest_lib', filename), 'w', encoding='utf-8') as fh:
        fh.write(content)
print("✅ dest_lib lista.")

sys.path.insert(0, os.path.abspath('.'))
from dest_lib.config import get_config
from dest_lib.runner import ExperimentRunner

CONFIG = get_config('PAPER')
CONFIG['output_dir'] = os.path.join(os.path.abspath('.'), 'dest_results_paper')
os.makedirs(CONFIG['output_dir'], exist_ok=True)
runner = ExperimentRunner(CONFIG)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Semillas que faltan ─────────────────────────────────────
FALTANTES = [
    [
        "CIFAR100",
        "collatz_v3",
        46
    ],
    [
        "CIFAR100",
        "collatz_v3",
        47
    ],
    [
        "CIFAR100",
        "collatz_v3",
        48
    ],
    [
        "CIFAR100",
        "collatz_v3",
        49
    ],
    [
        "CIFAR100",
        "collatz_v3",
        50
    ],
    [
        "CIFAR100",
        "collatz_v3",
        51
    ]
]

print(f"\nTotal semillas a completar: {len(FALTANTES)} (solo collatz_v3 seeds 46-51)")
print("="*60)

for ds, sampler, seed in FALTANTES:
    print(f"\n🚀 DS={ds} | SAMPLER={sampler} | SEED={seed}")
    print("="*60)

    res = runner.run_single_seed(
        exp_id=f"{ds}_{sampler}",
        sampler_name=sampler,
        seed=seed,
        dataset=ds,
        dropout_mode='deterministic',  # collatz_v3 siempre usa deterministic
    )
    print(f"✅ Semilla {seed} completada. Acc final: {res.final_test_acc:.2f}%")

# ── Empaquetar resultados ───────────────────────────────────
zip_filename = "resultados_CIFAR100_Final.zip"
print(f"\nEmpaquetando resultados en {zip_filename}...")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(CONFIG['output_dir']):
        for file in files:
            if not file.endswith('.zip'):
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, CONFIG['output_dir'])
                zipf.write(fpath, arcname)

size_mb = os.path.getsize(zip_filename) / 1024 / 1024
print(f"✅ ZIP listo: {zip_filename} ({size_mb:.2f} MB)")

try:
    from google.colab import files
    files.download(zip_filename)
    print("⬇️  Descarga iniciada automáticamente.")
except ImportError:
    print(f"Descarga manual: {zip_filename}")
