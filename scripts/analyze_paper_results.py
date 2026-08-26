# shim module for backward compatibility
from importlib import import_module
_mod = import_module('analyze_paper_results')
for k, v in vars(_mod).items():
    if not k.startswith('__'):
        globals()[k] = v
