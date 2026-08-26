# shim module for backward compatibility
from importlib import import_module
_mod = import_module('run_strict_validation')
for k, v in vars(_mod).items():
    if not k.startswith('__'):
        globals()[k] = v
