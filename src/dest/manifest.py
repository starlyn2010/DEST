import os
import json
import time
import datetime
import torch

class ExperimentManifest:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.manifest_path = os.path.join(output_dir, "experiment_manifest.json")
        self.manifest = self._load()

    def _load(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"experiments": {}, "system_info": self._get_sys_info()}
        return {"experiments": {}, "system_info": self._get_sys_info()}

    def _get_sys_info(self):
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__
        }

    def save(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)

    def is_experiment_complete(self, exp_id: str, n_seeds: int) -> bool:
        if exp_id in self.manifest["experiments"]:
            exp_data = self.manifest["experiments"][exp_id]
            return len(exp_data.get("completed_seeds", [])) >= n_seeds
        return False

    def is_seed_complete(self, exp_id: str, seed: int) -> bool:
        if exp_id in self.manifest["experiments"]:
            return seed in self.manifest["experiments"][exp_id].get("completed_seeds", [])
        return False

    def mark_seed_complete(self, exp_id: str, seed: int):
        if exp_id not in self.manifest["experiments"]:
            self.manifest["experiments"][exp_id] = {"completed_seeds": [], "status": "IN_PROGRESS"}
        if seed not in self.manifest["experiments"][exp_id]["completed_seeds"]:
            self.manifest["experiments"][exp_id]["completed_seeds"].append(seed)
        self.save()

    def mark_experiment_complete(self, exp_id: str):
        if exp_id in self.manifest["experiments"]:
            self.manifest["experiments"][exp_id]["status"] = "COMPLETED"
        else:
            self.manifest["experiments"][exp_id] = {"completed_seeds": [], "status": "COMPLETED"}
        self.save()
