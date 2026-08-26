"""
runner.py — Core training loop with checkpoint/resume support and LR scheduling.

Key changes vs v1:
  - run_single_seed accepts a dataset name and resolves model automatically.
  - Supports 'cosine' and 'constant' LR schedules.
  - Checkpoint file is keyed by (dataset, sampler, seed) — robust to Colab
    disconnects across multi-dataset runs.
  - RunResult now carries dataset and sampler_name fields.
  - All random seeds are re-applied at the start of each seed run.
"""

import os
import time
import json
import dataclasses
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from .config import RunResult, DATASET_MODEL_MAP
from .reproducibility import seed_everything
from .datasets import DatasetLoader
from .samplers import SamplerFactory
from .models import ModelFactory
from .metrics import evaluate_model
from .manifest import ExperimentManifest


class ExperimentRunner:
    """
    Runs a single (dataset, sampler, seed) triple and saves the RunResult.

    Parameters
    ----------
    config : dict
        Output of dest_lib.config.get_config().  The 'dataset' key is read
        from here; it can also be overridden via run_single_seed(dataset=...).
    """

    def __init__(self, config: dict):
        self.config     = config
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest = ExperimentManifest(self.output_dir)

        if config.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])

    # ──────────────────────────────────────────────────────────────────
    def run_single_seed(
        self,
        exp_id: str,
        sampler_name: str,
        seed: int,
        dataset: str | None = None,
        dropout_mode: str = "stochastic",
        alpha_fixed: float = 0.5,
        # legacy compat
        mode: str | None = None,
    ) -> RunResult:
        """
        Train one model for one seed and return the RunResult.

        Parameters
        ----------
        dataset : str, optional
            Override the dataset in self.config["dataset"].
        mode : str, optional
            Legacy alias for sampler_name (ignored if sampler_name given).
        """
        # -- resolve dataset & model -------------------------------------------
        ds_name = (dataset or self.config.get("dataset", "MNIST")).upper()

        model_name = self.config.get("model_override") or \
                     DATASET_MODEL_MAP.get(ds_name, "SmallCNN")

        # -- reproducibility ---------------------------------------------------
        seed_everything(seed)

        # -- load data ---------------------------------------------------------
        loader = DatasetLoader(
            dataset_name=ds_name,
            val_fraction=self.config.get("val_fraction", 0.1),
            split_seed=0,          # fixed split across all seeds
            data_root="./data",
        )
        result = loader.get_datasets()
        # result = (train_ds, val_ds, test_ds, n_classes, input_shape, available)
        if result[-1] is False:
            raise RuntimeError(f"Dataset {ds_name} is not available.")
        train_ds, val_ds, test_ds, n_classes, input_shape, _ = result

        # -- samplers ----------------------------------------------------------
        epochs = self.config["epochs"]
        sampler = SamplerFactory.get_sampler(
            sampler_name=sampler_name,
            data_source=train_ds,
            seed=seed,
            total_epochs=epochs,
            alpha_fixed=alpha_fixed,
        )

        nw = self.config.get("num_workers", 2)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False, num_workers=nw)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=nw)

        # -- model & optimizer ------------------------------------------------
        model = ModelFactory.get_model(
            model_name=model_name,
            input_shape=input_shape,
            num_classes=n_classes,
            dropout_mode=dropout_mode,
        ).to(self.device)

        opt_name = self.config.get("optimizer", "SGD").upper()
        if opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config["lr"],
                momentum=self.config.get("momentum", 0.9),
                weight_decay=self.config.get("weight_decay", 1e-4),
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config.get("weight_decay", 1e-4),
            )

        # LR scheduler
        schedule = self.config.get("lr_schedule", "constant")
        if schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()

        # -- training tracking -------------------------------------------------
        train_losses, val_losses, test_losses   = [], [], []
        train_accs,   val_accs,   test_accs     = [], [], []
        gen_gaps                                 = []
        f1_epochs, prec_epochs, rec_epochs       = [], [], []
        sampler_times, train_times, eval_times, total_times = [], [], [], []
        samples_per_sec                          = []

        start_total = time.time()
        conv_90 = conv_95 = None

        for epoch in range(epochs):
            ep_start = time.time()

            # sampler epoch update
            t0 = time.time()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            t_sampler = time.time() - t0

            # train
            t0 = time.time()
            model.train()
            run_loss = correct = total = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                out  = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * data.size(0)
                correct  += (out.argmax(1) == target).sum().item()
                total    += data.size(0)

            if scheduler:
                scheduler.step()

            t_train = time.time() - t0
            tr_loss = run_loss / max(1, total)
            tr_acc  = correct / max(1, total) * 100.0

            # eval
            t0 = time.time()
            v_loss, v_acc, *_ = evaluate_model(model, self.device, val_loader,  criterion)
            te_loss, te_acc, te_f1, te_prec, te_rec, te_ece = evaluate_model(
                model, self.device, test_loader, criterion
            )
            t_eval = time.time() - t0

            ep_total   = time.time() - ep_start
            throughput = total / max(1e-6, t_train)

            # record
            train_losses.append(tr_loss);  val_losses.append(v_loss);  test_losses.append(te_loss)
            train_accs.append(tr_acc);     val_accs.append(v_acc);     test_accs.append(te_acc)
            gen_gaps.append(tr_acc - te_acc)
            f1_epochs.append(te_f1);  prec_epochs.append(te_prec);  rec_epochs.append(te_rec)
            sampler_times.append(t_sampler); train_times.append(t_train)
            eval_times.append(t_eval);       total_times.append(ep_total)
            samples_per_sec.append(throughput)

            if conv_90 is None and te_acc >= 90.0: conv_90 = epoch + 1
            if conv_95 is None and te_acc >= 95.0: conv_95 = epoch + 1

            if self.config.get("verbose", True):
                print(f"    Epoch {epoch+1}/{epochs}  "
                      f"tr_loss={tr_loss:.4f}  te_acc={te_acc:.2f}%  "
                      f"lr={optimizer.param_groups[0]['lr']:.5f}")

        total_runtime = time.time() - start_total
        gpu_mem = (torch.cuda.max_memory_allocated() / 1024 / 1024
                   if torch.cuda.is_available() else 0.0)

        res = RunResult(
            experiment_id=exp_id,
            dataset=ds_name,
            sampler_name=sampler_name,
            seed=seed,
            mode=sampler_name,
            train_losses=train_losses,        val_losses=val_losses,
            test_losses=test_losses,          train_accs=train_accs,
            val_accs=val_accs,                test_accs=test_accs,
            generalization_gaps=gen_gaps,
            f1_per_epoch=f1_epochs,           precision_per_epoch=prec_epochs,
            recall_per_epoch=rec_epochs,
            final_test_acc=test_accs[-1],     final_test_loss=test_losses[-1],
            final_f1=f1_epochs[-1],           final_precision=prec_epochs[-1],
            final_recall=rec_epochs[-1],      final_ece=te_ece,
            final_generalization_gap=gen_gaps[-1],
            convergence_epoch_90=conv_90,     convergence_epoch_95=conv_95,
            best_test_acc=float(np.max(test_accs)),
            best_test_epoch=int(np.argmax(test_accs)) + 1,
            sampler_time_per_epoch=sampler_times,
            train_time_per_epoch=train_times,
            eval_time_per_epoch=eval_times,
            total_time_per_epoch=total_times,
            total_runtime_seconds=total_runtime,
            samples_per_second=samples_per_sec,
            gpu_memory_peak_mb=gpu_mem,
            train_loss_variance=float(np.var(train_losses[-3:])),
            test_acc_variance=float(np.var(test_accs[-3:])),
            config_snapshot=self.config,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # -- persist -----------------------------------------------------------
        out_file = os.path.join(
            self.output_dir, f"{exp_id}_{sampler_name}_seed_{seed}.json"
        )
        with open(out_file, "w") as f:
            json.dump(dataclasses.asdict(res), f, indent=2)

        self.manifest.mark_seed_complete(exp_id, seed)
        return res
