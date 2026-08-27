import numpy as np
import torch
from torch.utils.data import Sampler

class StochasticSampler(Sampler):
    def __init__(self, data_source, seed=42):
        super().__init__()
        self.num_samples = len(data_source)
        self.seed = seed
        self.epoch = 0
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self):
        indices = torch.randperm(self.num_samples, generator=self.generator).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class SequentialBaseSampler(Sampler):
    def __init__(self, data_source):
        super().__init__()
        self.num_samples = len(data_source)
    def set_epoch(self, epoch):
        pass
    def __iter__(self):
        return iter(range(self.num_samples))
    def __len__(self):
        return self.num_samples

class SobolPermutationSampler(Sampler):
    def __init__(self, data_source, seed=42):
        super().__init__()
        self.num_samples = len(data_source)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        current_seed = self.seed + self.epoch
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=1, scramble=True, seed=current_seed)
            sobol_points = sampler.random(n=self.num_samples).flatten()
            indices = np.argsort(sobol_points).tolist()
        except ImportError:
            a = 1664525
            c = 1013904223
            m = 2**32
            val = current_seed
            sequence = []
            for _ in range(self.num_samples):
                val = (a * val + c) % m
                sequence.append(val)
            indices = np.argsort(sequence).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class CollatzFix1Sampler(Sampler):
    def __init__(self, data_source, seed=42, K=50, c=1):
        super().__init__()
        self.num_samples = len(data_source)
        self.seed = seed
        self.K = K
        self.c = c
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _collatz_step(self, x, c):
        val = 3 * x + c
        v2 = (val & -val).bit_length() - 1
        return val >> v2

    def __iter__(self):
        offset = self.seed * 10000 + self.epoch * 7919
        values = np.empty(self.num_samples, dtype=np.float64)
        for i in range(self.num_samples):
            x = 2 * (i + offset) + 1
            for _ in range(self.K):
                x = self._collatz_step(x, self.c)
            values[i] = x
        indices = np.argsort(values).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class CollatzFix2Sampler(Sampler):
    def __init__(self, data_source, seed=42, K=50, noise_ratio=0.01):
        super().__init__()
        self.num_samples = len(data_source)
        self.seed = seed
        self.K = K
        self.noise_ratio = noise_ratio
        self.epoch = 0
        self.PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _collatz_step(self, x, c):
        val = 3 * x + c
        v2 = (val & -val).bit_length() - 1
        return val >> v2

    def __iter__(self):
        c = self.PRIMES[self.epoch % len(self.PRIMES)]
        offset = self.seed * 10000 + self.epoch * 7919
        values = np.empty(self.num_samples, dtype=np.float64)
        for i in range(self.num_samples):
            x = 2 * (i + offset) + 1
            for _ in range(self.K):
                x = self._collatz_step(x, c)
            values[i] = x

        vmin, vmax = values.min(), values.max()
        norm_values = (values - vmin) / (vmax - vmin + 1e-10)
        rng = np.random.RandomState(self.seed + self.epoch * 1337)
        noise = rng.uniform(-self.noise_ratio, self.noise_ratio, self.num_samples)
        indices = np.argsort(norm_values + noise).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class CollatzFix3Sampler(Sampler):
    def __init__(self, data_source, total_epochs=15, alpha_start=0.0, alpha_end=0.5, seed=42, K=50):
        super().__init__()
        self.num_samples = len(data_source)
        self.total_epochs = total_epochs
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.seed = seed
        self.K = K
        self.epoch = 0
        self.PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_alpha(self):
        t = self.epoch / max(1, self.total_epochs - 1)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * (1.0 - np.cos(np.pi * t)) / 2.0

    def _collatz_step(self, x, c):
        val = 3 * x + c
        v2 = (val & -val).bit_length() - 1
        return val >> v2

    def __iter__(self):
        alpha = self.get_alpha()
        c = self.PRIMES[self.epoch % len(self.PRIMES)]
        offset = self.seed * 10000 + self.epoch * 7919
        values = np.empty(self.num_samples, dtype=np.float64)
        for i in range(self.num_samples):
            x = 2 * (i + offset) + 1
            for _ in range(self.K):
                x = self._collatz_step(x, c)
            values[i] = x

        vmin, vmax = values.min(), values.max()
        norm_values = (values - vmin) / (vmax - vmin + 1e-10)
        if alpha > 0:
            rng = np.random.RandomState(self.seed + self.epoch * 9999)
            noise = rng.uniform(0.0, 1.0, self.num_samples)
            mixed = (1.0 - alpha) * norm_values + alpha * noise
            indices = np.argsort(mixed).tolist()
        else:
            indices = np.argsort(norm_values).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class CollatzSweepSampler(Sampler):
    """
    Sampler para el Experimento J (Randomness Sweep):
    Permite fijar alpha exacto en {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
    """
    def __init__(self, data_source, alpha_fixed=0.5, seed=42, K=50):
        super().__init__()
        self.num_samples = len(data_source)
        self.alpha_fixed = alpha_fixed
        self.seed = seed
        self.K = K
        self.epoch = 0
        self.PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _collatz_step(self, x, c):
        val = 3 * x + c
        v2 = (val & -val).bit_length() - 1
        return val >> v2

    def __iter__(self):
        alpha = self.alpha_fixed
        c = self.PRIMES[self.epoch % len(self.PRIMES)]
        offset = self.seed * 10000 + self.epoch * 7919
        values = np.empty(self.num_samples, dtype=np.float64)
        for i in range(self.num_samples):
            x = 2 * (i + offset) + 1
            for _ in range(self.K):
                x = self._collatz_step(x, c)
            values[i] = x

        vmin, vmax = values.min(), values.max()
        norm_values = (values - vmin) / (vmax - vmin + 1e-10)
        rng = np.random.RandomState(self.seed + self.epoch * 8888)
        noise = rng.uniform(0.0, 1.0, self.num_samples)
        mixed = (1.0 - alpha) * norm_values + alpha * noise
        indices = np.argsort(mixed).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class HaltonPermutationSampler(Sampler):
    def __init__(self, data_source, seed=42):
        super().__init__()
        self.num_samples = len(data_source)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        current_seed = self.seed + self.epoch
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=1, scramble=True, seed=current_seed)
            halton_points = sampler.random(n=self.num_samples).flatten()
            indices = np.argsort(halton_points).tolist()
        except ImportError:
            # fallback LCG distinto de Sobol
            a = 1103515245
            c = 12345
            m = 2**31
            val = current_seed * 7919 + 104729
            sequence = []
            for _ in range(self.num_samples):
                val = (a * val + c) % m
                sequence.append(val)
            indices = np.argsort(sequence).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


class SamplerFactory:
    @staticmethod
    def get_sampler(sampler_name: str, data_source, seed: int = 42, total_epochs: int = 15, alpha_fixed: float = 0.5):
        name = sampler_name.lower()
        if name in ['stochastic', 'random']:
            return StochasticSampler(data_source, seed=seed)
        elif name in ['sequential']:
            return SequentialBaseSampler(data_source)
        elif name in ['sobol', 'deterministic']:
            return SobolPermutationSampler(data_source, seed=seed)
        elif name in ['halton']:
            return HaltonPermutationSampler(data_source, seed=seed)
        elif name in ['collatz_v1', 'collatzv1']:
            return CollatzFix1Sampler(data_source, seed=seed)
        elif name in ['collatz_v2', 'collatzv2']:
            return CollatzFix2Sampler(data_source, seed=seed)
        elif name in ['collatz_v3', 'collatzv3', 'dest']:
            return CollatzFix3Sampler(data_source, total_epochs=total_epochs, seed=seed)
        elif name in ['collatz_sweep', 'sweep']:
            return CollatzSweepSampler(data_source, alpha_fixed=alpha_fixed, seed=seed)
        else:
            raise ValueError(f"Sampler desconocido: {sampler_name}")
