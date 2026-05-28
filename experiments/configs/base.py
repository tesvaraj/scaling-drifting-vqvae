"""Shared helpers for building phase configs.

A phase config is a Python module exposing:
    PHASE: str
    make_runs() -> list[TrainConfig]
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from experiments.train import TrainConfig


def run_id(method: str, K: int, seed: int, *, dataset: str = 'cifar10',
           suffix: str = '') -> str:
    s = f'{dataset}_{method}_K{K}_seed{seed}'
    if suffix:
        s += f'_{suffix}'
    return s


def variants(base: TrainConfig, overrides: Iterable[dict]) -> list[TrainConfig]:
    """Build a list of TrainConfigs by applying each overrides dict to a copy of base."""
    out = []
    for ov in overrides:
        cfg = deepcopy(base)
        for k, v in ov.items():
            if not hasattr(cfg, k):
                raise AttributeError(f'TrainConfig has no field {k!r}')
            setattr(cfg, k, v)
        out.append(cfg)
    return out
