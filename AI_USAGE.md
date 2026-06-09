# Generative AI Usage — CS231N Final Project

**Project**: Scaling Drifting VQ-VAE  
**Tool**: Claude Code CLI (model: claude-sonnet-4-6, Anthropic)

---

## Summary

Claude (via the Claude Code CLI) was used to write the large majority of implementation code in this repository. Research direction, experimental design, hypothesis formation, and interpretation of results were driven by the human authors. Every AI-generated artifact was reviewed, tested against real experimental data, and iterated on to meet the project's research goals.

This document is provided in compliance with the CS231N Generative AI policy.

---

## Division of labor

### Human-authored (research direction and intellectual contribution)

- **The core research idea**: Replacing the VQ-VAE commitment loss with a physics-inspired pairwise potential energy ("DriftingVQ"). The intuition that codebook collapse resembles a physics system lacking repulsion, and that an explicit U_nn (code-code repulsion) + U_pn (code-encoder attraction) potential could yield near-uniform codebook utilization — this was the human's conceptual contribution.

- **Experimental design**: Every phase's structure — what to compare, what to ablate, which hyperparameters to sweep, what constitutes a "decisive" result — was decided by the author after reading experimental output. The decision tree (e.g., "if no_pp beats EMA at 30k, declare drift dominant") was the author's.

- **Interpretation of results**: Reading the aggregate tables, identifying that U_pp (hidden-hidden repulsion) was hurting PSNR, discovering that STE outperforms rotation trick, identifying the high-hidden-norm mechanism driving uniform utilization — all human analysis.

- **Hypothesis iteration**: After each phase, the author decided what to run next and why. The progression from Phase 1 → 2 → 2b → hparam sweep → multi-dataset validation reflects the author's scientific judgment about what questions remained open.

- **Report and writeup**: All prose in the final report is the author's own writing. Claude was not used to draft report sections.

### AI-assisted (Claude Code CLI)

- **Experiment harness** (`experiments/`): The full training infrastructure — `train.py`, `modal_app.py`, `launch.py`, `data.py`, `models.py`, `metrics.py` — was largely written by Claude based on high-level specifications from the author ("I want a config-driven training loop that logs to both disk and wandb, runs on Modal L40S GPUs, and saves per-step metrics in machine-readable JSON/CSV").

- **Quantizer implementation** (`experiments/quantizers.py`): The factory and `DriftEMAVQ` hybrid class were written by Claude once the author specified the interface and logic.

- **Analysis scripts** (`experiments/scripts/`): `aggregate.py`, `status.py`, `diagnose.py`, `inspect_runs.py`, `recover_wandb.py`, `kill_run.py` — all written by Claude to the author's specifications, with iteration based on what outputs were actually useful.

- **Phase configs** (`experiments/configs/`): Each phase config was co-written: the author specified which methods, codebook sizes, seeds, and datasets to run; Claude translated this into the `make_runs()` dataclass list.

- **Core DriftingVQ module** (`vector_quantize_pytorch/drifting_vq.py`): The physics-energy formulation was the author's (based on the Ziming Liu reference implementation in `ziming_drift_vqvae.py`), with Claude implementing the PyTorch module structure, the `U_nn`/`U_pp`/`U_pn` energy terms, and the ablation flags.

- **Tests** (`tests/test_drifting_vq.py`): Written by Claude.

- **Boilerplate scripts** (`experiments/scripts/fetch_recon_figures.py`, `make_paper_grid.py`, `plot_convergence.py`, `aggregate_downstream.py`): Fully written by Claude.

---

## What was not AI-assisted

- `ziming_drift_vqvae.py` — reference implementation from Ziming Liu et al., included verbatim for comparison
- `vector_quantize_pytorch/vector_quantize_pytorch.py` and all other files in `vector_quantize_pytorch/` except `drifting_vq.py` — these are from the upstream `lucidrains/vector-quantize-pytorch` open-source library, cited in the report
- `latex/` — the CVPR paper template files
- All milestone PDFs

---

## Documentation artifacts

The following files in the repo serve as evidence of the iterative AI-assisted workflow:

- `CLAUDE.md` — the instruction context file given to Claude at the start of each session, showing what the AI was told about the project structure
- `experiments/CONTEXT.md` — a running log of every experimental phase, results, interpretation, and next steps; updated by the author after each run; shows the human-driven research loop
- `experiments/README.md` — documents the experiment harness architecture

These files together demonstrate a workflow where Claude handled implementation boilerplate and the author drove all research decisions.

---

## Citation

> Claude Code CLI (claude-sonnet-4-6). Anthropic, 2026. Used for implementation code generation under human direction. https://claude.ai/code
