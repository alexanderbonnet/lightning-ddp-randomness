# lightning-ddp-randomness

> **TL;DR:** set `seed_everything` with `workers=True` and `num_workers>=1` in your dataloader. THis should work for most applications.

A diagnostic tool for understanding how RNG states behave across processes when using [PyTorch Lightning](https://lightning.ai/) with DDP.

`seed_everything` is supposed to make training deterministic, but the interactions between Python, NumPy, PyTorch, and CUDA RNGs across multiple processes and dataloader workers can be surprising. This tool makes the behavior visible by sampling from each RNG in both worker and main processes, then comparing values across ranks.

## How it works

1. **`RNGProbeDataset`** samples from `torch`, `numpy`, and `random` RNGs inside dataloader workers.
2. **`RNGProbeModule`** records those values plus additional main-process RNG samples during `training_step`. Each rank writes its records to a JSON log.
3. After training, rank 0 prints a report covering sample distribution, shuffle order, per-step RNG values, cross-rank agreement, and weight initialization consistency.

## Installation

```bash
git clone https://github.com/alexander-bonnet/lightning-ddp-randomness.git
cd lightning-ddp-randomness
uv sync
```

## Usage

```bash
# Run with defaults (seed_everything=True, 2 devices, 3 epochs)
uv run python -m lddpr.main

# Disable seeding to see divergent RNG behavior
uv run python -m lddpr.main --seed_everything=False

# Customize the run
uv run python -m lddpr.main \
  --seed=123 \
  --num_devices=4 \
  --num_workers=4 \
  --persistent_workers=True \
  --batch_size=8 \
  --max_epochs=5 \
  --dataset_size=32
```

### Parameters

| Parameter            | Default | Description                                          |
|----------------------|---------|------------------------------------------------------|
| `seed_everything`    | `True`  | Call `L.seed_everything()` before training           |
| `seed`               | `42`    | Seed value passed to `seed_everything`               |
| `seed_workers`       | `True`  | Also seed dataloader worker processes                |
| `num_workers`        | `2`     | Number of dataloader workers per rank                |
| `persistent_workers` | `False` | Keep worker processes alive between epochs           |
| `num_devices`        | `2`     | Number of DDP devices (processes)                    |
| `batch_size`         | `4`     | Batch size per rank                                  |
| `max_epochs`         | `3`     | Number of training epochs                            |
| `dataset_size`       | `16`    | Total number of samples in the dataset               |

## License

MIT
