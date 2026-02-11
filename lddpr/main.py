from __future__ import annotations

import json
import random
import time
from pathlib import Path

import fire
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset, get_worker_info

LOG_DIR = Path("/tmp/lddpr_logs")

MAIN_RNG_KEYS = ["torch_rand", "np_rand", "py_rand"]
DL_RNG_KEYS = ["dl_torch_rand", "dl_np_rand", "dl_py_rand"]


class RNGProbeDataset(Dataset):
    """Dataset that records RNG values sampled inside __getitem__ (runs in worker process)."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        return (
            torch.tensor(idx),
            torch.tensor(worker_id),
            torch.tensor(torch.rand(1).item()),  # dl_torch_rand
            torch.tensor(np.random.random()),  # dl_np_rand
            torch.tensor(random.random()),  # dl_py_rand
        )


class RNGProbeModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.loss_fn = nn.MSELoss()
        self.epoch_records: list[dict] = []
        self._current_epoch_steps: list[dict] = []

    def training_step(self, batch, batch_idx):
        indices, worker_ids, dl_torch_rand, dl_np_rand, dl_py_rand = batch
        x = indices.float().unsqueeze(-1)
        y = self.model(x)
        loss = self.loss_fn(y, x)

        record = {
            "batch_idx": batch_idx,
            "sample_indices": indices.tolist(),
            "worker_ids": worker_ids.tolist(),
            # RNG values from dataloader workers
            "dl_torch_rand": dl_torch_rand.tolist(),
            "dl_np_rand": dl_np_rand.tolist(),
            "dl_py_rand": dl_py_rand.tolist(),
            # RNG values from main process (training_step)
            "torch_rand": torch.rand(1).item(),
            "np_rand": np.random.random(),
            "py_rand": random.random(),
        }
        if torch.cuda.is_available():
            record["cuda_rand"] = torch.randn(1, device=self.device).item()

        self._current_epoch_steps.append(record)
        return loss

    def on_train_epoch_end(self):
        self.epoch_records.append(
            {
                "epoch": self.current_epoch,
                "rank": self.global_rank,
                "steps": list(self._current_epoch_steps),
            }
        )
        self._current_epoch_steps = []

    def on_train_end(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"rank{self.global_rank}.json"
        with open(log_path, "w") as f:
            json.dump(self.epoch_records, f, indent=2)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _load_logs(num_ranks: int, max_epochs: int) -> dict[int, list[dict]]:
    """Poll for and load JSON log files written by each rank."""
    logs: dict[int, list[dict]] = {}
    for rank in range(num_ranks):
        log_path = LOG_DIR / f"rank{rank}.json"
        for _ in range(30):
            if log_path.exists():
                try:
                    with open(log_path) as f:
                        data = json.load(f)
                    if len(data) >= max_epochs:
                        logs[rank] = data
                        break
                except (json.JSONDecodeError, ValueError):
                    pass
            time.sleep(0.5)
        else:
            print(f"WARNING: Could not read complete log for rank {rank}")
            if log_path.exists():
                with open(log_path) as f:
                    logs[rank] = json.load(f)
    return logs


def _cross_rank_match(values: dict[int, object]) -> str:
    """Return 'SAME', 'DIFF', or 'N/A' depending on whether all rank values agree."""
    if len(values) < 2:
        return "N/A"
    vals = list(values.values())
    return "SAME" if all(v == vals[0] for v in vals[1:]) else "DIFF"


def _get_epoch_data(
    logs: dict[int, list[dict]], epoch: int
) -> tuple[dict[int, list[int]], dict[int, list[dict]]]:
    """Extract per-rank sample indices and step records for a given epoch."""
    indices_per_rank: dict[int, list[int]] = {}
    steps_per_rank: dict[int, list[dict]] = {}
    for rank in sorted(logs.keys()):
        epoch_data = next((e for e in logs[rank] if e["epoch"] == epoch), None)
        if epoch_data is None:
            print(f"  Rank {rank}: NO DATA")
            continue
        steps = epoch_data["steps"]
        indices = [idx for step in steps for idx in step["sample_indices"]]
        indices_per_rank[rank] = indices
        steps_per_rank[rank] = steps
    return indices_per_rank, steps_per_rank


def _main_rng_keys(steps_per_rank: dict[int, list[dict]]) -> list[str]:
    """Return the main-process RNG keys, including cuda_rand when present."""
    keys = list(MAIN_RNG_KEYS)
    if any("cuda_rand" in s for steps in steps_per_rank.values() for s in steps):
        keys.append("cuda_rand")
    return keys


def _print_sample_distribution(
    indices_per_rank: dict[int, list[int]],
    dataset_size: int,
):
    all_indices = set(range(dataset_size))
    print("\n  Sample Distribution:")
    for rank in sorted(indices_per_rank.keys()):
        print(f"    Rank {rank}: {indices_per_rank[rank]}")

    combined = [idx for indices in indices_per_rank.values() for idx in indices]
    combined_set = set(combined)
    missing = all_indices - combined_set
    duplicates = len(combined) - len(combined_set)

    print(f"\n  Coverage: {len(combined_set)}/{dataset_size} indices covered")
    if missing:
        print(f"  Missing indices: {sorted(missing)}")
    if duplicates:
        print(f"  Duplicate samples across ranks: {duplicates}")
    else:
        print("  No duplicates across ranks")

    # Overlap between rank pairs
    ranks = sorted(indices_per_rank.keys())
    for i, r_i in enumerate(ranks):
        for r_j in ranks[i + 1 :]:
            overlap = set(indices_per_rank[r_i]) & set(indices_per_rank[r_j])
            if overlap:
                print(f"  Overlap rank {r_i} & {r_j}: {sorted(overlap)}")


def _print_shuffle_change(
    indices_per_rank: dict[int, list[int]],
    prev_indices: dict[int, list[int]],
):
    if not prev_indices:
        return
    print("\n  Shuffle Change vs Previous Epoch:")
    for rank in sorted(indices_per_rank.keys()):
        if rank in prev_indices:
            same = indices_per_rank[rank] == prev_indices[rank]
            print(f"    Rank {rank}: {'SAME order' if same else 'DIFFERENT order'}")


def _print_step_rng(steps_per_rank: dict[int, list[dict]]):
    main_keys = _main_rng_keys(steps_per_rank)
    max_steps = max((len(s) for s in steps_per_rank.values()), default=0)

    for step_idx in range(max_steps):
        print(f"\n    Step {step_idx}:")

        # Worker IDs
        for rank in sorted(steps_per_rank.keys()):
            steps = steps_per_rank[rank]
            if step_idx < len(steps) and "worker_ids" in steps[step_idx]:
                print(f"      Rank {rank} worker_ids: {steps[step_idx]['worker_ids']}")

        # Dataloader RNG (per-sample values from workers)
        print("      --- dataloader (worker process) ---")
        for key in DL_RNG_KEYS:
            values = {
                rank: steps[step_idx][key]
                for rank, steps in sorted(steps_per_rank.items())
                if step_idx < len(steps) and key in steps[step_idx]
            }
            match_str = _cross_rank_match(values)
            for rank, v in sorted(values.items()):
                fmt = [f"{x:.6f}" for x in v] if isinstance(v, list) else [f"{v:.6f}"]
                print(f"      {key:16s} R{rank}: [{', '.join(fmt)}]  [{match_str}]")

        # Main process RNG (single value per step)
        print("      --- training_step (main process) ---")
        for key in main_keys:
            values = {
                rank: steps[step_idx][key]
                for rank, steps in sorted(steps_per_rank.items())
                if step_idx < len(steps) and key in steps[step_idx]
            }
            match_str = _cross_rank_match(values)
            parts = [f"R{rank}={v:.6f}" for rank, v in sorted(values.items())]
            print(f"      {key:16s}: {' | '.join(parts)}  [{match_str}]")


def _print_final_summary(logs: dict[int, list[dict]], max_epochs: int):
    print(f"\n{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")

    # Shuffle consistency across epochs
    print("\n  Shuffle across epochs:")
    for rank in sorted(logs.keys()):
        orders = []
        for epoch in range(max_epochs):
            epoch_data = next((e for e in logs[rank] if e["epoch"] == epoch), None)
            if epoch_data:
                orders.append(
                    [idx for step in epoch_data["steps"] for idx in step["sample_indices"]]
                )
        if len(orders) >= 2:
            all_same = all(o == orders[0] for o in orders[1:])
            print(
                f"    Rank {rank}: {'SAME every epoch' if all_same else 'CHANGES between epochs'}"
            )

    # Cross-rank RNG match (epoch 0, step 0)
    print("\n  RNG cross-rank match (epoch 0, step 0):")
    first_steps: dict[int, dict] = {}
    for rank in sorted(logs.keys()):
        epoch_data = next((e for e in logs[rank] if e["epoch"] == 0), None)
        if epoch_data and epoch_data["steps"]:
            first_steps[rank] = epoch_data["steps"][0]

    if len(first_steps) >= 2:
        main_keys = list(MAIN_RNG_KEYS)
        if any("cuda_rand" in s for s in first_steps.values()):
            main_keys.append("cuda_rand")

        print("    --- training_step (main process) ---")
        for key in main_keys:
            values = {r: s[key] for r, s in first_steps.items() if key in s}
            match = _cross_rank_match(values)
            print(
                f"    {key:16s}: {'SAME across ranks' if match == 'SAME' else 'DIFFERENT across ranks'}"
            )

        print("    --- dataloader (worker process) ---")
        for key in DL_RNG_KEYS:
            values = {r: s[key] for r, s in first_steps.items() if key in s}
            match = _cross_rank_match(values)
            print(
                f"    {key:16s}: {'SAME across ranks' if match == 'SAME' else 'DIFFERENT across ranks'}"
            )


def print_summary(num_ranks: int, max_epochs: int, dataset_size: int):
    logs = _load_logs(num_ranks, max_epochs)
    if not logs:
        print("ERROR: No logs found!")
        return

    prev_epoch_indices: dict[int, list[int]] = {}

    for epoch in range(max_epochs):
        print(f"\n{'=' * 70}")
        print(f" EPOCH {epoch}")
        print(f"{'=' * 70}")

        indices_per_rank, steps_per_rank = _get_epoch_data(logs, epoch)
        _print_sample_distribution(indices_per_rank, dataset_size)
        _print_shuffle_change(indices_per_rank, prev_epoch_indices)
        _print_step_rng(steps_per_rank)
        prev_epoch_indices = dict(indices_per_rank)

    _print_final_summary(logs, max_epochs)


def main(
    seed_everything: bool = True,
    seed: int = 42,
    persistent_workers: bool = False,
    num_workers: int = 2,
    seed_workers: bool = True,
    num_devices: int = 2,
    batch_size: int = 4,
    max_epochs: int = 3,
    dataset_size: int = 16,
):
    # Clean old logs
    if LOG_DIR.exists():
        for f in LOG_DIR.glob("*.json"):
            f.unlink()

    if seed_everything:
        L.seed_everything(seed, workers=seed_workers)

    dataset = RNGProbeDataset(dataset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    model = RNGProbeModule()
    trainer = Trainer(
        strategy="ddp" if num_devices > 1 else "auto",
        devices=num_devices,
        accelerator="auto",
        max_epochs=max_epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, dataloader)

    if trainer.global_rank == 0:
        print_summary(num_devices, max_epochs, dataset_size)
        print("\n" + "=" * 70)
        print("Configuration:")
        print(f"  seed={seed}, seed_workers={seed_workers}")
        print(f"  num_workers={num_workers}, persistent_workers={persistent_workers}")
        print(f"  num_devices={num_devices}, batch_size={batch_size}")
        print(f"  max_epochs={max_epochs}, dataset_size={dataset_size}")


if __name__ == "__main__":
    fire.Fire(main)
