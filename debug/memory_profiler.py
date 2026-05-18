"""
Training memory profiler — pinpoints where memory spikes and where it leaks.

Usage: Import and wrap a training step to log memory at each boundary.
"""
from __future__ import annotations

import gc
import os
import time
import json
from pathlib import Path

import torch
import psutil


def _read_rss_mb() -> float:
    """Read RSS (resident set size) of the current process in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _gpu_allocated_mb(device: torch.device | None = None) -> dict[str, float]:
    """Return allocated and reserved GPU memory per device in MB."""
    if not torch.cuda.is_available():
        return {}
    result = {}
    devices = [device] if device is not None else [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for dev in devices:
        idx = dev.index if dev.index is not None else 0
        result[f"gpu{idx}_alloc"] = torch.cuda.memory_allocated(idx) / (1024 * 1024)
        result[f"gpu{idx}_reserved"] = torch.cuda.memory_reserved(idx) / (1024 * 1024)
        result[f"gpu{idx}_peak"] = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
    return result


def snapshot(label: str, device: torch.device | None = None) -> dict:
    """Take a memory snapshot with a label."""
    snap = {
        "label": label,
        "timestamp": time.time(),
        "rss_mb": _read_rss_mb(),
        "pid": os.getpid(),
    }
    snap.update(_gpu_allocated_mb(device))
    return snap


class StepMemoryTracker:
    """Tracks memory across training step boundaries.

    Insert into a training loop:

        tracker = StepMemoryTracker(device, log_every=1)
        for step, batch in enumerate(train_loader):
            tracker.snap("before_move")
            batch = move_to_device(batch, device)
            tracker.snap("after_move")
            ...
            tracker.snap("after_backward")
            ...
            tracker.end_step(step)
    """

    def __init__(self, device: torch.device | None = None, log_every: int = 10,
                 output_path: str | None = None):
        self.device = device
        self.log_every = log_every
        self.output_path = output_path
        self.history: list[dict] = []
        self._step_snaps: list[dict] = []

    def snap(self, label: str) -> dict:
        s = snapshot(label, self.device)
        self._step_snaps.append(s)
        return s

    def end_step(self, step: int):
        """Record step-end summary. Computes deltas between key boundaries."""
        if self._step_snaps:
            step_summary = {
                "step": step,
                "num_snaps": len(self._step_snaps),
                "first": self._step_snaps[0],
                "last": self._step_snaps[-1],
            }
            # Compute RSS delta within this step
            if len(self._step_snaps) >= 2:
                rss_start = self._step_snaps[0].get("rss_mb", -1)
                rss_end = self._step_snaps[-1].get("rss_mb", -1)
                step_summary["rss_delta_mb"] = rss_end - rss_start if rss_start > 0 and rss_end > 0 else 0
                for gpu_key in [k for k in self._step_snaps[0] if "peak" in k]:
                    step_summary[f"{gpu_key}_delta"] = (
                        self._step_snaps[-1].get(gpu_key, 0) - self._step_snaps[0].get(gpu_key, 0)
                    )
            self.history.append(step_summary)

            if step % self.log_every == 0:
                self._print_step(step_summary)

            if self.output_path:
                self._flush()
        self._step_snaps = []

    def _print_step(self, summary: dict):
        rss = summary["last"].get("rss_mb", -1)
        rss_delta = summary.get("rss_delta_mb", 0)
        parts = [f"Step {summary['step']:>5d}"]
        parts.append(f"RSS={rss:.0f}MB" if rss > 0 else "RSS=N/A")
        if abs(rss_delta) > 0.5:
            parts.append(f"(Δ{rss_delta:+.0f}MB)")
        for k, v in summary["last"].items():
            if "peak" in k:
                parts.append(f"{k}={v:.0f}MB")
        print(f"[MEM] {' | '.join(parts)}", flush=True)

    def _flush(self):
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def summary(self):
        """Print summary statistics from all recorded steps."""
        if not self.history:
            print("[MEM] No data recorded.")
            return

        rss_values = [h["last"].get("rss_mb", -1) for h in self.history if h["last"].get("rss_mb", -1) > 0]
        rss_deltas = [h.get("rss_delta_mb", 0) for h in self.history]
        if rss_values:
            print(f"\n[MEM] --- Summary over {len(self.history)} steps ---")
            print(f"[MEM] RSS: min={min(rss_values):.0f}MB  max={max(rss_values):.0f}MB  "
                  f"first={rss_values[0]:.0f}MB  last={rss_values[-1]:.0f}MB  "
                  f"growth={rss_values[-1] - rss_values[0]:.0f}MB")
        if rss_deltas:
            growing = [d for d in rss_deltas if d > 0]
            shrinking = [d for d in rss_deltas if d < 0]
            print(f"[MEM] RSS per-step: mean_delta={sum(rss_deltas)/len(rss_deltas):.1f}MB  "
                  f"growing_steps={len(growing)}  shrinking_steps={len(shrinking)}")

        # Find top-5 largest RSS growth steps
        if len(self.history) >= 5:
            sorted_by_growth = sorted(self.history, key=lambda h: h.get("rss_delta_mb", 0), reverse=True)
            print(f"[MEM] Top-5 largest RSS growth steps:")
            for h in sorted_by_growth[:5]:
                print(f"[MEM]   Step {h['step']}: Δ{h.get('rss_delta_mb', 0):.0f}MB, RSS={h['last'].get('rss_mb', -1):.0f}MB")

        self.history = []


def profile_trainer_step(trainer, num_steps: int = 20):
    """Run a few training steps with memory profiling to identify bottlenecks.

    Call before the main training loop to get a baseline of where memory goes.
    """
    device = trainer.accelerator.device
    tracker = StepMemoryTracker(device=device, log_every=1,
                                output_path=f"{trainer.cfg.log.output_dir}/memory_profile.json")

    print(f"[MEM PROFILER] Running {num_steps} profiling steps...", flush=True)
    trainer.model.train()

    train_iter = iter(trainer.train_loader)
    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_loader)
            batch = next(train_iter)

        tracker.snap("0_before_move")
        batch = trainer.accelerator.device  # no-op, for labeling
        from utils.misc import move_to_device
        batch = move_to_device(batch, device)
        tracker.snap("1_after_move")

        with trainer.accelerator.accumulate(trainer.model):
            tracker.snap("2_before_forward")
            with trainer.accelerator.autocast():
                forward_output = trainer.forward_batch(batch, mode="train")
            tracker.snap("3_after_forward")

            batch_output = trainer.calculate_loss(forward_output, batch, mode="train")
            loss = batch_output.loss
            tracker.snap("4_after_loss")

            trainer.accelerator.backward(loss)
            tracker.snap("5_after_backward")

            if trainer.accelerator.sync_gradients:
                trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.cfg.train.clip_grad)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.lr_scheduler.step()
            tracker.snap("6_after_optimizer_step")

        del forward_output, batch_output, loss, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tracker.snap("7_after_cleanup")
        tracker.end_step(step)

    tracker.summary()
    if tracker.output_path:
        print(f"[MEM PROFILER] Detailed log saved to {tracker.output_path}", flush=True)


if __name__ == "__main__":
    # Standalone test: verify the tracker works
    tracker = StepMemoryTracker(log_every=1, output_path="/tmp/memory_test.json")
    for i in range(5):
        tracker.snap("test_start")
        x = torch.randn(1000, 1000)  # ~4MB
        tracker.snap("test_alloc")
        del x
        tracker.snap("test_free")
        tracker.end_step(i)
    tracker.summary()
    print("[MEM PROFILER] Standalone test passed.")
