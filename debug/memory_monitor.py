#!/usr/bin/env python3
"""
External memory monitor — runs alongside training to survive OOM kills.

Usage:
    # In terminal 2, before starting training:
    python debug/memory_monitor.py --pid $(pgrep -f train_pi3x | head -1) --interval 5 --output outputs/memory_log.csv

    # Or monitor a process by name pattern:
    python debug/memory_monitor.py --name "train_pi3x" --interval 5
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


def find_process(name_pattern: str) -> psutil.Process | None:
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline", []) or [])
            if name_pattern in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def memory_snapshot(proc: psutil.Process) -> dict:
    """Collect memory info for a process and all its children."""
    try:
        main_rss = proc.memory_info().rss / (1024 * 1024)
        main_vms = proc.memory_info().vms / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"alive": False}

    children_rss = 0.0
    children_count = 0
    try:
        for child in proc.children(recursive=True):
            try:
                children_rss += child.memory_info().rss / (1024 * 1024)
                children_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    gpu_info = {}
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpu_info[f"gpu{parts[0]}_used_mb"] = float(parts[1])
    except Exception:
        pass

    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "rss_mb": round(main_rss, 1),
        "vms_mb": round(main_vms, 1),
        "total_rss_mb": round(main_rss + children_rss, 1),
        "children_count": children_count,
        "children_rss_mb": round(children_rss, 1),
        **gpu_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor training process memory")
    parser.add_argument("--pid", type=int, help="PID of the training process")
    parser.add_argument("--name", type=str, default="train_pi3x", help="Name pattern to find process")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    args = parser.parse_args()

    if args.pid:
        try:
            proc = psutil.Process(args.pid)
        except psutil.NoSuchProcess:
            print(f"ERROR: No process with PID {args.pid}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Looking for process matching '{args.name}'...")
        proc = find_process(args.name)
        if proc is None:
            print(f"ERROR: No process found matching '{args.name}'", file=sys.stderr)
            sys.exit(1)

    print(f"Monitoring PID {proc.pid} (cmd: {' '.join(proc.cmdline()[:3])}...)")
    print(f"Interval: {args.interval}s | Output: {args.output or 'stdout'}")
    print(f"{'timestamp':<20s} {'rss_mb':>8s} {'total_mb':>8s} {'children':>8s} {'gpu_info'}")
    print("-" * 80)

    output_file = None
    writer = None
    if args.output:
        output_file = open(args.output, "w", newline="")
        writer = csv.DictWriter(output_file, fieldnames=[
            "timestamp", "rss_mb", "vms_mb", "total_rss_mb", "children_count", "children_rss_mb",
            "gpu0_used_mb", "gpu1_used_mb"
        ])
        writer.writeheader()

    start_time = time.time()
    last_rss = None

    try:
        while True:
            if args.duration and (time.time() - start_time) > args.duration:
                print("\nDuration reached. Stopping.")
                break

            snap = memory_snapshot(proc)
            if not snap.get("alive", False):
                print(f"\n[{datetime.now().isoformat()}] PROCESS EXITED (likely OOM killed)")
                print(f"Last RSS snapshot: {last_rss} MB")
                break

            gpu_str = " ".join(f"{k}={v:.0f}" for k, v in snap.items() if "gpu" in k)
            delta_str = ""
            if last_rss is not None:
                delta = snap["rss_mb"] - last_rss
                if abs(delta) > 0.1:
                    delta_str = f" Δ{delta:+.1f}"
            print(f"{snap['timestamp']:<20s} {snap['rss_mb']:>7.1f}MB{delta_str:<10s} "
                  f"{snap['total_rss_mb']:>7.1f}MB {snap['children_count']:>5d}    {gpu_str}",
                  flush=True)

            if writer:
                writer.writerow(snap)
                output_file.flush()

            last_rss = snap["rss_mb"]
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if output_file:
            output_file.close()
            print(f"\nLog saved to {args.output}")


if __name__ == "__main__":
    main()
