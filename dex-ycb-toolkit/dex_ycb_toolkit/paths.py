"""Shared path helpers for DexYCB toolkit integration."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def pi3_data_root() -> Path:
  return Path(os.environ.get("PI3_DATA_ROOT", repo_root() / "data")).expanduser()


def dexycb_root() -> str:
  root = os.environ.get(
      "DEXYCB_ROOT",
      os.environ.get("DEX_YCB_DIR", pi3_data_root() / "dataset" / "dexycb"),
  )
  return str(Path(root).expanduser())


def mano_root() -> str:
  root = os.environ.get(
      "MANO_ROOT",
      pi3_data_root() / "model" / "hamer" / "_DATA" / "data" / "mano",
  )
  return str(Path(root).expanduser())
