"""Smoke tests verifying basic project structure and metadata.

These are intentionally lightweight — they don't import the heavy ML
dependencies (torch, vectorbt, stable-baselines3) so CI stays fast.
Add module-level tests under tests/ as the project grows.
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MODULES = ("agents", "backtest", "data", "forecasting", "rl")


def test_python_version_supported() -> None:
    assert sys.version_info >= (3, 12), "Project requires Python >= 3.12"


def test_pyproject_parses() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    assert cfg["project"]["name"] == "quantagent-rl"
    assert cfg["project"]["requires-python"].startswith(">=3.12")


def test_expected_modules_present() -> None:
    for mod in EXPECTED_MODULES:
        init = ROOT / mod / "__init__.py"
        assert init.exists(), f"Missing {mod}/__init__.py"
