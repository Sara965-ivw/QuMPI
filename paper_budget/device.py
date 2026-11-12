"""Choose the device whose calibration the frozen hardware features are read from.

quest_phase2.hardware._device() is frozen thesis code: it builds the calibration tables for the
layout features and the layout-ESP from make_device("fake_brisbane") and caches them (lru_cache,
one entry). For the device-transfer test (notes/PREREG_transfer_sherbrooke.md) the same frozen
code must read another device's calibration. use_device(name) does this WITHOUT editing any file
in stage2_predictor/: it clears that cache and refills it while make_device is pointed at `name`,
then restores make_device. Every later call of _device() in this process returns the tables of
`name`, until use_device is called again.

  from paper_budget.device import use_device, current_device
  use_device("fake_sherbrooke")      # before any circuit is loaded
  assert current_device() == "fake_sherbrooke"

The default device of the thesis is unchanged: use_device("fake_brisbane") gives exactly the
tables that _device() builds by itself (tests/test_transfer.py checks this).
"""
from __future__ import annotations

import paper_budget.att_data  # noqa: F401  (puts stage2_predictor on sys.path)
import quest_phase1.labeling_v2 as _lv2  # noqa: E402  (frozen)
from quest_phase2 import hardware as _hw  # noqa: E402  (frozen)

DEVICES = ("fake_brisbane", "fake_sherbrooke", "fake_torino")   # those make_device knows
_NAMES = {"fake_brisbane": "ibm_brisbane", "fake_sherbrooke": "ibm_sherbrooke",
          "fake_torino": "ibm_torino"}                           # backend.name of each


def use_device(name: str) -> None:
    """Make quest_phase2.hardware read the calibration of `name` from now on (this process)."""
    if name not in DEVICES:
        raise ValueError(f"unknown device {name!r}; known: {DEVICES}")
    original = _lv2.make_device

    def pinned(_name: str = "fake_brisbane", seed: int = 1234, opt_level: int = 2):
        return original(name, seed=seed, opt_level=opt_level)

    _hw._device.cache_clear()
    _lv2.make_device = pinned
    try:
        _hw._device()                       # fills the one-entry cache with `name`
    finally:
        _lv2.make_device = original
    got = current_device()
    if got != name:
        raise RuntimeError(f"hardware tables are for {got}, expected {name}")


def current_device() -> str:
    """Name (as given to make_device) of the device the hardware tables were built from."""
    backend_name = _hw._device()[0].backend.name
    for k, v in _NAMES.items():
        if backend_name in (k, v):
            return k
    raise RuntimeError(f"unrecognised backend {backend_name!r}")
