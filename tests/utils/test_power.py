"""Power-state provenance — a rate measured off mains is worth about half."""

import subprocess
from unittest.mock import MagicMock

import pytest

from sisr.utils import power
from sisr.utils.power import (
    BEST_PERFORMANCE_OVERLAY,
    PowerState,
    read_power_state,
    warn_unless_reference_power,
)

OTHER_OVERLAY = "961cc777-2547-4f9d-8174-7d86181b8a7a"  # "Better battery"


@pytest.fixture(autouse=True)
def _uncached():
    """`read_power_state` is process-cached; each test needs its own answer."""
    read_power_state.cache_clear()
    yield
    read_power_state.cache_clear()


def _stub_powershell(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(power.shutil, "which", lambda _: "/usr/bin/powershell.exe")
    monkeypatch.setattr(
        power.subprocess, "run", lambda *a, **k: MagicMock(stdout=stdout, returncode=0)
    )


def test_mains_under_best_performance_is_the_reference_condition(monkeypatch):
    _stub_powershell(monkeypatch, f"True\r\n{BEST_PERFORMANCE_OVERLAY}\r\n")
    state = read_power_state()
    assert state.on_mains is True
    assert state.is_reference
    assert state.note == "mains, overlay Best performance"


def test_battery_is_detected_and_named_in_the_note(monkeypatch):
    _stub_powershell(monkeypatch, f"False\r\n{BEST_PERFORMANCE_OVERLAY}\r\n")
    state = read_power_state()
    assert state.on_mains is False
    assert not state.is_reference
    assert "BATTERY" in state.note


def test_mains_under_a_different_overlay_is_not_the_reference(monkeypatch):
    """The overlay is half the condition: 172 vs 366 batches/min was the same host."""
    _stub_powershell(monkeypatch, f"True\r\n{OTHER_OVERLAY}\r\n")
    state = read_power_state()
    assert state.on_mains is True
    assert not state.is_reference
    assert OTHER_OVERLAY in state.note


def test_a_host_with_no_battery_counts_as_mains(monkeypatch):
    _stub_powershell(monkeypatch, f"nobattery\r\n{BEST_PERFORMANCE_OVERLAY}\r\n")
    assert read_power_state().is_reference


def test_no_powershell_degrades_to_unknown_not_an_error(monkeypatch):
    """Outside WSL/Windows there is no instrument. Unknown is a recorded state."""
    monkeypatch.setattr(power.shutil, "which", lambda _: None)
    state = read_power_state()
    assert state == PowerState(None, None, state.note)
    assert "no powershell.exe" in state.note


@pytest.mark.parametrize(
    "boom", [OSError("nope"), subprocess.TimeoutExpired(cmd="powershell.exe", timeout=30)]
)
def test_a_failing_probe_degrades_to_unknown(monkeypatch, boom):
    monkeypatch.setattr(power.shutil, "which", lambda _: "/usr/bin/powershell.exe")

    def _raise(*_a, **_k):
        raise boom

    monkeypatch.setattr(power.subprocess, "run", _raise)
    assert read_power_state().on_mains is None


def test_unreadable_output_degrades_to_unknown(monkeypatch):
    _stub_powershell(monkeypatch, "\r\n")
    assert read_power_state().on_mains is None


def test_warning_fires_on_battery_and_names_what_is_in_force():
    state = PowerState(False, BEST_PERFORMANCE_OVERLAY, "BATTERY, overlay Best performance")
    with pytest.warns(UserWarning, match="not the reference condition"):
        warn_unless_reference_power(state)


def test_warning_is_silent_on_the_reference_condition(recwarn):
    state = PowerState(True, BEST_PERFORMANCE_OVERLAY, "mains, overlay Best performance")
    warn_unless_reference_power(state)
    assert not [w for w in recwarn if "reference condition" in str(w.message)]


def test_warning_is_silent_when_the_state_is_unknown(recwarn):
    """No instrument is not evidence of a problem — it must not cry wolf."""
    warn_unless_reference_power(PowerState(None, None, "unknown: no powershell.exe on PATH"))
    assert not [w for w in recwarn if "reference condition" in str(w.message)]


def test_state_is_read_once_per_process(monkeypatch):
    """Two consumers, one answer — and no subprocess call in the checkpoint path."""
    calls = []
    monkeypatch.setattr(power.shutil, "which", lambda _: "/usr/bin/powershell.exe")
    monkeypatch.setattr(
        power.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(1) or MagicMock(stdout=f"True\r\n{BEST_PERFORMANCE_OVERLAY}\r\n")
        ),
    )
    read_power_state()
    read_power_state()
    assert len(calls) == 1


def test_as_dict_is_plain_scalars_for_weights_only_safety():
    state = PowerState(True, BEST_PERFORMANCE_OVERLAY, "mains, overlay Best performance")
    assert state.as_dict() == {
        "on_mains": True,
        "overlay": BEST_PERFORMANCE_OVERLAY,
        "note": "mains, overlay Best performance",
    }
    assert all(isinstance(v, bool | str | type(None)) for v in state.as_dict().values())
