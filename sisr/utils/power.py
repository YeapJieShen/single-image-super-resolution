"""Host power state, recorded so a throughput number carries its own conditions.

A rate measured off mains is worth about half: the same adversarial run on the
same config managed 172 batches/min on a draining battery against 366 on mains
under Windows' Best-performance overlay. The crash when the battery ran flat was
loud and cost 87 minutes; the contaminated rate was silent and went into the
notes as sound.

**Advisory, never fatal.** A correctness-only run on a laptop is legitimate; not
knowing afterwards which conditions produced a number is not. So this warns and
records, and refuses nothing.

WSL/Windows only, because that is where the instrument is: ``nvidia-smi`` cannot
read the enforced power limit under WSL, and ``powercfg /getactivescheme``
reports the underlying scheme rather than the overlay sitting on top of it, so
the registry value is what has to be read. Anywhere else this degrades to
"unknown", which is a recorded state and not an error.

Torch-free, like everything else in this package -- see the package docstring.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from typing import Any

#: Windows' "Best performance" power overlay -- the reference condition every
#: throughput figure in this project's notes was measured under.
BEST_PERFORMANCE_OVERLAY = "ded574b5-45a0-4f42-8737-46345c09c238"

_SCHEME_KEY = r"HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"

_PROBE = (
    "$b=Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus "
    "-ErrorAction SilentlyContinue;"
    "if ($null -eq $b) { Write-Output 'nobattery' } else { Write-Output $b.PowerOnline };"
    f"$p=Get-ItemProperty '{_SCHEME_KEY}';"
    "Write-Output $p.ActiveOverlayAcPowerScheme"
)


@dataclass(frozen=True)
class PowerState:
    """What the host's power looked like, or why that could not be established.

    Args:
        on_mains: ``True`` on mains, ``False`` on battery, ``None`` when
            unreadable — a machine reporting no battery counts as ``True``.
        overlay: Active AC power overlay GUID, or ``None`` if unreadable.
        note: Always set, and always readable by a human. This is the field
            worth putting in front of someone; the other two are for code.
    """

    on_mains: bool | None
    overlay: str | None
    note: str

    @property
    def is_reference(self) -> bool:
        """Whether this is the condition the project's throughput figures assume."""
        return self.on_mains is True and self.overlay == BEST_PERFORMANCE_OVERLAY

    def as_dict(self) -> dict[str, Any]:
        """Plain-scalar form for artifact metadata (``weights_only``-safe)."""
        return {"on_mains": self.on_mains, "overlay": self.overlay, "note": self.note}


@functools.cache
def read_power_state(timeout: float = 30.0) -> PowerState:
    """Read the host power state once per process.

    Cached because the two consumers want the *same* answer: the warning at
    startup and the provenance recorded in every artifact the run writes. It
    therefore describes the state at first read, near the start of the run --
    not at each checkpoint save, which would put a subprocess call, and a
    chance to hang, in the save path.

    Args:
        timeout: Seconds to wait on PowerShell before giving up.

    Returns:
        A :class:`PowerState`; never raises, since no reading is a recorded
        state rather than a failure.
    """
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return PowerState(None, None, "unknown: no powershell.exe on PATH (not WSL/Windows)")
    try:
        probe = subprocess.run(
            [powershell, "-NoProfile", "-Command", _PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PowerState(None, None, f"unknown: power probe failed ({type(exc).__name__})")

    lines = [ln.strip() for ln in probe.stdout.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 2:
        return PowerState(None, None, f"unknown: power probe returned {probe.stdout!r}")

    battery, overlay = lines[0], lines[1]
    if battery == "nobattery":
        on_mains: bool | None = True
    elif battery.lower() in ("true", "false"):
        on_mains = battery.lower() == "true"
    else:
        return PowerState(None, overlay, f"unknown: unreadable battery state {battery!r}")

    supply = "mains" if on_mains else "BATTERY"
    best = "Best performance" if overlay == BEST_PERFORMANCE_OVERLAY else overlay
    return PowerState(on_mains, overlay, f"{supply}, overlay {best}")


def warn_unless_reference_power(state: PowerState | None = None) -> PowerState:
    """Warn when the host is readably *not* in the condition throughput figures assume.

    Silent when the state is unknown: there is no instrument outside
    WSL/Windows, so warning there would fire on every run everywhere and say
    nothing. Unknown is a recorded state, not a suspicion.

    Args:
        state: State to judge; read via :func:`read_power_state` when omitted.

    Returns:
        The state judged, so a caller can record what it warned about.
    """
    state = read_power_state() if state is None else state
    if state.on_mains is not None and not state.is_reference:
        warnings.warn(
            f"Power state is not the reference condition for throughput: {state.note}. "
            f"Timings from this run are not comparable to figures measured on mains "
            f"under the Best-performance overlay -- off mains costs roughly half. "
            f"Training is unaffected; this is recorded in the run's artifact metadata.",
            UserWarning,
            stacklevel=2,
        )
    return state
