"""Resource ceilings for local child processes — the missing half of the local backend's posture.

The local backend is the *trusted operator shell* (SECURITY.md §3.2): it runs commands as the user,
with the user's files. That stays true. What it has never had is any ceiling on what one command can
*consume* — an OOM, a fork bomb, or a 40 GB file takes down the gateway and the machine with it, and
the operator's recovery window is the same box that is now unresponsive.

This module supplies a POSIX-only, fork-side ceiling (``preexec_fn``) applied to the spawned shell and
inherited by everything it forks. It is deliberately *not* a security boundary — it confines resource
consumption, not reach. Two design constraints come from the container reality:

* ``RLIMIT_NPROC`` is per-UID and therefore only safe when this process owns its UID (a container with
  its own user). On a shared host it fork-kills unrelated processes belonging to the same uid, so it is
  gated on an explicit container signal and is off by default.
* ``RLIMIT_AS`` must not be applied to interpreters that reserve large *virtual* ranges (V8/Node
  reserves multiple GB of address space it never commits). A hard address-space cap turns those into an
  instant SIGTRAP with no output, so the ceiling is set above the largest configured need rather than
  tuned tight, and the caller can raise it per call.

Fail-open by design: if ``resource`` is unavailable, or a limit cannot be set, the child runs exactly as
it did before. A ceiling that cannot be applied must never break the operator's shell.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

POSIX = sys.platform != "win32" and os.name == "posix"

# Above the largest virtual reservation a common runtime makes (Node/V8 reserves ~2-4 GB of address
# space it never commits). Under this, `node` dies with SIGTRAP and no stdout — measured, not guessed.
DEFAULT_MEM_MB = 8192
DEFAULT_CPU_S = 900
DEFAULT_FSIZE_MB = 4096
DEFAULT_NOFILE = 4096
DEFAULT_PROCS = 2048

# Naming: an operator sets this to 1 inside a container whose UID this process owns. It is NOT a
# security knob — it is permission to use a per-UID limit that is unsafe on a shared host.
_NPROC_ENV = "HERMES_LOCAL_LIMITS_NPROC"


def nproc_safe() -> bool:
    """True only when this process owns its UID (container-like), so RLIMIT_NPROC cannot misfire."""
    return POSIX and os.environ.get(_NPROC_ENV, "0").strip() in {"1", "true", "yes"}


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.debug("Ignoring non-integer %s=%r; using %d", name, raw, default)
        return default
    return value if value >= minimum else default


def limits_for_child() -> Optional[dict]:
    """The limits to apply, or None when they are disabled/unavailable. Read at spawn time so a config
    or env change lands without a restart (matching the local backend's other per-call reads)."""
    if not POSIX or os.environ.get("HERMES_LOCAL_LIMITS", "1").strip() in {"0", "false", "no"}:
        return None
    try:
        import resource  # noqa: F401 — presence check only
    except Exception:
        return None
    return {
        "mem_mb": _int_env("HERMES_LOCAL_LIMIT_MEM_MB", DEFAULT_MEM_MB, minimum=64),
        "cpu_s": _int_env("HERMES_LOCAL_LIMIT_CPU_S", DEFAULT_CPU_S, minimum=1),
        "fsize_mb": _int_env("HERMES_LOCAL_LIMIT_FSIZE_MB", DEFAULT_FSIZE_MB, minimum=1),
        "nofile": _int_env("HERMES_LOCAL_LIMIT_NOFILE", DEFAULT_NOFILE, minimum=16),
        "procs": _int_env("HERMES_LOCAL_LIMIT_PROCS", DEFAULT_PROCS, minimum=1),
    }


def make_preexec(limits: Optional[dict] = None):
    """A ``preexec_fn`` applying *limits*, or None when there is nothing to apply.

    Runs between fork and exec: no logging, no allocation, no Python that can raise uncaught. Each
    limit is applied independently so one failure cannot take the others down with it, and
    ``PR_SET_NO_NEW_PRIVS`` is best-effort (a setuid child would otherwise escape the ceilings by
    dropping them).
    """
    limits = limits if limits is not None else limits_for_child()
    if not limits:
        return None

    def _pre() -> None:  # pragma: no cover — child side of fork
        try:
            import resource

            try:
                import ctypes

                ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
            except Exception:
                pass

            def _set(kind, value) -> None:
                try:
                    resource.setrlimit(kind, (value, value))
                except (ValueError, OSError, AttributeError):
                    pass

            _set(resource.RLIMIT_CPU, limits["cpu_s"])
            _set(resource.RLIMIT_AS, limits["mem_mb"] * 1024 * 1024)
            _set(resource.RLIMIT_FSIZE, limits["fsize_mb"] * 1024 * 1024)
            _set(resource.RLIMIT_NOFILE, limits["nofile"])
            if nproc_safe():
                _set(resource.RLIMIT_NPROC, limits["procs"])
        except Exception:
            # A ceiling that cannot be applied must not break the operator's shell.
            pass

    return _pre


def describe(limits: Optional[dict] = None) -> str:
    """One-line, user-facing summary for status output; empty when disabled."""
    limits = limits if limits is not None else limits_for_child()
    if not limits:
        return ""
    parts = [
        f"mem {limits['mem_mb']} MB",
        f"cpu {limits['cpu_s']}s",
        f"file {limits['fsize_mb']} MB",
        f"fds {limits['nofile']}",
    ]
    if nproc_safe():
        parts.append(f"procs {limits['procs']}")
    return "local child limits: " + ", ".join(parts)
