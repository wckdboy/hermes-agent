"""Local child resource ceilings — E2E against real forked processes.

These assert the ceilings actually bind (a real fork is killed, a real write is refused), never that a
particular constant appears in source. Every test spawns through the same seam the local backend uses.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tools.environments import local_limits

pytestmark = pytest.mark.platforms("posix")


def _spawn(cmd, **kw):
    """Spawn exactly the way the local backend does: bash -c, new session, preexec_fn."""
    pre = local_limits.make_preexec(kw.pop("limits", None))
    return subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=60,
        preexec_fn=pre, start_new_session=True, **kw)


def test_disabled_env_yields_no_preexec(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_LIMITS", "0")
    assert local_limits.limits_for_child() is None
    assert local_limits.make_preexec() is None


def test_address_space_ceiling_kills_a_real_greedy_child(monkeypatch):
    """A python child told to allocate past the ceiling dies; it does not take the host with it."""
    monkeypatch.setenv("HERMES_LOCAL_LIMIT_MEM_MB", "256")
    proc = _spawn("python3 -c \"b = bytearray(600 * 1024 * 1024); print('allocated')\"")
    assert "allocated" not in proc.stdout
    assert proc.returncode != 0, "child exceeded its address-space ceiling but exited 0"


def test_file_size_ceiling_stops_a_runaway_write(monkeypatch, tmp_path):
    """A write past RLIMIT_FSIZE fails inside the child instead of filling the disk."""
    monkeypatch.setenv("HERMES_LOCAL_LIMIT_FSIZE_MB", "1")
    target = tmp_path / "big.bin"
    proc = _spawn(f"dd if=/dev/zero of={target} bs=1M count=16 status=none; echo rc=$?")
    assert proc.returncode != 0 or "rc=0" not in proc.stdout
    assert target.stat().st_size <= 2 * 1024 * 1024, "child wrote past its file-size ceiling"


def test_ceiling_is_inherited_by_grandchildren(monkeypatch):
    """The limit binds the whole subtree, not just the shell that was spawned."""
    monkeypatch.setenv("HERMES_LOCAL_LIMIT_MEM_MB", "256")
    proc = _spawn("bash -c 'python3 -c \"bytearray(600 * 1024 * 1024)\"' && echo grandchild-ok")
    assert "grandchild-ok" not in proc.stdout, "grandchild escaped the inherited ceiling"


def test_limits_do_not_break_ordinary_work(monkeypatch):
    """The default ceiling must be high enough that a normal command is untouched."""
    monkeypatch.delenv("HERMES_LOCAL_LIMIT_MEM_MB", raising=False)
    proc = _spawn("echo hello && python3 -c \"print(sum(range(1000)))\"")
    assert proc.returncode == 0
    assert "hello" in proc.stdout and "499500" in proc.stdout


def test_node_reaches_its_virtual_reservation_floor(monkeypatch):
    """Regression: a tight address-space cap makes V8 SIGTRAP with no output. The default must not.

    Skips (rather than silently passing) when node is absent, so this never becomes a false green.
    """
    if not any(os.access(os.path.join(p, "node"), os.X_OK) for p in os.environ["PATH"].split(os.pathsep) if p):
        pytest.skip("node not installed on this host")
    monkeypatch.delenv("HERMES_LOCAL_LIMIT_MEM_MB", raising=False)
    proc = _spawn("node -e \"console.log('v8 ok')\"")
    assert "v8 ok" in proc.stdout, f"node died under the default ceiling: rc={proc.returncode} {proc.stderr[:200]}"


def test_nproc_gate_is_off_unless_opt_in(monkeypatch):
    """Per-UID limits are unsafe on a shared host; they must not engage by accident."""
    monkeypatch.delenv("HERMES_LOCAL_LIMITS_NPROC", raising=False)
    assert local_limits.nproc_safe() is False
    monkeypatch.setenv("HERMES_LOCAL_LIMITS_NPROC", "1")
    assert local_limits.nproc_safe() is True


def test_describe_is_honest_about_what_is_applied(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_LIMIT_MEM_MB", "512")
    monkeypatch.delenv("HERMES_LOCAL_LIMITS_NPROC", raising=False)
    text = local_limits.describe()
    assert "512 MB" in text
    assert "procs" not in text, "describe claims a per-UID limit that is not being applied"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only ceilings")
def test_windows_returns_none(monkeypatch):
    monkeypatch.setattr(local_limits, "POSIX", False)
    assert local_limits.limits_for_child() is None
    assert local_limits.make_preexec() is None
