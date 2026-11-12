"""Collect environment metadata recorded with every run."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

from groundlab import __version__

PACKAGES = ("bm25s", "PyStemmer", "numpy", "pytrec-eval-terrier", "torch", "psutil")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
    except ImportError as exc:
        info["error"] = f"torch import failed: {exc}"
        return info
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["devices"] = (
        [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "capability": list(torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ]
        if info["cuda_available"]
        else []
    )
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["nvidia_driver"] = smi.splitlines()[0] if smi else None
    return info


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def git_info(repo_root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def redact_home(path: str) -> str:
    """Replace the user's home directory with `~` so records carry no username."""
    home = str(Path.home())
    return "~" + path[len(home) :] if path.lower().startswith(home.lower()) else path


def collect(repo_root: Path, disk_path: Path | None = None) -> dict[str, Any]:
    disk_path = disk_path or repo_root
    usage = shutil.disk_usage(disk_path)
    return {
        "collected_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "groundlab": __version__,
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": redact_home(sys.executable),
            "bits": 64 if sys.maxsize > 2**32 else 32,
        },
        "cpu": {
            "name": cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": os.cpu_count(),
        },
        "memory_total_bytes": psutil.virtual_memory().total,
        "disk": {
            "path": redact_home(str(disk_path)),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        },
        "gpu": gpu_info(),
        "packages": package_versions(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "git": git_info(repo_root),
    }
