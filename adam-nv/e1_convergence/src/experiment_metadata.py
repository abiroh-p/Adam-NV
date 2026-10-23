"""Runtime metadata sidecars for reproducible E1 experiment logs."""

import hashlib
import json
import platform
from pathlib import Path
import socket
import sys

import torch
import torchvision


_HASHED_FILES = ("train_resnet.py", "adam_nv.py", "gono.py", "experiment_metadata.py")


def _code_hashes() -> dict:
    """SHA-256 of the source files that define a run's behaviour."""
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in _HASHED_FILES if (here / name).exists()}


def write_metadata(csv_path: str, arguments: dict, script_version: str) -> str:
    path = Path(csv_path).with_name(Path(csv_path).stem + "_metadata.json")
    payload = {
        "script_version": script_version,
        "command": [sys.executable, *sys.argv],
        "arguments": arguments,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "code_sha256": _code_hashes(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
