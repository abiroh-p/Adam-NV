"""Runtime metadata sidecars for reproducible E1 experiment logs."""

import json
import platform
from pathlib import Path
import socket
import sys

import torch
import torchvision


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
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
