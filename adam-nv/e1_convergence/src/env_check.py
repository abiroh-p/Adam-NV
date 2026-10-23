"""Environment guard for Protocol v2 runs.

Protocol v2 requires every new run to use the SAME software stack as the
existing E1 results (Kaggle, Python 3.12.13, PyTorch 2.10.0+cu128,
torchvision 0.25.0+cu128). Mixing stacks is what created the E1 provenance
problem, so run this before launching anything:

    python3 env_check.py

Exit code 0 = environment matches; 2 = mismatch (runs should NOT be started).
Pass --allow-mismatch only if you deliberately accept the deviation, and
then record it in the paper.
"""

import argparse
import platform
import sys

import torch
import torchvision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-python", default="3.12.13")
    ap.add_argument("--expect-torch", default="2.10.0+cu128")
    ap.add_argument("--expect-torchvision", default="0.25.0+cu128")
    ap.add_argument("--allow-mismatch", action="store_true")
    args = ap.parse_args()

    found = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    expected = {
        "python": args.expect_python,
        "torch": args.expect_torch,
        "torchvision": args.expect_torchvision,
    }
    mismatches = [k for k in expected if found[k] != expected[k]]

    print("Environment check")
    for k in expected:
        flag = "OK      " if k not in mismatches else "MISMATCH"
        print(f"  [{flag}] {k:12s} found={found[k]:<16s} expected={expected[k]}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA available : {cuda}")
    print(f"  CUDA runtime   : {torch.version.cuda}")
    print(f"  cuDNN version  : {torch.backends.cudnn.version()}")
    n = torch.cuda.device_count()
    print(f"  GPU count      : {n}")
    for i in range(n):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    if not cuda:
        print("\nFAIL: CUDA is not available. Do not start training runs on CPU.")
        return 2
    if mismatches and not args.allow_mismatch:
        print(f"\nFAIL: software stack differs from the E1 stack ({', '.join(mismatches)}).")
        print("Do not start Protocol v2 runs in this environment "
              "(or pass --allow-mismatch and document the deviation).")
        return 2
    print("\nPASS" + (" (mismatch explicitly allowed)" if mismatches else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
