# Experiment entry points

Use `part_b_resnet_v8.py` for all new E2/E13/E14/E15 training. The files
`part_b_resnet.py` and `part_b_resnet_v1.py` through `part_b_resnet_v7.py` are
historical snapshots retained only to explain how the experiment evolved.
They are not authoritative and should not be used for new reported results.

Use `diagnose_response_lag.py` for step-level response diagnostics.
`diagnose_response_lag_v1.py` is retained for provenance only.

Authoritative suite runners:

- `run_nonstationary_noise_suite.py` — E13
- `run_beta2_sweep_suite.py` — E14
- `run_vnoise_suite.py` — E15
- `diagnose_response_lag.py` — corrected protocol-v2 lag diagnostics
- `run_response_lag_v2_suite.py` — corrected multi-seed lag reruns
- `run_cifar100_generality_suite.py` — E16 CIFAR-100 generality test
- `run_controller_direction_suite.py` — controller-direction ablation
