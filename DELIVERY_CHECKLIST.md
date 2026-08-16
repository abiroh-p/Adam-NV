# Adam-NV completion checklist

The repository-level issues are resolved. The following steps require the
target PyTorch/CUDA training environment and produce the evidence still needed
for a submission-ready paper.

## Execute in this order

1. Create the environment and run the tests in `README.md`. No PyTorch test may
   be skipped on the training machine.
2. Complete E1 seeds 2 and 3 for Adam, AdamW, and AdaBelief.
3. Replace the incomplete E14 intermittent `beta2=0.95`, seed-1 run.
4. Run `analysis/analyze_e13_e15.py` without `--allow-incomplete`. Publication
   output must succeed before using the table in a manuscript.
5. Run the protocol-v2 response-lag suite. Never combine these results with the
   invalid legacy diagnostic.
6. Run the controller-direction ablation.
7. Run the four-seed CIFAR-100 suite.
8. Regenerate label-noise runs with direct corrupted-subset diagnostics.
9. Run the synchronized overhead benchmark on the target GPU.
10. Update the revised papers only from complete outputs, rerender both DOCX
    files, and visually inspect every page before submission.

Exact commands, expected directories, statistical policy, and validity gates
are in `README.md`. Current evidence and issue status are in
`RESEARCH_STATUS.md`.

## Submission gates

- Every declared condition has all four seeds and 60 epochs.
- Protocol-v2 is present in every response-lag CSV.
- Paired confidence intervals, exact sign-flip checks, and Holm-adjusted
  p-values are reported.
- Cross-condition claims use direct contrasts or interaction tests.
- The paper calls `sigma2` a heuristic noise score, not a calibrated variance.
- Null findings say “no detectable advantage under the tested conditions,” not
  equivalence or absence.
- Runtime claims come only from the synchronized benchmark with uncertainty.
