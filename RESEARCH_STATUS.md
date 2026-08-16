# Adam-NV issue-resolution status

| Issue | Repository resolution | Remaining evidence |
|---|---|---|
| Lag diagnostic scheduler stepped per minibatch | Fixed; regression test added; v2 logs versioned | Rerun v2 suite on seeds 0--3 |
| Old 1,317-to-47-step claim | Legacy logs rejected and marked invalid; revised paper withdraws the claim | Replace only after protocol-v2 reruns |
| E1 baseline seed mismatch | Suite defaults to four seeds; analysis refuses missing seeds | Run seeds 2--3 for Adam, AdamW, AdaBelief |
| E14 incomplete run | Analysis detects and blocks publication output | Rerun intermittent, coupled beta2=0.95, seed 1 |
| Independent/Welch testing | Replaced by paired contrasts with confidence intervals | Regenerate final table after missing run |
| Multiple comparisons | Holm adjustment implemented by experiment family | Use adjusted values in paper |
| Missing direct interactions | E13 schedule-by-optimizer gap contrasts implemented | Interpret cautiously at current sample sizes |
| Overstated E15 claims | Exploratory reanalysis shows no Holm-significant E15 contrast; revised conclusions are conservative | Regenerate after the incomplete run |
| “Byte-identical noise” wording | Corrected to matched random directions with run-specific magnitudes in code and paper | None |
| Noise measure called variance estimator | Code and revised paper label it a heuristic noise score | None |
| Controller direction unsupported | Alternative smoothing response and suite implemented | Run direction suite |
| Label memorization transition inferred only | Clean/noisy-label subset diagnostics implemented | Rerun four paired seeds and plot curves |
| Timing claims weak | Repeated synchronized benchmark with metadata and CIs implemented | Run on target GPU |
| CIFAR-100 absent | CIFAR-100 support and four-seed suite implemented | Run E16 |
| README/dependencies missing | Added setup, authoritative entry points, commands, requirements | Record exact environment after final runs |
| Optimizer/unit tests absent | Added schedule, equivalence, decoupling, direction, protocol, and stats tests | Execute PyTorch tests on training machine |
| E13--E15 absent from old paper | Code/statistical pipeline prepared; revised DOCX records exploratory results and completeness limits | Replace exploratory values after validated reruns |

## Current exploratory statistical result

Using the trailing-five-epoch metric and existing saved logs:

- E13: no Adam-NV versus Adam contrast is significant after Holm correction.
- E13 schedule-by-optimizer interaction contrasts are not significant.
- E15: no decoupled-versus-coupled contrast is significant after Holm correction.
- E14: coupled `beta2=0.95` is worse than default on ramp-up in the
  Holm-adjusted paired t-test, but the exact four-seed sign-flip p-value is
  `0.125`; treat this as suggestive, not definitive.
- The E14/E15 intermittent `0.95` contrasts remain incomplete because seed 1
  ends at epoch 32.

The generated exploratory table is
`analysis/results/e13_e15_primary_statistics.csv`. Do not use it as the final
paper table until the incomplete run is replaced and the analysis succeeds
without `--allow-incomplete`.
