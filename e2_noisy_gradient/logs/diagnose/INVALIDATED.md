# Invalidated response-lag logs

All unversioned `*_lagdiag*.csv` files in this directory were generated before
the learning-rate scheduler fix. The diagnostic advanced
`CosineAnnealingLR(T_max=60)` once per minibatch, whereas the main experiments
advance it once per epoch. Consequently, response-time values derived from
these files—including the previously reported 1,317-to-47-step comparison—are
invalid and must not be cited.

Corrected runs use diagnostic protocol version 2. Their filenames contain
`_lagdiag_v2`, and both CSV outputs contain a `protocol_version` column. The
analysis script rejects unversioned legacy logs by default.
