# AD-core validation layout

This tree mirrors `modules/ad-core` in the local MRST reference checkout.
New tests belong under the matching subsystem directory:

```text
tests/ad_core/
  backends/
  initialization/
  models/
  simulators/
  solvers/
  statefunctions/
  timesteps/
  upscale/
  utils/
  workflows/
```

Use three validation levels:

1. Unit parity: identical inputs and outputs for one MRST function/class.
2. Subsystem parity: interacting objects such as backend plus state functions.
3. Workflow parity: complete schedules, restart, wells, and adjoints.

Every MATLAB-backed test should identify its MRST symbol in the test docstring
so `scripts/audit_ad_core_parity.py` can associate it with the parity ledger.
Self-tests without a MATLAB result should say so explicitly.
