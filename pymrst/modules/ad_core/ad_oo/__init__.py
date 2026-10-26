"""MRST's AD-OO framework, ported faithfully beside PyMRST's legacy contract.

`physical_model`, `reservoir_model`, `nonlinear_solver` and
`simulate_schedule_ad` follow MRST's own classes and functions: equations come
back as a `LinearizedProblem`, increments as one block per primary variable,
and driving forces are passed into every call. PyMRST's earlier models use the
legacy classes in `pymrst.modules.ad_core.models`, which keep their behaviour.
"""
