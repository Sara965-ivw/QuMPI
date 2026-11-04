# Predictions recorded BEFORE running obs_structure.py (exploratory, holdout3 v3.1)

Holdout3 is used here for design decisions only; it is not a test set for the paper.

## A. Is the device-induced error mostly multiplicative damping?
A1. For observables with |mu*| > 0.2, at least 80% of ratios mu/mu* lie in [0, 1.05].
A2. For single-qubit Z observables with |mu*| < 0.02, the median noisy value mu is
    positive (T1 relaxation pushes qubits towards |0>).
Decision: if A1 holds, the paper's learning target becomes the attenuation f;
otherwise the model predicts both f and an offset o.

## B. Does a cut inside the backward light cone of S raise the error on S?
B1. Per (circuit, observable), mean log10 error of candidates with >= 1 cut inside
    the light cone minus candidates with none: median > 0, and > 0 in at least 60%.
Decision: if B1 holds, the model is conditioned on the observable and its light cone;
otherwise fragment-graph conditioning (innovation 1) takes priority.
