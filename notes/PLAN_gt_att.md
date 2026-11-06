# Plan: Graph Transformer as a residual attenuation model (draft for review, 6 Nov 2026)

Status: plan only. No code is changed until Sara approves this plan. Hypotheses become
`notes/PREREG_gt_att.md`, committed before any training run and before any held-out evaluation.
Supervisor approval of the budget-aware objective (error at N shots instead of fidelity): stated by
Sara on 6 Nov 2026; to be recorded in `notes/DECISIONS_objective.md`.

---------------------------------------------------------------------------------------------------
## 1. Mathematical formulation

### 1.1 Quantities from the labels (exact, no shot noise)
For circuit c, candidate cut k and observable S in the basket:
- μ*_S        ideal expectation value (obs_ideal)
- μ_kS        noisy expectation value of cut k with unlimited shots (obs_mu)
- κ²_k        sampling overhead of cut k (gamma2 in code; 16 per wire cut, (1 + 2|sin θ|)² per gate cut)

Informative observables: I_c = { S : |μ*_S| > 0.2 } (same threshold as headroom.py).

**Target (attenuation label).**
  a_k = (1/|I_c|) Σ_{S ∈ I_c} −ln( μ_kS / μ*_S )
Identical to the definition in headroom.load (consistency test T1). Observed ratios μ/μ* lie in
[0, 1.05] (obs_structure), so a_k ≥ −0.05 in the data; a_k is not clipped for training.

**True error at N shots** (paper_budget.targets, used only for evaluation):
  MSE_k(N) = b²_k + v_k / N,  b²_k = mean_S (μ_kS − μ*_S)²,  v_k = mean_S max(κ²_k − μ²_kS, 0).

### 1.2 ESP (estimated success probability) and its assumptions
Layout-ESP of cut k, as implemented in quest_phase2/hardware.layout_features (frozen code):
  ln ESP_k = Σ_{1q gates} ln(1 − e_sx(p))
           + Σ_{2q gates} ln(1 − e_2q(p₁, p₂))
           + Σ_{cut gates, both halves} [ ln(1 − e_sx(p)) + ln(1 − 2 e_ro(p)) ]
           + Σ_{wire-cut measurements} ln(1 − 2 e_ro(p))
           + Σ_{output qubits} ln(1 − e_ro(p))
summed over both fragments, with p the physical qubit of each fragment's layout (phys_a, phys_b).
For an uncoupled pair at coupling distance d: e_2q = 1 − (1 − ē_ecr)^(1 + 3(d − 1)) (a SWAP = 3 ECRs).
Feature used below: e_k = −ln ESP_k ≥ 0.

Assumptions (each a stated limitation):
- A1 errors are independent and gate-local (matches the labeller's NoiseModel.from_backend);
- A2 no idle decoherence (T1/T2 during waiting) and no crosstalk;
- A3 routing cost from coupling distance, not from the compiled circuit (approximates Nishio et al. 2020);
- A4 initial layout only (the records store initial, not final, layouts);
- A5 one attenuation per cut for all observables (obs_structure: noise acts as damping; per-observable
  differences exist, per_observable.py, and are out of scope here).

### 1.3 Models (all predict â_k; all trained on the training split only)
Prior (fixed after fitting on the training circuits, ordinary least squares):
  p_k = α + β e_k

M0 **ESP only**:            â_k = p_k
M1 **ESP + linear correction**:
  â_k = p_k + wᵀ (z_k − z̄_c),  z_k = [e_k, 7 cut features],  z̄_c = mean over the circuit's candidates
  w by least squares on the training residuals (a − p) after within-circuit demeaning.
M2 **ESP + Graph Transformer residual (candidate model)**:
  â_k = p_k + r_θ(G_k) − (1/|C_c|) Σ_{j ∈ C_c} r_θ(G_j)
  r_θ = Graph Transformer (frozen RQ1 architecture gt_size: additive readout, size-free globals,
  random-walk PE, layout hardware features), last readout layer initialised to zero.
M3 **Direct Graph Transformer**: â_k = g_θ(G_k), same architecture, no prior, standard initialisation.
Secondary (reported, not part of the decision): GBM residual (Step 8), minimum-κ² rule, IBM find_cuts,
Qdislib (holdout3 labels from compare_tools).

**Zero initialisation (M2).** With the last layer's weight and bias at zero, r_θ ≡ 0, so the untrained
M2 equals M0 exactly (unit test T3). Early stopping starts from this state, so if training never
improves the validation loss, M2 returns M0: the model cannot do worse than ESP on validation.

**Within-circuit demeaning.** Σ_k [r_θ(G_k) − r̄_c] = 0, so the network cannot change a circuit's
average attenuation; the level comes from the prior. Why: about 90% of label variance is between
circuits (RQ1 diagnosis) and is already carried by ESP; the decision depends on differences between
cuts of one circuit.

### 1.4 Loss
Equal weight per circuit:
  L(θ) = (1/|T|) Σ_{c ∈ T} (1/|C_c|) Σ_{k ∈ C_c} (â_k − a_k)²
For M2, write u_k = a_k − p_k and ũ_k = u_k − ū_c. Then
  (1/|C_c|) Σ_k (â_k − a_k)² = (1/|C_c|) Σ_k (r̃_k − ũ_k)² + ū_c²,  r̃_k = r_θ(G_k) − r̄_c
(cross term vanishes because Σ_k r̃_k = 0). The second term does not depend on θ: M2 is trained
only on the within-circuit part of what ESP misses. M3 uses the same loss on the raw â.
Alternatives considered: ranking losses (RQ1) ignore magnitudes, which the budget rule needs;
Huber loss (robust) kept as a sensitivity check only if residual outliers appear in training.

### 1.5 Physically valid predictions
Predicted damping f̂_k = exp(−â_k). Bounds:
  â_k ← clip(â_k, a_min, a_max),  a_min = min training label (expected ≈ −0.05), a_max = −ln 0.01
f > 1 is allowed up to the observed maximum because amplitude damping (T1) can increase ⟨Z⟩ toward
|0⟩; a hard f ≤ 1 would be wrong for those observables. Clipping is applied only at prediction time,
never inside the loss (it would zero the gradients); the share of clipped predictions is reported.
A softplus parametrisation was rejected: it breaks the exact M2-at-initialisation = M0 property.
Variance estimate clipped at zero: v̂_k = max(κ²_k − f̂²_k m, 0).

### 1.6 Selection rule and regret
Chosen cut at budget N (deployable_rule.py):
  k̂(N) = argmin_k (1 − f̂_k)² m + (κ²_k − f̂²_k m) / N,  m = mean_S μ*_S²
Primary: true m (isolates the predictor; Step 7 showed m costs ≤ 3.9%). Secondary: 10% pilot m.
**Budget-aware regret:** ρ_c(N) = MSE_{k̂(N)}(N) / min_k MSE_k(N) ≥ 1 (true labels),
reported as the geometric mean over circuits and the share of circuits with ρ > 1.10, at N = 10³…10⁹.

---------------------------------------------------------------------------------------------------
## 2. Metrics
Prediction (held-out circuits, seed-averaged predictions):
- RMSE and MAE of â vs a, pooled and within-circuit (after demeaning both);
- R² pooled and within-circuit;
- Kendall τ_b between â and a within each circuit, averaged over circuits;
Decision: regret ρ(N) at 7 budgets (primary: N = 10⁸ and 10⁹, where Step 3 located the headroom).
Statistics: paired Wilcoxon signed-rank over circuits; Holm over all pre-registered tests; 95% bootstrap
CI (10,000 resamples of circuits) for each geometric-mean regret ratio; seed spread (sd over 5 seeds).
Meaningful effect: ≥ 2% lower geometric-mean regret (Step 8's linear gain was 2% and not significant).

---------------------------------------------------------------------------------------------------
## 3. Data requirements
| Split | Use | Status |
|---|---|---|
| train_v31 (210 circuits, 6–14 qubits) | train 80% / validation 20%, stratified by (family, width), seed 2026, circuit-level | relabelling, ~17–21 h left; 33/33 checked: same candidates, fidelity differs only by layout ties (5 candidates) |
| holdout3 (16–20 qubits) | exploratory test (already used for design analyses) | ready |
| holdout4 (fresh, 16–20 qubits, PYTHONHASHSEED=0) | confirmatory test, once everything is frozen | to generate |
Training filter: ≥ 2 candidates and ≥ 1 informative observable. Evaluation filter: as in Steps 7–8
(≥ 3 candidates, ≥ 1 informative observable). The number of usable training circuits is counted
before training; if it is below 100, the plan is reviewed before training (overfitting risk).
Leakage controls: circuit IDs disjoint across splits (asserted); the training script refuses to run
if QUEST_UNBLIND is set or a holdout path is passed; α, β, w and early stopping use training/validation only.
Known shift: training circuits are smaller (6–14 qubits) than test circuits (16–20). The ESP prior is
size-extensive, which is why it carries the level; the residual only models within-circuit differences.

---------------------------------------------------------------------------------------------------
## 4. Implementation plan (each step is a separate commit by Sara; nothing in stage2_predictor/ changes)
1. `notes/RESULTS_relabel_train.md`: completion and --check results of the relabelling.
2. `paper_budget/att_data.py`: wraps the frozen quest_phase2.data.load_split (aggregation "layout") and
   attaches a, informative mask, m, b², v, κ², e. Tests T1 (label equals headroom.load on a sample),
   T2 (split disjointness).
3. `paper_budget/models_att.py`: M1, M2 (wrapper around the frozen GraphTransformer gt_size backbone,
   zero-initialised readout, prior and demeaning outside the network), M3. Tests T3 (untrained M2 = M0
   exactly), T4 (residual sums to zero per circuit), T5 (clipping bounds).
4. `paper_budget/train_gt_att.py`: frozen HP of run_rq1 (d 64, 4 heads, 3 layers, dropout 0.1, lr 3e-4,
   weight decay 1e-4, ≤ 80 epochs, patience 15, 8 circuits per step); early stopping on validation loss;
   5 seeds × {M2, M3}; `--smoke` mode first. Outputs only in runs/paper_budget/gt_att/; the writer refuses
   to overwrite an existing checkpoint; config, git hash and package versions are logged.
5. `notes/PREREG_gt_att.md`: hypotheses, metrics and decision criteria, committed BEFORE step 4 runs.
6. Train on train_v31 (no held-out data read).
7. `paper_budget/eval_gt_att.py` on holdout3 (exploratory) → `notes/RESULTS_gt_att.md`.
8. holdout4 generation and the confirmatory evaluation, with the same frozen code and checkpoints.

---------------------------------------------------------------------------------------------------
## 5. Draft hypotheses (to be fixed in PREREG_gt_att.md)
- H1 (fit): M2 within-circuit R² > M1 within-circuit R² on the test circuits.
- H2 (decision, primary): M2 geometric-mean regret at 10⁸ and 10⁹ is ≥ 2% lower than M1,
  Holm-adjusted p < 0.05 at both budgets.
- H3 (value of the prior): M2 regret < M3 regret at 10⁸ and 10⁹ (Holm p < 0.05).
- H4 (no harm): M2 regret is never more than 1% above M0 at any budget.
Decision criteria: "the Graph Transformer improves budget-aware cut selection" is claimed only if H2
holds on holdout3 AND on holdout4. If H2 fails, the thesis reports M1 as the best predictor and the
Transformer result as negative. H1 without H2 is reported as "better fit, no decision gain".

## 6. Main risks (stated in advance)
Small training set (≈ 200 circuits before filtering); width extrapolation (6–14 → 16–20 qubits); the
remaining headroom is small (about 8% at 10⁹ on holdout3), so a real but small gain may not reach
significance with ~160 test circuits; holdout3 is no longer clean.
