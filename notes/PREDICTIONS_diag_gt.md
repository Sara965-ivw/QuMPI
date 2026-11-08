# Predictions: stage-1 diagnostics of the attenuation Transformer (diag_gt_att.py)

Written and committed before running. Training data only (train_v31); no holdout is read.
Motivation: notes/RESULTS_gt_att_holdout3.md (H2 not met). Exploratory: nothing here can support a
claim; it decides which model, if any, is pre-registered for the confirmatory test on holdout4.

## Design (fixed)
- 168 train_v31 circuits usable for training and evaluation; 5-fold StratifiedGroupKFold by family,
  seed 2026; out-of-fold predictions pooled over all circuits.
- Training fractions 25%, 50%, 100% of each fold's training circuits (nested subsets); 20% of them
  (at least 5) for early stopping.
- Models: M0, M1 (closed form); M2 (residual on M0), M2b (residual on M1), M3 (direct); 3 seeds each,
  predictions averaged; frozen RQ1 hyperparameters (no tuning); true m for selection.
- Metrics: within-circuit R^2, Kendall tau_b, geometric-mean regret at 1e8 and 1e9.

## Questions and predictions
- D1 (data). Does the Transformer residual improve with more data?
  Prediction: no. M2's within-circuit R^2 rises by less than 0.05 from 50% to 100% of the data.
- D2 (calibration). Does the graph add information after the linear calibration?
  Prediction: no. M2b's within-circuit R^2 is less than M1's + 0.03 at 100% data, and M2b's regret
  is within 1% of M1's at 1e8 and 1e9.
- D3 (direct model). Does M3's within-circuit ranking improve with data?
  Prediction: yes. M3's Kendall tau_b rises by at least 0.05 from 25% to 100%.

## Decision rule
- D1 rises by >= 0.05 (learning curve still climbing): generating more training circuits is justified;
  estimate the cost before deciding.
- D2 met in the other direction (M2b R^2 >= M1 + 0.03 and regret >= 2% lower at 1e8 and 1e9): M2b
  becomes the candidate model for the holdout4 pre-registration.
- Both predictions D1 and D2 hold (flat curve, no gain over M1): the basket-level Transformer is not
  pursued further; stage 2 (readout test, then observable-conditioned prediction) is next.

## Limitations stated in advance
Cross-validation on 6-14 qubit circuits only (the size shift to 16-20 qubits is not tested here);
3 seeds; untuned hyperparameters; 25% subsets give about 27 training circuits per fold.
