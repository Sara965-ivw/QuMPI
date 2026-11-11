# holdout4: n = 156 vs 144
144 = sampling stopping rule (16 eligible circuits x 9 cells, PREREG_holdout4.md §3).
156 = pre-registered analysis set (§3: every labelled holdout4 circuit with >= 3 candidates,
>= 1 informative observable, layout-ESP for every candidate; the holdout3 filter).
eval_holdout4.py loads the analysis set via load_test("holdout4"); it does not read
holdout4_selection.json. No deviation from the pre-registration.

Filter check (att_data.py): usable_eval() = n_cand >= 3 AND all(isfinite(att)) AND esp_missing == 0.
att = mean over informative observables (|mu*| > 0.2) of -ln(mu_kS/mu*_S), so isfinite(att) requires
>= 1 informative observable and a positive noisy/ideal ratio for every candidate. Matches PREREG §3.
