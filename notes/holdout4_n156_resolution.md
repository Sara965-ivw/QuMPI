# holdout4: n = 156 vs 144
144 = sampling stopping rule (16 eligible circuits x 9 cells, PREREG_holdout4.md §3).
156 = pre-registered analysis set (§3: every labelled holdout4 circuit with >= 3 candidates,
>= 1 informative observable, layout-ESP for every candidate; the holdout3 filter).
eval_holdout4.py loads the analysis set via load_test("holdout4"); it does not read
holdout4_selection.json. No deviation from the pre-registration.
