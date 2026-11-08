"""holdout4 is generated exactly like holdout3, with new seeds (PREREG_holdout4.md)."""
from paper_budget.datagen.gen_holdout4 import SEED_BASE_H4, SETTINGS_H4, build_spec_h4, corpus


def test_spec_reproduces_holdout3_with_its_seed_base():
    assert build_spec_h4(corpus.SEED_BASE["holdout3"], "holdout3") == corpus.build_spec("holdout3")


def test_settings_equal_holdout3():
    assert SETTINGS_H4 == corpus.SETTINGS["holdout3"]


def test_seeds_are_new():
    used = set()
    for split in ("train", "holdout", "holdout2", "holdout3"):
        used |= {kw["seed"] for _, _, kw in corpus.build_spec(split)}
    new = {kw["seed"] for _, _, kw in build_spec_h4()}
    assert len(new) == 9 * corpus.H3_MAX_ATTEMPTS
    assert not new & used
    assert min(new) == SEED_BASE_H4 + 1


def test_circuit_ids_are_holdout4():
    assert all(cid.startswith("holdout4_") for cid, _, _ in build_spec_h4())
