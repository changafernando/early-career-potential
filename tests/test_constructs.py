from ecp.constructs import build_cohort, split_masks
from ecp.seasons import build_player_seasons


def test_rate_construct_rewards_efficiency_and_counts_washouts_as_zero(cfg, toy_league):
    pts, teams, players = toy_league
    ps = build_player_seasons(pts, teams, cfg)
    coh = build_cohort(ps, players, cfg)
    # player 1 (efficient, long career) is a contributor; player 2 washed out after 3 seasons
    assert coh.loc[1, "y_rate"] == 1
    assert coh.loc[2, "y_rate"] == 0
    # player 3 plays big minutes but is inefficient: never in the top third of rotation players
    assert coh.loc[3, "y_rate"] == 0


def test_debut_classes_without_an_observed_year_8_are_unlabelled(cfg, toy_league):
    pts, teams, players = toy_league
    ps = build_player_seasons(pts, teams, cfg)
    c2 = dict(cfg, cohort=dict(cfg["cohort"], last_labelled_debut=1996))
    coh = build_cohort(ps, players, c2)
    assert coh.loc[coh.debut == 1997, "y"].isna().all()
    assert coh.loc[coh.debut == 1996, "y"].notna().all()


def test_split_is_forward_chaining(cfg, toy_league):
    pts, teams, players = toy_league
    ps = build_player_seasons(pts, teams, cfg)
    coh = build_cohort(ps, players, cfg)
    m = split_masks(coh, cfg)
    assert not (m["train"] & m["test"]).any()
    if m["test"].any() and m["train"].any():
        assert coh.debut[m["train"]].max() < coh.debut[m["test"]].min()
