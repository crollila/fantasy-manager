"""Feature-set selection and matrix preparation.

Feature sets are named lists of columns drawn from the feature sidecar families so every
experiment states exactly which information it used. ``market`` is excluded from every
market-free set by construction.
"""
from __future__ import annotations
import re
import numpy as np
import polars as pl

COMPACT_FORM_METRICS = ["points", "points_allowed", "margin", "epa_play", "epa_pass", "epa_rush", "epa_neutral", "epa_play_allowed", "epa_pass_allowed", "epa_rush_allowed", "epa_neutral_allowed", "success_rate", "success_rate_allowed",
                        "points_drive", "points_drive_allowed", "explosive20", "explosive20_allowed", "sack_rate", "sack_rate_allowed", "turnover_drive", "turnover_drive_allowed", "three_out_rate", "three_out_rate_allowed",
                        "sec_per_play_neutral", "pass_rate_neutral", "proe", "cpoe", "cpoe_allowed", "fg_made_rate", "st_epa", "rz_td_rate", "rz_td_rate_allowed", "third_conv", "third_conv_allowed", "plays", "int_rate", "fumble_lost_rate", "giveaways", "giveaways_allowed", "epa_drive", "epa_drive_allowed"]
COMPACT_WINDOWS = ("ewm", "std", "prev", "l8", "l3")
COMPACT_QB = ["qb_epa_career", "qb_epa_season", "qb_epa_last4", "qb_epa_ewm", "qb_epa_prev_season", "qb_cpoe_career", "qb_sack_rate_career", "qb_int_rate_career", "qb_success_career", "qb_anya_career", "qb_dropbacks_career", "qb_dropbacks_season",
              "qb_starts_career", "qb_no_history", "qb_changed", "qb_new_to_team", "qb_log_dropbacks", "qb_draft_round", "qb_age", "qb_seasons_exp", "qb_epa_neutral_career", "qb_scramble_rate_career"]
COMPACT_AVAIL = ["value_lost_off", "value_lost_def", "inj_lost_QB", "inj_lost_RB", "inj_lost_WR", "inj_lost_TE", "inj_lost_OL", "inj_lost_DL", "inj_lost_LB", "inj_lost_DB", "starters_out_off", "starters_out_def", "starters_questionable",
                 "qb_starter_listed_out", "qb_starter_questionable", "reserve_lost_off", "reserve_lost_def", "reserve_lost_QB", "reserve_lost_OL", "injury_source_available", "injury_listings", "dnp_count"]


def feature_sets(meta: dict, columns: list[str]) -> dict[str, list[str]]:
    fam = meta["families"]
    present = set(columns)

    def pick(names):
        return [c for c in names if c in present]

    side = lambda names: [f"{s}_{n}" for s in ("home", "away") for n in names]  # noqa: E731
    compact_form = side([f"{m}__{w}" for m in COMPACT_FORM_METRICS for w in COMPACT_WINDOWS] + ["games_played_std", "games_played_total", "wins__std", "wins__l5", "win_pct__prev", "epa_play__sd8", "margin__sd8", "epa_play__trend", "margin__trend"])
    base = pick(fam.get("schedule", []) + fam.get("rest_travel", []) + fam.get("weather", []) + fam.get("coaching", []) + fam.get("officials", []) + fam.get("elo", []) + fam.get("ratings_margin", []) + fam.get("ratings_efficiency", []) + fam.get("matchup", []))
    diffs = [c for c in columns if c.startswith("diff_")]
    compact = list(dict.fromkeys(base + diffs + pick(compact_form) + pick(side(COMPACT_QB)) + pick(side(COMPACT_AVAIL))))
    all_features = [c for f, cols in fam.items() for c in cols if c in present]
    full = [c for c in all_features if not c.startswith("mkt_")]
    market = pick(fam.get("market", []))
    core_only = [c for c in compact if not any(c.startswith(p) for p in ("home_inj", "away_inj", "home_value_lost", "away_value_lost", "home_starters", "away_starters", "home_qb_starter", "away_qb_starter", "home_reserve", "away_reserve", "home_injury", "away_injury", "home_dnp", "away_dnp", "diff_value_lost", "diff_inj"))]
    return {"compact": compact, "full": full, "core": core_only, "compact_market": compact + market, "full_market": full + market, "market_only": market + ["ctx_neutral", "ctx_season"], "elo_only": pick(["elo_diff", "elo_prob_home", "elo_margin", "ctx_neutral"])}


def family_columns(meta: dict, family: str) -> list[str]:
    return list(meta["families"].get(family, []))


def to_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()


def season_weights(seasons: np.ndarray, current: int, half_life: float | None) -> np.ndarray:
    if not half_life:
        return np.ones(len(seasons))
    age = current - np.asarray(seasons, dtype=float)
    return 0.5 ** (age / half_life)
