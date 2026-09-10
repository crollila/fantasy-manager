"""Unit tests for the NFL forecasting engine (synthetic data; no network, no real data files)."""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import polars as pl
import pytest
from app.nfl.reference.teams import canonical, venue_for_game, haversine_miles, DISPLAY
from app.nfl.normalize import kickoff_utc
from app.nfl.features.rolling import prepare_team_games, rolling_features
from app.nfl.features.ratings import elo_features
from app.nfl.features.availability import miss_probability, availability_features
from app.nfl.features.context import market_features
from app.nfl.models.metrics import log_loss, brier, expected_calibration_error, evaluate, bootstrap_difference, crps_samples
from app.nfl.models.ensemble import fit_weights, apply_weights
from app.nfl.models.calibration import win_probability_from_margin, ProbabilityCalibrator, fit_residual_model, margin_sd
from app.nfl.models.simulate import simulate_game
from app.nfl.models.registry import Registry

UTC = timezone.utc


def synthetic_games(n_weeks=6, teams=("AAA", "BBB", "CCC", "DDD"), season=2020, start=datetime(2020, 9, 13, 17, tzinfo=UTC)):
    rows = []
    rng = np.random.default_rng(3)
    for w in range(1, n_weeks + 1):
        order = list(teams)
        rng.shuffle(order)
        for i in range(0, len(order), 2):
            home, away = order[i], order[i + 1]
            ko = start + timedelta(days=7 * (w - 1))
            hs, as_ = int(rng.integers(10, 35)), int(rng.integers(10, 35))
            rows.append({"game_id": f"{season}_{w:02d}_{away}_{home}", "season": season, "week": w, "game_type": "REG", "playoff": False, "kickoff_utc": ko, "home_team": home, "away_team": away, "home_score": hs, "away_score": as_, "completed": True, "neutral": False,
                         "margin": hs - as_, "total_points": hs + as_, "spread_line": 3.0, "total_line": 44.5, "home_moneyline": -150.0, "away_moneyline": 130.0})
    return pl.DataFrame(rows).with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))


def synthetic_team_games(games: pl.DataFrame):
    rows = []
    rng = np.random.default_rng(5)
    for g in games.to_dicts():
        for side, other in (("home_team", "away_team"), ("away_team", "home_team")):
            rows.append({"game_id": g["game_id"], "team": g[side], "opponent": g[other], "is_home": side == "home_team", "season": g["season"], "week": g["week"], "kickoff_utc": g["kickoff_utc"], "game_type": "REG", "playoff": False,
                         "epa_play": float(rng.normal(0, 0.1)), "points_drive": float(rng.uniform(1, 3)), "fg_att": 2, "fg_made": 1, "sec_per_play_neutral": 30.0, "success_rate": 0.45})
    return pl.DataFrame(rows).with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))


def test_team_canonicalisation_and_venues():
    assert canonical("STL") == "LA" and canonical("SD") == "LAC" and canonical("OAK") == "LV" and canonical("WSH") == "WAS"
    assert canonical(None) is None and DISPLAY["LA"] == "LAR"
    london = venue_for_game("JAX", 2019, "Wembley Stadium", True)
    assert london["tz"] == "Europe/London"
    home = venue_for_game("GB", 2019, "Lambeau Field", False)
    assert home["matched"] == "home" and abs(home["lat"] - 44.5) < 0.1
    assert 2300 < haversine_miles(40.8, -74.07, 33.95, -118.34) < 2600


def test_kickoff_conversion_handles_missing_time():
    ko, imputed = kickoff_utc("2024-09-08", "13:00")
    assert ko.hour == 17 and not imputed  # 13:00 ET in September = 17:00 UTC
    ko2, imputed2 = kickoff_utc("1999-09-12", None)
    assert imputed2 and ko2 is not None
    assert kickoff_utc(None, None) == (None, False)


def test_rolling_windows_exclude_current_game_and_respect_seasons():
    games = synthetic_games()
    tg = prepare_team_games(synthetic_team_games(games), games)
    roll = rolling_features(tg, core=["epa_play", "points"], extended=["success_rate"])
    team = tg.filter(pl.col("team") == "AAA").sort("kickoff_utc")
    feats = roll.filter(pl.col("team") == "AAA").sort("kickoff_utc")
    # Week-1 features must be null (no prior games) and games_played_std must be 0.
    first = feats.row(0, named=True)
    assert first["epa_play__l1"] is None and first["epa_play__std"] is None and first["games_played_std"] == 0
    # l1 equals previous game's value; std equals the mean of all previous games this season.
    vals = team["epa_play"].to_list()
    for i in range(1, len(vals)):
        row = feats.row(i, named=True)
        assert abs(row["epa_play__l1"] - vals[i - 1]) < 1e-9
        assert abs(row["epa_play__std"] - float(np.mean(vals[:i]))) < 1e-9
        assert row["games_played_std"] == i
    # Defensive 'allowed' metric is the opponent's offensive number from the previous game.
    prev = team.row(0, named=True)
    opp_val = tg.filter((pl.col("game_id") == prev["game_id"]) & (pl.col("team") == prev["opponent"]))["epa_play"][0]
    assert abs(feats.row(1, named=True)["epa_play_allowed__l1"] - opp_val) < 1e-9


def test_elo_pregame_rating_is_independent_of_that_games_result():
    games = synthetic_games()
    a = elo_features(games)
    flipped = games.with_columns([pl.col("away_score").alias("home_score"), pl.col("home_score").alias("away_score")]).with_columns((pl.col("home_score") - pl.col("away_score")).alias("margin"))
    b = elo_features(flipped)
    # First week ratings identical; later weeks differ because history changed, but the
    # pre-game rating of week-1 games can never depend on week-1 outcomes.
    first_ids = games.filter(pl.col("week") == 1)["game_id"].to_list()
    for gid in first_ids:
        ra = a.filter(pl.col("game_id") == gid).row(0, named=True)
        rb = b.filter(pl.col("game_id") == gid).row(0, named=True)
        assert ra["home_elo_pre"] == rb["home_elo_pre"] == 1500.0
    assert (a["elo_prob_home"] > 0).all() and (a["elo_prob_home"] < 1).all()


def test_miss_probabilities_are_ordered():
    assert miss_probability("out", None) == 1.0
    assert miss_probability("doubtful", "did not participate in practice") > miss_probability("questionable", "did not participate in practice") > miss_probability("questionable", "limited participation in practice") > miss_probability(None, "full participation in practice")


def test_availability_only_uses_reports_known_before_prediction_time():
    games = synthetic_games(n_weeks=2)
    ko = games.filter(pl.col("game_id").str.contains("_01_"))["kickoff_utc"][0]
    home = games.filter(pl.col("week") == 1)["home_team"][0]
    injuries = pl.DataFrame({"season": [2020, 2020], "game_type": ["REG", "REG"], "team": [home, home], "week": [1, 1], "gsis_id": ["p1", "p2"], "position": ["QB", "WR"], "full_name": ["A", "B"], "report_primary_injury": ["knee", "ankle"],
                             "report_status": ["out", "out"], "practice_status": ["did not participate in practice", "did not participate in practice"], "date_modified": [ko - timedelta(hours=40), ko + timedelta(hours=1)]}).with_columns(pl.col("date_modified").cast(pl.Datetime("us", "UTC")))
    snaps = pl.DataFrame({"game_id": [], "season": [], "week": [], "team": [], "position": [], "gsis_id": [], "offense_pct": [], "defense_pct": [], "st_pct": [], "game_type": []}, schema={"game_id": pl.String, "season": pl.Int32, "week": pl.Int32, "team": pl.String, "position": pl.String, "gsis_id": pl.String, "offense_pct": pl.Float64, "defense_pct": pl.Float64, "st_pct": pl.Float64, "game_type": pl.String})
    empty = pl.DataFrame()
    feats = availability_features(games, injuries, snaps, empty, empty, horizon_hours=1.0)
    row = feats.filter((pl.col("team") == home) & pl.col("game_id").str.contains("_01_")).row(0, named=True)
    # Only the QB report (known 40h before kickoff) counts; the WR report arrived after kickoff.
    assert row["inj_lost_QB"] == pytest.approx(0.25) and row["inj_lost_WR"] == 0.0 and row["injury_listings"] == 1


def test_market_no_vig_probability():
    games = synthetic_games(n_weeks=1)
    m = market_features(games).row(0, named=True)
    assert 0.5 < m["mkt_home_prob"] < 0.7
    assert m["mkt_home_implied"] + m["mkt_away_implied"] == pytest.approx(44.5)
    assert m["mkt_home_implied"] - m["mkt_away_implied"] == pytest.approx(3.0)


def test_metrics_and_bootstrap():
    p = np.array([0.9, 0.1, 0.6, 0.4])
    y = np.array([1, 0, 1, 0])
    assert log_loss(p, y) < log_loss(np.full(4, 0.5), y)
    assert brier(p, y) < 0.25 and expected_calibration_error(p, y) <= 1
    frame = pd.DataFrame({"actual_home_points": [24, 17], "actual_away_points": [20, 27], "pred_home_points": [23.0, 20.0], "pred_away_points": [21.0, 24.0], "pred_home_win_probability": [0.6, 0.4], "pred_margin_sd": [13.0, 13.0]})
    e = evaluate(frame)
    assert e["games"] == 2 and e["margin_mae"] == pytest.approx(4.0) and e["total_mae"] == pytest.approx(0.0) and "log_loss" in e and "crps_margin" in e
    boot = bootstrap_difference(np.array([1, 2, 3, 4.0]), np.array([2, 3, 4, 5.0]), np.array(["a", "a", "b", "b"]), n=200)
    assert boot["mean_difference"] == pytest.approx(-1.0) and boot["ci95"][0] <= -1.0 <= boot["ci95"][1]
    assert crps_samples(np.zeros((3, 100)), np.zeros(3)) == pytest.approx(0.0)


def test_ensemble_weights_sum_to_one_and_prefer_better_model():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 10, 500)
    frame = pd.DataFrame({"good": truth + rng.normal(0, 3, 500), "bad": truth + rng.normal(0, 12, 500)})
    w = fit_weights(frame, ["good", "bad"], truth)
    assert abs(sum(w["weights"].values()) - 1) < 1e-6 and w["weights"]["good"] > 0.7
    blended = apply_weights(frame, w)
    assert np.mean(np.abs(blended - truth)) <= w["single_losses"]["good"] + 1e-6


def test_calibration_and_simulation_are_consistent():
    p = win_probability_from_margin(np.array([-7, 0, 7.0]), np.array([13.0, 13.0, 13.0]), df=10, tie_rate=0.002)
    assert p[0] < p[1] < p[2] and abs(p[1] - 0.5) < 1e-9
    cal = ProbabilityCalibrator("platt").fit(np.linspace(0.05, 0.95, 200), (np.linspace(0.05, 0.95, 200) > 0.5).astype(float))
    restored = ProbabilityCalibrator.from_dict(cal.to_dict())
    assert np.allclose(cal.predict(np.array([0.3, 0.7])), restored.predict(np.array([0.3, 0.7])))
    rng = np.random.default_rng(1)
    preds = pd.DataFrame({"season": np.repeat(np.arange(2010, 2020), 40), "actual_home_points": rng.integers(10, 35, 400).astype(float), "actual_away_points": rng.integers(10, 35, 400).astype(float)})
    preds["m"] = (preds["actual_home_points"] - preds["actual_away_points"]) + rng.normal(0, 10, 400)
    preds["t"] = (preds["actual_home_points"] + preds["actual_away_points"]) + rng.normal(0, 9, 400)
    residual = fit_residual_model(preds, "m", "t")
    assert 5 < residual["margin_sd"] < 20 and len(residual["residual_pool"]["margin"]) > 100
    sim = simulate_game(3.0, 45.0, residual, n=20000, seed=1)
    assert abs(sim["home_win"] + sim["away_win"] + sim["tie"] - 1) < 1e-9
    assert abs(sim["margin_mean"] - 3.0) < 1.0 and abs(sim["total_mean"] - 45.0) < 1.0
    covers = {c["home_line"]: c for c in sim["spread_cover"]}
    assert covers[-7.0]["home_cover"] < covers[-3.0]["home_cover"] < covers[3.0]["home_cover"]
    assert abs(covers[-3.0]["home_cover"] + covers[-3.0]["push"] + covers[-3.0]["away_cover"] - 1) < 1e-9 and covers[-3.0]["push"] > 0
    assert sim["home_interval_80"][0] < sim["home_mean"] < sim["home_interval_80"][1]
    assert sum(sim["margin_distribution"].values()) == pytest.approx(1.0, abs=0.01)
    assert (margin_sd(residual, np.array([30.0, 60.0])) >= 9).all()


def test_registry_seeds_from_bundled_models(tmp_path):
    bundled = tmp_path / "bundle"
    source = Registry(bundled)
    source.register("seed-1", "game_forecast", {"weights": 2}, {"holdout": {"margin_mae": 9.8}}, {"c": 1})
    source.promote("game_forecast", "seed-1", "test")
    target = Registry(tmp_path / "fresh")
    assert target.champion("game_forecast") is None
    target.seed_from(source.models)
    assert target.champion("game_forecast")["model_id"] == "seed-1" and target.load("seed-1") == {"weights": 2}
    assert target.read()["history"][-1]["seeded_from_bundle"]


def test_registry_promotion_rule(tmp_path):
    reg = Registry(tmp_path)
    assert reg.champion("game_forecast") is None
    rec = reg.register("m1", "game_forecast", {"weights": 1}, {"holdout": {"margin_mae": 10.4, "log_loss": 0.63}}, {"c": 1})
    ok, reason = Registry.challenger_beats_champion({"holdout": rec["metrics"]["holdout"]}, None)
    assert ok
    reg.promote("game_forecast", "m1", reason)
    assert reg.champion("game_forecast")["model_id"] == "m1" and reg.load("m1") == {"weights": 1}
    weak = {"holdout": {"margin_mae": 10.38, "log_loss": 0.635}, "bootstrap_vs_champion": {"ci95": [-0.1, 0.05]}}
    assert not Registry.challenger_beats_champion(weak, reg.champion("game_forecast"))[0]
    strong = {"holdout": {"margin_mae": 10.2, "log_loss": 0.628}, "bootstrap_vs_champion": {"ci95": [-0.35, -0.05]}}
    assert Registry.challenger_beats_champion(strong, reg.champion("game_forecast"))[0]
