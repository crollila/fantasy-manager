from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from app.domain import League, Player
from app.scoring import score, score_week, replacement_levels
from app.data import historical, STAT_COLUMNS
from app.storage import DATA

# Per-game priors deliberately conservative. Replaced by historical position medians when data exists.
PRIORS = {
    "QB": {"attempts": 29, "carries": 3, "targets": 0, "ypa": 6.8, "ypc": 4.3, "ypt": 0, "catch": 0, "pass_td": .043, "rush_td": .035, "rec_td": 0, "int": .025},
    "RB": {"attempts": 0, "carries": 8, "targets": 2, "ypa": 0, "ypc": 4.2, "ypt": 5.8, "catch": .75, "pass_td": 0, "rush_td": .028, "rec_td": .025, "int": 0},
    "WR": {"attempts": 0, "carries": .2, "targets": 4, "ypa": 0, "ypc": 5, "ypt": 7.8, "catch": .63, "pass_td": 0, "rush_td": .02, "rec_td": .045, "int": 0},
    "TE": {"attempts": 0, "carries": 0, "targets": 3, "ypa": 0, "ypc": 3, "ypt": 7.2, "catch": .69, "pass_td": 0, "rush_td": 0, "rec_td": .045, "int": 0},
}


def project_history(df, season):
    before = df[(df.season < season) & (df.season >= season - 3)].copy()
    players = []
    for pid, rows in before.groupby("player_id"):
        rows = rows.sort_values("season")
        last = rows.iloc[-1]
        pos = str(last.position)
        if pos not in PRIORS:
            continue
        # Exclude players absent for a full season; current roster import can restore them as uncertain priors.
        if int(last.season) != season - 1:
            continue
        prior = PRIORS[pos]
        weights = np.power(.55, season - 1 - rows.season.to_numpy())
        games = rows.games.clip(lower=1).to_numpy()
        exposure = float(np.dot(weights, games))
        totals = {s: float(np.dot(weights, rows[s].fillna(0))) for s in STAT_COLUMNS}
        expected_games = min(17., max(2., (float(np.average(games, weights=weights)) * 2 + 15.5) / 3))
        opp = {k: (totals[k] + prior[k] * 3) / (exposure + 3) for k in ("attempts", "carries", "targets")}
        def rate(numerator, denominator, base, shrink=75):
            return (totals[numerator] + base * shrink) / (totals[denominator] + shrink)
        a, c, t = (opp[k] * expected_games for k in ("attempts", "carries", "targets"))
        stats = {"attempts": a, "carries": c, "targets": t,
                 "passing_yards": a * rate("passing_yards", "attempts", prior["ypa"]),
                 "passing_tds": a * rate("passing_tds", "attempts", prior["pass_td"], 200),
                 "passing_interceptions": a * rate("passing_interceptions", "attempts", prior["int"], 150),
                 "rushing_yards": c * rate("rushing_yards", "carries", prior["ypc"]),
                 "rushing_tds": c * rate("rushing_tds", "carries", prior["rush_td"], 125),
                 "receptions": t * rate("receptions", "targets", prior["catch"]),
                 "receiving_yards": t * rate("receiving_yards", "targets", prior["ypt"]),
                 "receiving_tds": t * rate("receiving_tds", "targets", prior["rec_td"], 100),
                 "fumbles_lost": totals["fumbles_lost"] / max(1, exposure) * expected_games,
                 "two_point_conversions": totals["two_point_conversions"] / max(1, exposure) * expected_games}
        opp_per_game = (rows.targets + rows.carries + rows.attempts).to_numpy() / games
        volatility = float(np.std(opp_per_game) / max(1, np.mean(opp_per_game))) if len(rows) > 1 else .3
        players.append(Player(id=str(pid), name=str(last.player_display_name), position=pos, team=str(last.get("team", "FA")), ids={"gsis": str(pid)}, stats=stats, games=expected_games, workload_cv=max(.18, min(.65, .15 + volatility)), warnings=["No verified current role adjustment", "ADP unavailable"] ))
    return players


def build_catalog(season, cache_path=DATA / "cache"):
    df = historical(cache_path)
    if df.season.max() < season - 1:
        raise ValueError(f"Latest history is {df.season.max()}; refusing to label outdated history as a {season} model")
    players = project_history(df, season)
    by_id = {p.id: p for p in players}
    ids_path = Path(cache_path) / "players.parquet"
    if ids_path.exists():
        ids = pd.read_parquet(ids_path).fillna("")
        for row in ids.to_dict("records"):
            p = by_id.get(str(row.get("gsis_id", "")))
            if p and row.get("espn_id"):
                p.ids["espn"] = str(row["espn_id"]).removesuffix(".0")
    roster_path = Path(cache_path) / f"roster_{season}.parquet"
    if roster_path.exists():
        roster = pd.read_parquet(roster_path).fillna("")
        active_ids = set()
        for row in roster.to_dict("records"):
            pid = str(row.get("gsis_id", ""))
            pos = row.get("position", "")
            if not pid or pos not in PRIORS:
                continue
            active_ids.add(pid)
            if pid not in by_id:
                # Rookies and returning players have no NFL opportunity evidence. Never assign star volume by name.
                prior = PRIORS[pos]
                games = 12.
                t, c, a = [prior[k] * games * .5 for k in ("targets", "carries", "attempts")]
                p = Player(id=pid, name=str(row.get("full_name") or row.get("player_name") or pid), position=pos, team=str(row.get("team", "FA")), stats={"attempts": a, "targets": t, "carries": c, "passing_yards": a * prior["ypa"], "passing_tds": a * prior["pass_td"], "rushing_yards": c * prior["ypc"], "rushing_tds": c * prior["rush_td"], "receptions": t * prior["catch"], "receiving_yards": t * prior["ypt"], "receiving_tds": t * prior["rec_td"]}, games=games, workload_cv=.65, efficiency_cv=.25, ids={"gsis": pid}, warnings=["Prior-only: import rookie/role evidence before trusting", "ADP unavailable"])
                players.append(p)
                by_id[pid] = p
            by_id[pid].team = str(row.get("team", "FA"))
            if row.get("espn_id"):
                by_id[pid].ids["espn"] = str(row["espn_id"]).removesuffix(".0")
        if active_ids:
            players = [p for p in players if p.id in active_ids]
    else:
        for p in players:
            p.warnings.append(f"{season} roster unavailable; team/active status unverified")
    schedule = Path(cache_path) / "schedules.parquet"
    if schedule.exists():
        games = pd.read_parquet(schedule)
        games = games[(games.season == season) & (games.game_type == "REG")]
        for p in players:
            weeks = set(games.loc[(games.home_team == p.team) | (games.away_team == p.team), "week"])
            byes = set(range(1, 19)) - weeks
            if len(byes) == 1:
                p.bye = int(byes.pop())
    from app.special import special_players,reconcile_opportunity
    from app.learning import fit_selected
    players,_ = fit_selected(df,season,players,Path(cache_path).parent / f"model_{season}.json")
    players.extend(special_players(season,cache_path))
    return reconcile_opportunity(players,season,cache_path)


def apply_events(players, events, as_of):
    out = {p.id: p.model_copy(deep=True) for p in players}
    from datetime import datetime
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    for e in sorted(events, key=lambda e: e.known_at):
        known = datetime.fromisoformat(e.known_at.replace("Z", "+00:00"))
        if known.tzinfo is None or cutoff.tzinfo is None:
            raise ValueError("Event and cutoff timestamps require timezone")
        if not e.confirmed or known > cutoff or e.player_id not in out:
            continue
        p = out[e.player_id]
        old_games = max(.01, p.games)
        p.games = max(0, min(17, p.games + e.games_delta))
        for k in p.stats:
            eff = 1 if k in ("attempts", "carries", "targets", "receptions") else e.efficiency_multiplier
            p.stats[k] *= e.workload_multiplier * eff * p.games / old_games
        if p.market_stats:
            # News belongs in the independent model; imported market timestamp is separate.
            p.warnings.append("Market projection may predate confirmed news")
        if e.team:
            p.team = e.team
        p.warnings.append(f"Confirmed {e.kind}: {e.note}")
    return list(out.values())


def distribution(p: Player, league: League, n=2048, seed=42):
    stable = int.from_bytes(hashlib.sha256(p.id.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed + stable)
    workload = rng.lognormal(-.5 * math.log1p(p.workload_cv ** 2), math.sqrt(math.log1p(p.workload_cv ** 2)), n)
    efficiency = rng.lognormal(-.5 * math.log1p(p.efficiency_cv ** 2), math.sqrt(math.log1p(p.efficiency_cv ** 2)), n)
    games = rng.binomial(17, p.games / 17, n)
    independent = score(p.stats, league)
    market = score(p.market_stats, league) if p.market_stats is not None else None
    w = p.market_weight if market is not None else 0
    totals = np.zeros(n)
    # Bonuses apply per game, not against season totals.
    mixed = {k: (1-w) * p.stats.get(k, 0) + w * (p.market_stats or {}).get(k, 0) for k in set(p.stats) | set(p.market_stats or {})}
    for week in range(17):
        active = week < games
        weekly_noise = rng.lognormal(-.5 * .5 ** 2, .5, n)
        stats = {k: v / max(.01, p.games) * workload * (1 if k in ("attempts", "targets", "carries", "receptions") else efficiency) * weekly_noise for k, v in mixed.items()}
        points = sum(stats.get(k, 0) * v for k, v in league.scoring.items())
        for b in league.bonuses:
            points += (stats.get(b.stat, np.zeros(n)) >= b.threshold) * b.points
        totals += np.asarray(points) * active
    q = np.quantile(totals, [.1, .25, .5, .75, .9])
    center = float(np.mean(totals))
    return totals, {"independent": independent, "market": market, "market_weight": w, "mean": center, "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4]), "sd": float(np.std(totals)), "expected_games": p.games, "bust": float(np.mean(totals < center * .65)), "breakout": float(np.mean(totals > center * 1.35)), "workload_cv": p.workload_cv, "efficiency_cv": p.efficiency_cv}


def project_all(players, league, n=1024):
    samples, reports = [], []
    import json
    calibration_path=DATA / f"model_{league.season}.json"
    calibration=json.loads(calibration_path.read_text()).get("residuals",[]) if calibration_path.exists() else []
    for p in players:
        draws, report = distribution(p, league, n=n)
        if calibration and p.source.startswith("independent:") and p.games>0 and not league.bonuses:
            from app.learning import calibrate_samples
            draws,info=calibrate_samples(draws,p,league,calibration)
            q=np.quantile(draws,[.1,.25,.5,.75,.9])
            center=float(np.mean(draws))
            report.update(mean=center,p10=float(q[0]),p25=float(q[1]),median=float(q[2]),p75=float(q[3]),p90=float(q[4]),sd=float(np.std(draws)),bust=float(np.mean(draws<center*.65)),breakout=float(np.mean(draws>center*1.35)),calibration=info)
        samples.append(draws)
        reports.append({**p.model_dump(exclude={"market_stats"}), **report})
    if not players:
        return []
    matrix = np.asarray(samples)
    replacement = replacement_levels(players, [r["mean"] for r in reports], league)
    for pos in {p.position for p in players}:
        indices = [i for i, p in enumerate(players) if p.position == pos]
        ranks = np.argsort(np.argsort(-matrix[indices], axis=0), axis=0) + 1
        ordered = sorted(indices, key=lambda i: reports[i]["mean"], reverse=True)
        tier = 1
        for j, i in enumerate(ordered):
            if j and reports[ordered[j-1]]["mean"] - reports[i]["mean"] > max(12, .10 * reports[ordered[j-1]]["mean"]):
                tier += 1
            reports[i]["tier"] = tier
        for j, i in enumerate(indices):
            reports[i].update({f"top{k}": float(np.mean(ranks[j] <= k)) for k in (3, 5, 12)})
            reports[i]["vorp"] = reports[i]["mean"] - replacement[pos]["starter"]
            reports[i]["bench_replacement"] = replacement[pos]["bench"]
    return sorted(reports, key=lambda r: r["mean"], reverse=True)


def ensemble_weight(independent_errors, market_errors):
    a, b = np.asarray(independent_errors), np.asarray(market_errors)
    if len(a) < 30 or a.shape != b.shape:
        raise ValueError("At least 30 aligned, pre-season out-of-sample errors required")
    d = b - a
    return float(np.clip(-np.mean(a * d) / max(1e-9, np.mean(d*d)), 0, .8))
