from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr
from app.domain import League, Player, Pick
from app.scoring import lineup, feasible_add


def survival(player, current_pick, next_pick, picks, players):
    if next_pick is None:
        return 0.
    if player.adp is None:
        return None
    by_id = {p.id: p for p in players}
    observations = [(p, by_id[p.player_id]) for p in picks[-24:] if p.player_id in by_id]
    reaches = [p.number - q.adp for p, q in observations if q.adp is not None]
    reach = float(np.mean(reaches)) * len(reaches) / (len(reaches) + 12) if reaches else 0
    recent = [q.position for _, q in observations[-6:]]
    historical_share = sum(p.position == player.position for p in players) / max(1, len(players))
    run = max(0, recent.count(player.position) - len(recent) * historical_share)
    mu = player.adp + reach - run * 2.5
    sd = max(4., player.adp_sd)
    # Conditional survival given still available now. next_pick itself is not an opponent pick.
    denominator = ndtr((mu - (current_pick - .5)) / sd)
    numerator = ndtr((mu - (next_pick - .5)) / sd)
    return float(np.clip(numerator / max(1e-12, denominator), 0, 1))


def quick_recommendations(players, reports, league, picks):
    drafted = {p.player_id for p in picks}
    available = [p for p in players if p.id not in drafted]
    roster = [p for p in players if any(k.player_id == p.id and k.team == league.my_team for k in picks)]
    by_report = {r["id"]: r for r in reports}
    values = [by_report[p.id]["mean"] for p in roster]
    _, before = lineup(roster, values, league)
    current = len(picks) + 1
    target = league.next_pick(current - 1)
    if target is None:
        return []
    following = league.next_pick(target)
    survive_by_id={p.id:survival(p,current-1,following,picks,players) for p in available}
    fallback_by_pos={}
    for q in available:
        fallback_by_pos.setdefault(q.position,[]).append((by_report[q.id]["mean"]*(survive_by_id[q.id] or 0),q.id))
    for pos in fallback_by_pos:fallback_by_pos[pos].sort(reverse=True)
    rows = []
    for p in available:
        if not feasible_add(roster, p, league):
            continue
        r = by_report[p.id]
        _, after = lineup(roster + [p], values + [r["mean"]], league)
        marginal = after - before
        survive = survive_by_id[p.id]
        fallback = [v for v,pid in fallback_by_pos[p.position][:2] if pid!=p.id]
        vonp = marginal - (fallback[0] if fallback else 0)
        depth = max(0, r["mean"] - r["bench_replacement"]) * .12
        urgency = .5 if survive is None else 1-survive
        utility = marginal + depth + .22 * vonp * urgency + .04 * (r["p90"]-r["median"])
        rows.append({**r, "survival": survive, "gone": None if survive is None else 1-survive, "vonp": vonp, "marginal_lineup": marginal, "quick_value": utility, "championship_probability": None, "championship_delta": None, "confidence": "low • heuristic; simulation pending"})
    return sorted(rows, key=lambda r: r["quick_value"], reverse=True)
