from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.domain import League, Player, ELIGIBLE


def score(stats: dict, league: League):
    return sum(stats.get(k, 0) * v for k, v in league.scoring.items())


def score_week(stats: dict, league: League):
    return score(stats, league) + sum(b.points for b in league.bonuses if stats.get(b.stat, 0) >= b.threshold)


def slots(league):
    return [s for s, n in league.slots.items() for _ in range(n)]


def lineup(players: list[Player], values, league: League):
    names = slots(league)
    if not names:
        return [], 0.
    # Dummy replacement slots represent missing starters; no player can fill two slots.
    matrix = np.full((len(names), len(players) + len(names)), -1e9)
    matrix[:, len(players):] = 0
    for i, slot in enumerate(names):
        for j, p in enumerate(players):
            if p.position in ELIGIBLE[slot]:
                matrix[i, j] = values[j]
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    chosen = [{"slot": names[i], "player_id": players[j].id if j < len(players) else None, "value": float(matrix[i, j])} for i, j in zip(rows, cols)]
    return chosen, float(matrix[rows, cols].sum())


def feasible_add(roster, candidate, league):
    if len(roster) >= league.roster_size or candidate.id in {p.id for p in roster}:
        return False
    selected, _ = lineup(roster + [candidate], [1.] * (len(roster) + 1), league)
    missing = sum(x["player_id"] is None for x in selected)
    return missing <= league.roster_size - len(roster) - 1


def replacement_levels(players, means, league):
    # Solve all teams' starting slots simultaneously, respecting FLEX competition.
    big = league.model_copy(deep=True)
    big.slots = {s: n * league.teams for s, n in league.slots.items()}
    chosen, _ = lineup(players, means, big)
    starters = {x["player_id"] for x in chosen}
    result = {}
    for pos in ELIGIBLE:
        ordered = sorted([(float(v), p.id) for p, v in zip(players, means) if p.position in ELIGIBLE[pos]], reverse=True)
        remaining = [v for v, pid in ordered if pid not in starters]
        result[pos] = {"starter": remaining[0] if remaining else 0., "bench": remaining[min(len(remaining) - 1, max(0, league.teams * league.bench // 4))] if remaining else 0.}
    return result
