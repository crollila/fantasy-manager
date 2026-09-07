"""Batched stochastic opponent drafts and weekly season tournaments.

Win probabilities are conditional on this model, not empirically calibrated odds.
Common random numbers reduce comparison noise between candidate actions.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
import numpy as np
from app.domain import POSITIONS, ELIGIBLE, League, Pick
from app.scoring import slots


def hall_constraints(league):
    masks = np.array([[bool(bits & (1 << p)) for p in range(6)] for bits in range(1, 64)], dtype=np.int16)
    slot_masks = [sum(1 << POSITIONS.index(p) for p in ELIGIBLE[s]) for s in slots(league)]
    demand = np.array([sum(mask & bits == mask for mask in slot_masks) for bits in range(1, 64)])
    return masks, demand


def valid_positions(counts, remaining_after, league, constraints=None):
    masks, demand = constraints or hall_constraints(league)
    deficits = demand - counts @ masks.T
    return np.stack([np.maximum(0, (deficits - masks[:, p]).max(axis=-1)) <= remaining_after for p in range(6)], axis=-1)


def simulate_drafts(players, reports, league, picks, n=1000, seed=42, candidate=None, policy="adaptive"):
    if league.draft_type == "auction":
        raise ValueError("Auction simulations are not supported")
    if len(players) < league.teams * league.roster_size:
        raise ValueError("Player pool is too small to complete the draft; import missing K/DST/rookies")
    n = int(n)
    rng = np.random.default_rng(seed)
    index = {p.id: i for i, p in enumerate(players)}
    rp = {r["id"]: r for r in reports}
    means = np.array([rp[p.id]["mean"] for p in players])
    vorp = np.array([rp[p.id]["vorp"] for p in players])
    pos = np.array([POSITIONS.index(p.position) for p in players])
    ranks = np.argsort(np.argsort(-means)) + 1
    adp = np.array([p.adp or float(ranks[i]) for i, p in enumerate(players)])
    espn = np.array([p.espn_rank or adp[i] for i, p in enumerate(players)])
    ecr = np.array([p.ecr or adp[i] for i, p in enumerate(players)])
    available = np.ones((n, len(players)), dtype=bool)
    rosters = np.full((n, league.teams, league.roster_size), -1, dtype=np.int32)
    counts = np.zeros((n, league.teams, 6), dtype=np.int16)
    sizes = np.zeros(league.teams, dtype=int)
    rows = np.arange(n)
    for pick in picks:
        idx = index[pick.player_id]
        available[:, idx] = False
        rosters[:, pick.team, sizes[pick.team]] = idx
        counts[:, pick.team, pos[idx]] += 1
        sizes[pick.team] += 1
    constraints = hall_constraints(league)
    fixed = np.array([league.slots.get(p, 0) for p in POSITIONS])
    aggression = np.zeros((league.teams, 6))
    observed = np.zeros((league.teams, 6))
    for k in picks:
        i = index[k.player_id]
        aggression[k.team, pos[i]] += (adp[i] - k.number) / max(10, adp[i])
        observed[k.team, pos[i]] += 1
    aggression /= observed + 4
    # Each simulation samples a stable manager style; update affinity from observed rank errors.
    styles = rng.uniform(.2, .8, (n, league.teams, 1))
    preference = rng.normal(0, .13, (n, league.teams, 6)) + aggression[None, :, :]
    paths = []
    target = league.next_pick(len(picks))
    for number in range(len(picks)+1, league.teams*league.roster_size+1):
        team = league.owner(number)
        left = league.roster_size - sizes[team] - 1
        valid = valid_positions(counts[:, team], left, league, constraints)
        allowed = available & valid[:, pos]
        needs = np.maximum(0, fixed - counts[:, team])
        # Diminishing bench value; flex eligibility is enforced by Hall constraints, not rigid strategies.
        position_boost = .75 * needs[:, pos] / np.maximum(1, fixed[pos]) - .16 * np.maximum(0, counts[:, team, pos] - fixed[pos])
        rank = styles[:, team] * adp + (1-styles[:, team]) * espn
        logits = -np.log(rank + 5) + position_boost + preference[:, team, pos]
        if team == league.my_team:
            if policy in ("adp", "espn", "ecr"):
                source = {"adp": adp, "espn": espn, "ecr": ecr}[policy]
                logits = np.broadcast_to(-source, available.shape).copy()
            elif policy == "bpa":
                logits = np.broadcast_to(means, available.shape).copy()
            elif policy == "vorp":
                logits = np.broadcast_to(vorp, available.shape).copy()
            else:
                logits = logits + means / max(1, means.max()) * .5
        noise = rng.gumbel(0, .32, available.shape)
        scores = np.where(allowed, logits + (noise if team != league.my_team or policy == "adaptive" else 0), -np.inf)
        if not np.all(np.any(allowed, axis=1)):
            raise ValueError("Insufficient eligible players to finish a legal roster")
        chosen = np.argmax(scores, axis=1)
        if candidate is not None and number == target:
            forced = index[candidate]
            # Before our turn, opponents may take the candidate. Do not resurrect drafted players.
            if number != len(picks)+1:
                raise ValueError("Candidate simulation requires our current pick")
            if not np.all(allowed[:, forced]):
                raise ValueError("Candidate violates roster constraints or was drafted")
            chosen[:] = forced
        rosters[rows, team, sizes[team]] = chosen
        available[rows, chosen] = False
        counts[rows, team, pos[chosen]] += 1
        sizes[team] += 1
        if team == league.my_team:
            paths.append((number, chosen.copy()))
    return rosters, paths


def weekly_lineup_scores(rosters, players, reports, league, rng, week, season_factors=None, health=None):
    rp = {r["id"]: r for r in reports}
    per_game = np.array([rp[p.id]["mean"] / max(1, p.games) for p in players])
    cvs = np.array([math.sqrt(p.workload_cv**2 + p.efficiency_cv**2) for p in players])
    n = rosters.shape[0]
    # Shared NFL-team shock induces positive teammate correlation; the approximation is documented.
    teams = {team: i for i, team in enumerate(sorted({p.team for p in players}))}
    team_index = np.array([teams[p.team] for p in players])
    shared = rng.normal(0, .10, (n, len(teams)))[:, team_index]
    seasonal = season_factors if season_factors is not None else rng.lognormal(-.5*cvs**2, cvs, (n, len(players)))
    weekly = rng.lognormal(-.5*.55**2, .55, (n, len(players)))
    active = health[:, :, week-1].copy() if health is not None else rng.random((n, len(players))) < np.array([p.games/17 for p in players])
    active[:, [i for i,p in enumerate(players) if p.bye == week]] = False
    actual = per_game * seasonal * weekly * np.exp(shared-.005) * active
    expected = np.broadcast_to(per_game, (n, len(players))).copy() * active
    picked = np.zeros_like(rosters, dtype=bool)
    points = np.zeros(rosters.shape[:2])
    positions = np.array([POSITIONS.index(p.position) for p in players])
    # For standard nested slots this is exact: fill fixed positions, FLEX, then SUPERFLEX.
    # For overlapping RB/WR + WR/TE, use exact assignment to avoid greedy overlap.
    ordered_slots = sorted(slots(league), key=lambda s: len(ELIGIBLE[s]))
    overlapping = "RB/WR" in league.slots and "WR/TE" in league.slots
    if overlapping:
        from scipy.optimize import linear_sum_assignment
        for trial in range(n):
            for team in range(league.teams):
                ids = rosters[trial,team]
                mat = np.array([[expected[trial,i] if players[i].position in ELIGIBLE[s] else -1e9 for i in ids] for s in ordered_slots])
                a,b = linear_sum_assignment(mat, maximize=True)
                points[trial,team] = actual[trial,ids[b]].sum()
        return points
    row = np.arange(n)[:,None]
    team = np.arange(league.teams)[None,:]
    for slot in ordered_slots:
        eligible = np.isin(positions[rosters], [POSITIONS.index(p) for p in ELIGIBLE[slot]]) & ~picked
        val = np.where(eligible, expected[np.arange(n)[:,None,None], rosters], -np.inf)
        chosen = val.argmax(axis=2)
        ids = rosters[row, team, chosen]
        points += np.where(eligible.any(axis=2), actual[row, ids], 0)
        picked[row, team, chosen] = True
    return points


def season_outcomes(rosters, players, reports, league, seed=101):
    rng = np.random.default_rng(seed)
    n = rosters.shape[0]
    rp={r["id"]:r for r in reports}
    cvs = np.array([max(math.sqrt(p.workload_cv**2+p.efficiency_cv**2), min(1.5,rp[p.id]["sd"]/max(1,rp[p.id]["mean"]))) for p in players])
    season_factors = rng.lognormal(-.5*cvs**2, cvs, (n,len(players)))
    # Persistent season quality and contiguous missed-game blocks preserve roster risk.
    missed = rng.binomial(17, np.array([1-p.games/17 for p in players]), (n,len(players)))
    injury_start = rng.integers(0,18,(n,len(players)))
    health = ((np.arange(18)[None,None,:] - injury_start[:,:,None]) % 18 >= missed[:,:,None])
    wins = np.zeros((n, league.teams))
    total = np.zeros_like(wins)
    # Circle round-robin schedule, supports an odd team bye.
    order = list(range(league.teams)) + ([-1] if league.teams % 2 else [])
    for week in range(1, league.regular_weeks+1):
        points = weekly_lineup_scores(rosters, players, reports, league, rng, week, season_factors, health)
        total += points
        for a,b in zip(order[:len(order)//2], reversed(order[len(order)//2:])):
            if a >= 0 and b >= 0:
                wins[:,a] += (points[:,a] > points[:,b]) + .5*(points[:,a] == points[:,b])
                wins[:,b] += (points[:,b] > points[:,a]) + .5*(points[:,a] == points[:,b])
        order = [order[0], order[-1]] + order[1:-1]
    seeds = np.argsort(-(wins * 100000 + total), axis=1)[:,:league.playoff_teams]
    qualified = np.mean(np.any(seeds == league.my_team, axis=1))
    alive = seeds.copy()
    week = league.regular_weeks + 1
    while alive.shape[1] > 1:
        k = alive.shape[1]
        bracket_size = 2 ** math.ceil(math.log2(k))
        byes = bracket_size - k
        points = sum(weekly_lineup_scores(rosters, players, reports, league, rng, week+w, season_factors, health) for w in range(league.playoff_weeks_per_round))
        winners = [alive[:,i] for i in range(byes)]
        for i in range((k-byes)//2):
            a, b = alive[:,byes+i], alive[:,k-1-i]
            rows = np.arange(n)
            winners.append(np.where(points[rows,a] >= points[rows,b], a, b))
        alive = np.stack(winners, axis=1)
        # Re-seed survivors by original seed, including first-round byes.
        ranks = np.argmax(alive[:,:,None] == seeds[:,None,:], axis=2)
        alive = np.take_along_axis(alive, np.argsort(ranks, axis=1), axis=1)
        week += league.playoff_weeks_per_round
    champions = alive[:,0]
    return {"championship_probability": float(np.mean(champions == league.my_team)), "playoff_probability": float(qualified), "expected_wins": float(wins[:,league.my_team].mean()), "lineup_points": float(total[:,league.my_team].mean())}, champions


def evaluate_candidates(players, reports, league, picks, candidates, n=1000, seed=42):
    if league.owner(len(picks)+1) != league.my_team:
        return {"status": "waiting", "reason": "Candidate championship comparisons run on your turn", "results": []}
    baseline_rosters, _ = simulate_drafts(players,reports,league,picks,n,seed,policy="adp")
    baseline, base_champions = season_outcomes(baseline_rosters,players,reports,league,seed+100)
    results = []
    for candidate in candidates:
        rosters, paths = simulate_drafts(players,reports,league,picks,n,seed,candidate=candidate,policy="adp")
        outcome, champions = season_outcomes(rosters,players,reports,league,seed+100)
        paired = (champions == league.my_team).astype(float) - (base_champions == league.my_team).astype(float)
        se = float(np.std(paired, ddof=1)/math.sqrt(n)) if n > 1 else 1
        outcome.update({"id":candidate, "championship_delta": float(np.mean(paired)), "delta_ci95": [float(np.mean(paired)-1.96*se), float(np.mean(paired)+1.96*se)], "simulations":n, "next_targets": [{"id": players[i].id, "probability": count/n} for i,count in Counter(paths[1][1]).most_common(3)] if len(paths)>1 else [], "confidence":"simulation estimate • uncalibrated"})
        results.append(outcome)
    return {"status":"complete", "baseline":baseline, "baseline_policy":"ADP drafting + legal roster constraints; same ADP continuation after each candidate", "results":sorted(results,key=lambda r:r["championship_probability"],reverse=True), "seed":seed, "caveat":"Conditional simulation estimates, not validated championship odds. Intervals cover Monte Carlo sampling error only; candidate selection adds winner's bias."}


def mock_report(players,reports,league,n=1000,seed=42):
    rosters, paths = simulate_drafts(players,reports,league,[],n,seed)
    outcome, _ = season_outcomes(rosters,players,reports,league,seed+100)
    triples = Counter(tuple(int(paths[j][1][i]) for j in range(min(3,len(paths)))) for i in range(n))
    sequences = Counter(tuple(players[int(paths[j][1][i])].position for j in range(min(3,len(paths)))) for i in range(n))
    return {"simulations":n, "outcome":outcome, "openings":[{"players":[players[i].id for i in seq],"frequency":c/n} for seq,c in triples.most_common(10)], "sequences":[{"positions":seq,"frequency":c/n} for seq,c in sequences.most_common(10)], "picks":[{"pick":number,"targets":[{"id":players[int(i)].id,"frequency":c/n} for i,c in Counter(chosen).most_common(5)]} for number,chosen in paths], "caveat":"Sampled adaptive paths, not proven optimal opening strategies."}
