"""Synthetic fixtures; deliberately fictional player names, never current advice."""
import numpy as np
from app.domain import Player, League


def demo_players(n=360):
    rng = np.random.default_rng(2026)
    players = []
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
    positions = ["RB", "WR", "WR", "RB", "TE", "QB", "WR", "RB", "QB", "TE", "K", "DST"]
    for i in range(n):
        pos = positions[i % len(positions)]
        counts[pos] += 1
        rank = counts[pos]
        scale = max(.06, np.exp(-rank / (30 if pos in ("WR", "RB") else 18)))
        g = float(rng.uniform(12, 17))
        stats = {"passing_yards": 4600*scale, "passing_tds": 36*scale, "passing_interceptions": 12*scale, "rushing_yards": 400*scale, "rushing_tds": 4*scale} if pos == "QB" else {"rushing_yards": (1550 if pos == "RB" else 60)*scale, "rushing_tds": (13 if pos == "RB" else .5)*scale, "receiving_yards": (500 if pos == "RB" else 1500 if pos == "WR" else 1100)*scale, "receiving_tds": 10*scale, "receptions": (48 if pos == "RB" else 100)*scale}
        if pos in ("K", "DST"):
            stats = {"special_points": 145*scale}
        players.append(Player(id=f"demo-{i+1}", name=f"Demo {pos} {rank:02}", position=pos, team=f"T{i%32+1:02}", ids={"espn": str(900000+i)}, stats=stats, games=g, adp=1, adp_sd=12+rank*.7, workload_cv=float(rng.uniform(.18,.42)), source="SYNTHETIC DEMO", bye=i%10+5))
    from app.scoring import score
    league = demo_league()
    ordered = sorted(players, key=lambda p: score(p.stats, league) * (.72 if p.position == "QB" else .4 if p.position in ("K", "DST") else 1), reverse=True)
    for i, p in enumerate(ordered, 1):
        p.adp = float(i)
        p.espn_rank = float(i)
        p.ecr = float(i)
    return players


def demo_league():
    l = League(id="demo", name="Demo • 12-team PPR", source="SYNTHETIC DEMO", my_team=7)
    l.scoring["special_points"] = 1
    return l
