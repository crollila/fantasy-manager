"""Strict normalization boundary for browser-observed state, never cookie ingestion."""
from app.domain import Pick, League
from app.identity import Identity


def normalize_snapshot(payload, league, players):
    identity = Identity(players)
    picks = []
    for row in payload.get("picks",[]):
        if "espn_id" not in row:
            raise ValueError("ESPN snapshot requires an explicit ESPN player ID")
        picks.append(Pick(number=int(row["number"]),team=int(row["team"]),player_id=identity.resolve("espn",str(row["espn_id"]))))
    from app.domain import validate_picks
    validate_picks(league,picks,players)
    return picks
