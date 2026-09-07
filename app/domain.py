from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
ELIGIBLE = {p: {p} for p in POSITIONS} | {"FLEX": {"RB", "WR", "TE"}, "SUPERFLEX": {"QB", "RB", "WR", "TE"}, "RB/WR": {"RB", "WR"}, "WR/TE": {"WR", "TE"}}
DEFAULT_SCORING = {"passing_yards": .04, "passing_tds": 4., "passing_interceptions": -2., "rushing_yards": .1, "rushing_tds": 6., "receiving_yards": .1, "receiving_tds": 6., "receptions": 1., "fumbles_lost": -2., "two_point_conversions": 2.}
DEFAULT_SCORING.update({"fg_made_0_19":3.,"fg_made_20_29":3.,"fg_made_30_39":3.,"fg_made_40_49":4.,"fg_made_50_59":5.,"fg_made_60_":5.,"pat_made":1.,"fg_missed":-1.,"def_sacks":1.,"def_interceptions":2.,"fumble_recovery_opp":2.,"def_tds":6.,"def_safeties":2.,"def_punt_blocks":2.,"def_pat_blocks":2.,"def_fg_blocks":2.,"special_teams_tds":6.})


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Bonus(Strict):
    stat: str
    threshold: float = Field(gt=0)
    points: float


class League(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=120)
    season: int = Field(default=2026, ge=2005, le=2100)
    teams: int = Field(default=12, ge=2, le=20)
    my_team: int = Field(default=0, ge=0)
    team_names: list[str] = Field(default_factory=list)
    draft_type: Literal["snake", "linear", "auction"] = "snake"
    draft_order: list[int] = Field(default_factory=list)
    slots: dict[str, int] = Field(default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1})
    bench: int = Field(default=6, ge=0, le=20)
    ir: int = Field(default=1, ge=0, le=10)
    scoring: dict[str, float] = Field(default_factory=lambda: DEFAULT_SCORING.copy())
    bonuses: list[Bonus] = Field(default_factory=list)
    playoff_teams: int = Field(default=6, ge=2, le=16)
    regular_weeks: int = Field(default=14, ge=1, le=16)
    playoff_weeks_per_round: int = Field(default=1, ge=1, le=2)
    waiver: Literal["faab", "priority", "rolling"] = "faab"
    faab_budget: int = Field(default=100, ge=0)
    mode: Literal["DRAFT", "SEASON"] = "DRAFT"
    settings_verified: bool = False
    source: str = "manual"

    @model_validator(mode="after")
    def validate_rules(self):
        if self.my_team >= self.teams or self.playoff_teams > self.teams:
            raise ValueError("Team index or playoff count exceeds league size")
        if not self.slots or any(s not in ELIGIBLE or n < 0 or n > 10 for s, n in self.slots.items()):
            raise ValueError("Unsupported lineup slot or slot count")
        if not 1 <= sum(self.slots.values()) <= 20:
            raise ValueError("Require between 1 and 20 starters")
        if not self.draft_order:
            self.draft_order = list(range(self.teams))
        if sorted(self.draft_order) != list(range(self.teams)):
            raise ValueError("Draft order must contain each team exactly once")
        if self.team_names and len(self.team_names) != self.teams:
            raise ValueError("Provide one name per team")
        import math
        rounds = math.ceil(math.log2(self.playoff_teams))
        if self.regular_weeks + rounds * self.playoff_weeks_per_round > 18:
            raise ValueError("Playoffs extend beyond week 18")
        return self

    @property
    def roster_size(self):
        return sum(self.slots.values()) + self.bench

    def owner(self, pick: int) -> int:
        if pick < 1:
            raise ValueError("Picks are one-based")
        r, i = divmod(pick - 1, self.teams)
        if self.draft_type == "auction":
            raise ValueError("Auction optimization is not implemented; use manual season tools")
        return self.draft_order[self.teams - 1 - i if self.draft_type == "snake" and r % 2 else i]

    def next_pick(self, after: int, team: int | None = None) -> int | None:
        target = self.my_team if team is None else team
        return next((p for p in range(after + 1, self.teams * self.roster_size + 1) if self.owner(p) == target), None)


class Player(Strict):
    id: str
    name: str
    position: Literal["QB", "RB", "WR", "TE", "K", "DST"]
    team: str = "FA"
    ids: dict[str, str] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)
    games: float = Field(default=14.5, ge=0, le=17)
    workload_cv: float = Field(default=.25, ge=.01, le=2)
    efficiency_cv: float = Field(default=.15, ge=.01, le=2)
    adp: float | None = Field(default=None, gt=0)
    adp_sd: float = Field(default=18, gt=0)
    espn_rank: float | None = Field(default=None, gt=0)
    ecr: float | None = Field(default=None, gt=0)
    market_stats: dict[str, float] | None = None
    market_weight: float = Field(default=0, ge=0, le=.8)
    bye: int | None = Field(default=None, ge=1, le=18)
    source: str = "independent"
    warnings: list[str] = Field(default_factory=list)


class Pick(Strict):
    number: int = Field(ge=1)
    team: int = Field(ge=0)
    player_id: str


class Snapshot(Strict):
    picks: list[Pick]
    source: str = "manual"
    timer: str | None = None


class Event(Strict):
    id: str
    player_id: str
    kind: Literal["traded", "released", "signed", "injured", "returned", "suspended", "promoted", "demoted", "teammate_injured", "qb_changed", "coach_changed", "coordinator_changed", "camp_report", "usage"]
    occurred_at: str
    known_at: str
    source_url: str
    note: str = Field(max_length=2000)
    workload_multiplier: float = Field(default=1, ge=0, le=2)
    efficiency_multiplier: float = Field(default=1, ge=.5, le=1.5)
    games_delta: float = Field(default=0, ge=-17, le=17)
    team: str | None = None
    confirmed: bool = False


def validate_picks(league: League, picks: list[Pick], players: list[Player]):
    known = {p.id for p in players}
    if len(picks) > league.roster_size * league.teams:
        raise ValueError("Too many picks")
    if len({p.player_id for p in picks}) != len(picks):
        raise ValueError("Player drafted more than once")
    for i, p in enumerate(sorted(picks, key=lambda p: p.number), 1):
        if p.number != i or p.team != league.owner(i) or p.player_id not in known:
            raise ValueError(f"Pick {p.number}: gap, wrong owner, or unknown player ID")
