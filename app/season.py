from __future__ import annotations
import numpy as np
from app.scoring import lineup
from app.simulation import season_outcomes


def start_sit(roster, reports, league, week, risk="balanced", overrides=None):
    if risk not in ("floor","balanced","upside"):
        raise ValueError("Risk must be floor, balanced or upside")
    rp = {r["id"]:r for r in reports}
    field = {"floor":"p25","balanced":"mean","upside":"p75"}[risk]
    values = []
    for p in roster:
        r = rp[p.id]
        value = (overrides or {}).get(p.id, r[field]/max(1,p.games))
        values.append(0 if p.bye == week else value)
    chosen,total = lineup(roster,values,league)
    return {"lineup":chosen,"objective":total,"risk":risk,"week":week,"caveat":"Uses season per-game estimates unless weekly overrides supplied. Confirm active status and locks in ESPN."}


def waivers(players,reports,league,picks,week,faab_remaining):
    rp = {r["id"]:r for r in reports}
    owned = {p.player_id for p in picks}
    ids = {p.player_id for p in picks if p.team == league.my_team}
    roster = [p for p in players if p.id in ids]
    before = start_sit(roster,reports,league,week)["objective"]
    results = []
    for p in players:
        if p.id in owned:
            continue
        after = start_sit(roster+[p],reports,league,week)["objective"]
        r = rp[p.id]
        immediate = after-before
        remaining = max(0,18-week-(1 if p.bye and p.bye >= week else 0))
        ros = r["vorp"]*remaining/17
        utility = immediate + max(0,ros)*.12 + r["breakout"]*2
        bid = min(faab_remaining, round(faab_remaining * min(.5, max(0,utility)/100))) if league.waiver=="faab" else None
        results.append({"id":p.id,"name":p.name,"position":p.position,"ros_vorp":ros,"immediate_lineup_gain":immediate,"breakout":r["breakout"],"suggested_faab":bid,"value":utility})
    return {"players":sorted(results,key=lambda r:r["value"],reverse=True)[:30],"caveat":"FAAB is a budget heuristic, not an auction-equilibrium optimum. Import current ESPN rosters/free agents before acting; draft ownership alone becomes stale."}


def trade(players,reports,league,picks,give,receive,week=1):
    by_id = {p.id:p for p in players}
    owned = {p.player_id:p.team for p in picks}
    if not give or not receive or set(give)&set(receive) or len(set(give))!=len(give) or len(set(receive))!=len(receive):
        raise ValueError("Trade needs distinct nonempty give and receive lists")
    if any(owned.get(p)!=league.my_team for p in give):
        raise ValueError("You do not own every offered player")
    owners = {owned.get(p) for p in receive}
    if len(owners)!=1 or None in owners or league.my_team in owners:
        raise ValueError("Received players must belong to one other team")
    other = next(iter(owners))
    team_ids = [[p.player_id for p in picks if p.team==t] for t in range(league.teams)]
    if len(give)!=len(receive):
        raise ValueError("Unequal trades require explicit add/drop roster moves first")
    mine = [by_id[p] for p in team_ids[league.my_team]]
    after = [p for p in mine if p.id not in give]+[by_id[p] for p in receive]
    rp = {r["id"]:r for r in reports}
    before_l,start = lineup(mine,[rp[p.id]["mean"] for p in mine],league)
    after_l,end = lineup(after,[rp[p.id]["mean"] for p in after],league)
    result = {"lineup_delta":end-start,"before_lineup":before_l,"after_lineup":after_l,"championship_before":None,"championship_after":None,"caveat":"Full league rosters required for championship comparison. Weekly schedule and injury certainty are model approximations."}
    if all(len(ids)==league.roster_size for ids in team_ids) and week==1:
        idx = {p.id:i for i,p in enumerate(players)}
        before_array = np.tile([[idx[p] for p in ids] for ids in team_ids],(512,1,1))
        for team, outgoing, incoming in ((league.my_team,give,receive),(other,receive,give)):
            team_ids[team] = [p for p in team_ids[team] if p not in outgoing]+incoming
        after_array = np.tile([[idx[p] for p in ids] for ids in team_ids],(512,1,1))
        b,_ = season_outcomes(before_array,players,reports,league)
        a,_ = season_outcomes(after_array,players,reports,league)
        result.update(championship_before=b,championship_after=a)
    elif week>1:
        result["caveat"] = "Midseason championship odds require current standings and remaining schedule; unavailable. Lineup delta is a season-total equivalent, not remaining-season points."
    return result
