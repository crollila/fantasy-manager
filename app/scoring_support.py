"""Normalize ESPN yardage units and isolate scoring support by player position.

Stat ID definitions verified against ESPN scoring support and espn-api constants:
https://github.com/cwendt94/espn-api/blob/master/espn_api/football/constant.py
Unsupported nonlinear/bonus components retain the exact provider fallback.
"""
UNITS={5:('passing_yards',5),6:('passing_yards',10),7:('passing_yards',20),8:('passing_yards',25),9:('passing_yards',50),10:('passing_yards',100),11:('completions',5),12:('completions',10),13:('incompletions',5),14:('incompletions',10),27:('rushing_yards',5),28:('rushing_yards',10),29:('rushing_yards',20),30:('rushing_yards',25),31:('rushing_yards',50),32:('rushing_yards',100),33:('carries',5),34:('carries',10),47:('receiving_yards',5),48:('receiving_yards',10),49:('receiving_yards',20),50:('receiving_yards',25),51:('receiving_yards',50),52:('receiving_yards',100),54:('receptions',5),55:('receptions',10)}
EXTRA={2:'incompletions',19:'passing_2pt_conversions',26:'rushing_2pt_conversions',44:'receiving_2pt_conversions',59:'receiving_yards_after_catch',63:'fumble_recovery_tds',64:'sacks_suffered',69:'sack_fumbles_lost',70:'rushing_fumbles_lost',71:'receiving_fumbles_lost',83:'fg_made',84:'fg_att',87:'pat_att',201:'fg_made_60_',203:'fg_missed_60_',105:'def_and_special_tds',97:'blocked_kicks'}


def normalized_item(stat_id,points,stat_key):
    stat_id=int(stat_id)
    if stat_id in UNITS:
        key,unit=UNITS[stat_id];return {key:points/unit}
    if stat_id==74:return {'fg_made_50_59':points,'fg_made_60_':points}
    if stat_id==80:return {'fg_made_0_19':points,'fg_made_20_29':points,'fg_made_30_39':points}
    return {EXTRA.get(stat_id,stat_key(stat_id)):points}


def player_scoring(league,player,meta,provider):
    from app.league_sync import stat_key
    items=meta.get('scoring_items')
    if not items:return league,meta.get('independent_scoring_incomplete',False)
    position_id={'QB':'1','RB':'2','WR':'3','TE':'4','K':'5','DST':'16'}[player.position]
    scoring={};incomplete=False
    for item in items:
        sid=int(item['statId']);points=float(item.get('pointsOverrides',{}).get(position_id,item.get('points',0)))
        if not points:continue
        # Defense and kicking-only categories do not disable a receiver's independent projection.
        if player.position not in ('K','DST') and sid>=74:continue
        if player.position=='K' and not 74<=sid<=88 and sid not in (201,202,203):continue
        converted=normalized_item(sid,points,stat_key)
        for key,value in converted.items():
            scoring[key]=scoring.get(key,0)+value
            if key.startswith('espn_stat_'):
                incomplete=True
            elif key not in player.stats and value:
                # A positively projected unsupported component needs the provider's exact total.
                raw=provider.get('projected_stats',{}).get(stat_key(sid),0)
                if raw or player.position in ('K','DST'):incomplete=True
    return league.model_copy(update={'scoring':scoring}),incomplete
