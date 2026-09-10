"""Opponent-adjusted fantasy production and learned weather/play-calling effects."""
import math
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from app.game_model import kickoff,number,ridge_fit,predict,weather_features
from app.research_sources import read_frame
from app.domain import DEFAULT_SCORING


def completed_stats(cache_path, season, as_of):
    schedule=enrich_weather(read_frame(Path(cache_path)/'schedules.parquet'),cache_path,season)
    known={r['game_id']:r for r in schedule.to_dict('records') if kickoff(r) and kickoff(r)+timedelta(hours=8)<as_of and pd.notna(r.get('home_score')) and pd.notna(r.get('away_score')) and r.get('game_type')=='REG'}
    frames=[]
    for year in range(season-3,season+1):
        frame=read_frame(Path(cache_path)/f'stats_{year}.parquet')
        if not frame.empty and 'game_id' in frame:frames.append(frame[frame.game_id.isin(known)])
    return (pd.concat(frames,ignore_index=True).drop_duplicates(['game_id','player_id']) if frames else pd.DataFrame()),known


def fantasy_points(frame, scoring):
    result=pd.Series(0.,index=frame.index)
    for stat,weight in scoring.items():
        if not weight:continue
        if stat=='fumbles_lost':values=sum((frame.get(k,pd.Series(0.,index=frame.index)).fillna(0) for k in ('rushing_fumbles_lost','receiving_fumbles_lost','sack_fumbles_lost')))
        elif stat=='two_point_conversions':values=sum((frame.get(k,pd.Series(0.,index=frame.index)).fillna(0) for k in ('passing_2pt_conversions','rushing_2pt_conversions','receiving_2pt_conversions')))
        else:values=frame.get(stat,pd.Series(0.,index=frame.index)).fillna(0)
        result+=values*weight
    return result


def fit_matchups(stats, schedule, as_of, scoring=None,position_scoring=None):
    if stats.empty:return {}
    stats=stats.copy();stats['points']=fantasy_points(stats,scoring or DEFAULT_SCORING)
    models={}
    for position in ('QB','RB','WR','TE','K'):
        subset=stats[stats.position==position].copy()
        if position_scoring and position in position_scoring:subset['points']=fantasy_points(subset,position_scoring[position])
        if subset.empty:continue
        grouped=subset.groupby(['game_id','team','opponent_team'],as_index=False).points.sum()
        xs=[];ys=[];weights=[];counts={};raw={};opponents={}
        for row in grouped.to_dict('records'):
            game=schedule.get(row['game_id'])
            if game is None:continue
            team=row['team'];opponent=row['opponent_team'];date=kickoff(game)
            offense_key=f"{team}:{game.get('season',as_of.year)}"
            x={f'offense:{offense_key}':5.,f'defense:{opponent}':1.,'home':float(team==game['home_team']) if game.get('location')!='Neutral' else .5}
            w={'temperature_f':number(game.get('temp'),60),'wind_mph':number(game.get('wind')),'wet':bool(game.get('wet',False))}
            if str(game.get('roof','')).lower() in ('dome','closed'):w={'temperature_f':70,'wind_mph':0}
            x.update(weather_features(team,w));xs.append(x);ys.append(math.log1p(max(0,row['points'])));weights.append(.5**((as_of-date).days/200))
            counts[opponent]=counts.get(opponent,0)+1;raw.setdefault(opponent,[]).append(row['points']);opponents.setdefault(opponent,[]).append(offense_key)
        if len(xs)<80:continue
        model=ridge_fit(xs,ys,weights,alpha=10)
        model['defenses']={t:{'games':counts[t],'raw_points_allowed':float(np.mean(raw[t])),'opponent_offense_strength':float(np.mean([5*model['coefficients'].get('offense:'+o,0) for o in opponents[t]])),'adjusted_multiplier':float(np.clip(math.exp(model['coefficients'].get('defense:'+t,0)),.7,1.3))} for t in counts}
        model['method']='Joint season-specific offense/defense regression; offensive strength receives less shrinkage than defensive effects, with recency and home/weather controls'
        models[position]=model
    return models


def matchup_adjustment(models,position,team,opponent,home,weather):
    model=models.get(position)
    if not model:return {'multiplier':1.,'status':'insufficient history'}
    coefs=model['coefficients'];defense=float(coefs.get('defense:'+opponent,0))
    home_effect=coefs.get('home',0)*(float(home)-.5)
    weather_x=weather_features(team,weather or {})
    weather_effect=sum(coefs.get(k,0)*v for k,v in weather_x.items())
    return {'multiplier':float(np.clip(math.exp(defense+home_effect+weather_effect),.65,1.4)),'defense_multiplier':float(np.clip(math.exp(defense),.7,1.3)),'home_multiplier':math.exp(home_effect),'weather_multiplier':float(np.clip(math.exp(weather_effect),.8,1.2)),'opponent':opponent,'evidence':model['defenses'].get(opponent,{}),'status':'modeled','method':model['method']}


def play_calling(cache_path,season,as_of,schedule):
    frames=[]
    for y in range(season-2,season+1):
        frame=read_frame(Path(cache_path)/f'pbp_{y}.parquet')
        if not frame.empty:frames.append(frame)
    if not frames:return {'status':'unavailable','teams':{},'players':{}}
    pbp=pd.concat(frames,ignore_index=True);pbp=pbp[pbp.game_id.isin(schedule)].drop_duplicates(['game_id','play_id'])
    offense=pbp[(pbp.get('pass',0)==1)|(pbp.get('rush',0)==1)].copy()
    if offense.empty:return {'status':'no completed plays','teams':{},'players':{}}
    offense['neutral']=offense.score_differential.abs().le(7)&offense.game_seconds_remaining.gt(120)
    xs=[];ys=[];weights=[];teams={}
    for (gid,team),part in offense.groupby(['game_id','posteam']):
        game=schedule[gid];neutral=part[part.neutral]
        if len(neutral)<10:continue
        rate=float(neutral['pass'].mean());weather={'wind_mph':number(game.get('wind')),'temperature_f':number(game.get('temp'),60),'wet':bool(game.get('wet',False))}
        if str(game.get('roof','')).lower() in ('dome','closed'):weather={'wind_mph':0,'temperature_f':70}
        x={'team:'+team:1.,'home':float(team==game['home_team'])};x.update(weather_features(team,weather))
        xs.append(x);ys.append(rate);weights.append(.5**((as_of-kickoff(game)).days/200))
        teams.setdefault(team,[]).append({'date':kickoff(game).isoformat(),'season':game['season'],'week':game['week'],'plays':len(part),'neutral_pass_rate':rate,'pass_epa':float(part.loc[part['pass']==1,'epa'].mean()),'rush_epa':float(part.loc[part['rush']==1,'epa'].mean())})
    players={}
    latest=offense[offense.season==max(offense.season)]
    for field,role in [('receiver_player_id','receiving'),('rusher_player_id','rushing')]:
        for pid,part in latest.dropna(subset=[field]).groupby(field):
            games=max(1,part.game_id.nunique());r=players.setdefault(pid,{})
            r.update({role+'_red_zone_per_game':float(part.yardline_100.le(20).sum()/games),role+'_goal_line_per_game':float(part.yardline_100.le(5).sum()/games),role+'_epa_per_opportunity':float(part.epa.mean()),role+'_opportunities_per_game':len(part)/games})
    return {'status':'modeled','model':ridge_fit(xs,ys,weights,alpha=15) if len(xs)>80 else None,'teams':{t:sorted(rows,key=lambda r:r['date'])[-8:] for t,rows in teams.items()},'players':players,'plays':len(offense),'limitation':'Routes are not inferred from targets; current route participation requires a separate licensed or delayed feed'}


def play_type_adjustment(calling,team,weather):
    model=calling.get('model');history=calling.get('teams',{}).get(team,[])
    if not model or not history:return {'pass_factor':1.,'rush_factor':1.}
    x={'team:'+team:1.,'home':.5};x.update(weather_features(team,weather or {}))
    forecast=float(np.clip(predict(model,x),.3,.8));baseline=float(np.mean([r['neutral_pass_rate'] for r in history]))
    return {'pass_factor':float(np.clip(forecast/max(.1,baseline),.85,1.15)),'rush_factor':float(np.clip((1-forecast)/max(.1,1-baseline),.85,1.15)),'predicted_neutral_pass_rate':forecast,'recent_neutral_pass_rate':baseline}


def enrich_weather(schedule,cache_path,season):
    weather={}
    for year in range(season-2,season+1):
        path=Path(cache_path)/f'pbp_{year}.parquet'
        if not path.exists():continue
        try:
            frame=pd.read_parquet(path,columns=['game_id','weather']).dropna().drop_duplicates('game_id')
            for row in frame.itertuples():
                import re
                weather[row.game_id]=bool(re.search(r'\b(rain|rainy|raining|snow|snowy|snowing|showers|drizzle)\b',str(row.weather),re.I))
        except (ValueError,KeyError):continue
    result=schedule.copy()
    if not result.empty:result['wet']=result.game_id.map(weather).fillna(False).astype(bool)
    return result
