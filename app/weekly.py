"""League-specific weekly distributions and exact constrained lineup assignment."""
import hashlib
import math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from app.domain import ELIGIBLE
from app.scoring import score_week,slots
from app.storage import DATA,now
from app.injuries import availability
from app.availability import participation
from app.context_models import matchup_adjustment,play_type_adjustment

def recent_usage(player,season,week,cache_path):
    path=Path(cache_path)/f'stats_{season}.parquet'
    if not path.exists():return {},None
    frame=pd.read_parquet(path)
    if 'player_id' not in frame:return {},None
    frame=frame[(frame.player_id==player.ids.get('gsis',player.id))&(frame.week<week)&(frame.season_type=='REG')].sort_values('week').tail(4)
    if frame.empty:return {},None
    weights=np.power(.7,np.arange(len(frame)-1,-1,-1))
    values={k:float(np.average(frame[k].fillna(0),weights=weights)) for k in ('targets','carries','attempts','receptions','receiving_yards','rushing_yards','passing_yards','receiving_tds','rushing_tds','passing_tds','target_share','air_yards_share') if k in frame}
    return values,{'weeks':frame.week.astype(int).tolist(),'source':path.name,'sample_size':len(frame)}

def game_context(player,season,week,cache_path):
    path=Path(cache_path)/'schedules.parquet'
    if not path.exists():return {}
    frame=pd.read_parquet(path);team=player.team
    frame=frame[(frame.season==season)&(frame.week==week)&(frame.game_type=='REG')&((frame.home_team==team)|(frame.away_team==team))]
    if frame.empty:return {}
    row=frame.iloc[0];out={'game_id':str(row.game_id),'opponent':str(row.away_team if row.home_team==team else row.home_team),'home':bool(row.home_team==team),'date':str(row.get('gameday',''))}
    # nflverse kickoff clocks are Eastern, not UTC/local machine time.
    try:
        from zoneinfo import ZoneInfo
        date=datetime.fromisoformat(str(row.gameday)+'T'+str(row.gametime)).replace(tzinfo=ZoneInfo('America/New_York'))
        out['kickoff']=date.astimezone(timezone.utc).isoformat();out['started']=date<=datetime.now(timezone.utc)
    except (ValueError,KeyError,TypeError):out['started']=False
    return out

def project_week(player,league,week,provider,injury,source_health,cache_path=DATA/'cache',draws=4096,context=None):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{player.id}:{league.season}:{week}'.encode()).digest()[:4],'little'))
    context=context or {}
    warnings=[]
    if not context.get('games'):warnings.append('No matching week context: opponent, weather and personnel adjustments are unavailable')
    stats={k:v/max(1,player.games) for k,v in player.stats.items()}
    usage,evidence=recent_usage(player,league.season,week,cache_path)
    if evidence:
        weight=len(evidence['weeks'])/(len(evidence['weeks'])+4)
        for opportunity,related in [('targets',['receptions','receiving_yards','receiving_tds']),('carries',['rushing_yards','rushing_tds']),('attempts',['passing_yards','passing_tds','passing_interceptions'])]:
            if opportunity not in usage:continue
            base=stats.get(opportunity,0);updated=(1-weight)*base+weight*usage[opportunity]
            if base>0:
                for key in related:stats[key]=stats.get(key,0)*updated/base
            stats[opportunity]=updated
    else:warnings.append('No prior current-season usage available; historical baseline used')
    game=game_context(player,league.season,week,cache_path)
    live=next((g for g in context.get('games',[]) if g.get('game_id')==game.get('game_id')),{})
    weather=live.get('weather',{})
    replacement=context.get('rosters',{}).get(player.team,{}).get('opportunities',{}).get(player.ids.get('espn',''),{})
    for opportunity,related in [('targets',['receptions','receiving_yards','receiving_tds']),('carries',['rushing_yards','rushing_tds']),('attempts',['passing_yards','passing_tds','passing_interceptions'])]:
        if opportunity in replacement and stats.get(opportunity,0)>0:
            ratio=replacement[opportunity]/stats[opportunity]
            for k in related:stats[k]=stats.get(k,0)*ratio
            stats[opportunity]=replacement[opportunity]
    models=context.get('league_matchups',{}).get(league.id,context.get('matchups',{}))
    adjustment=matchup_adjustment(models,player.position,player.team,game.get('opponent',''),game.get('home',False),weather)
    calling=play_type_adjustment(context.get('play_calling',{}),player.team,weather)
    # Learned fantasy weather effects already include play-calling changes; do not apply both multipliers.
    game_forecast=live.get('forecast') or {}
    side='home' if game.get('home') else 'away'
    components=game_forecast.get('components',{}).get(side,{})
    raw=components.get('raw_points') or 22.
    personnel=float(np.clip(1+(components.get('offensive_replacements',0)+components.get('opposing_defensive_replacements',0))/max(10,raw),.85,1.15))
    adjustment['personnel_multiplier']=personnel
    baseline=score_week(stats,league)*adjustment['multiplier']*personnel

    espn=provider.get('espn_projection')
    if espn is not None and not math.isfinite(float(espn)):raise ValueError('Nonfinite ESPN projection')
    incomplete=provider.get('independent_scoring_incomplete',False)
    if incomplete or not player.stats:
        if espn is None:return None,None
        center=float(espn);weights={'independent':0,'espn':1}
        warnings.append('ESPN exact league forecast used because independent scoring/history is incomplete')
    elif espn is None:
        center=baseline;weights={'independent':1,'espn':0};warnings.append('ESPN weekly projection unavailable')
    else:
        weight=float(context.get('blend_weights',{}).get(player.position,.65))
        center=weight*baseline+(1-weight)*float(espn);weights={'independent':weight,'espn':1-weight}
        warnings.append('Blend uses completed archived errors when sufficient; otherwise provisional 65/35')
    status=injury.get('status') or provider.get('injury_status','UNKNOWN')
    stamp=injury.get('reported_at')
    if stamp:
        try:
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(stamp.replace('Z','+00:00'))).total_seconds()/86400
            if age>10 and status not in ('INJURED_RESERVE','IR','SUSPENDED'):
                warnings.append('Injury designation older than 10 days; verify game-week status')
                status=provider.get('injury_status','UNKNOWN')
        except ValueError:warnings.append('Injury timestamp cannot be parsed')
    synced_status=provider.get('injury_status','UNKNOWN')
    if synced_status!='UNKNOWN' and availability(synced_status)<availability(status):
        warnings.append('Conflicting designations: conservatively using the more restrictive synced ESPN status')
        status=synced_status
    if source_health!='fresh':warnings.append(f'Injury feed {source_health}; verify active status before kickoff')
    if not injury and status=='UNKNOWN':warnings.append('No verified game status; missing report does not establish health')
    participation_evidence=participation(status,injury)
    playing=participation_evidence['probability']
    likely=participation_evidence['likely']
    bye=player.bye==week
    if bye:playing=0;likely=False
    workload=rng.lognormal(-.5*player.workload_cv**2,player.workload_cv,draws)
    efficiency=rng.lognormal(-.5*player.efficiency_cv**2,player.efficiency_cv,draws)
    weekly=rng.lognormal(-.5*.5**2,.5,draws)
    shared=np.random.default_rng(int.from_bytes(hashlib.sha256(f"{game.get('game_id',week)}:environment".encode()).digest()[:4],'little')).lognormal(-.5*.12**2,.12,draws)
    team_shared=np.random.default_rng(int.from_bytes(hashlib.sha256(f"{player.team}:{league.season}:{week}:passing".encode()).digest()[:4],'little')).lognormal(-.5*.18**2,.18,draws)
    conditional=center*workload*efficiency*weekly*shared*(team_shared if player.position in ('QB','WR','TE') else 1)
    points_if_active=float(np.mean(conditional))
    samples=conditional if likely else np.zeros(draws)

    # Frozen actual scoring is used only when both the game lock and current-week actual are known.
    locked=bool(provider.get('locked')) or game.get('started',False)
    if locked and provider.get('actual_points') is not None:
        if live.get('completed'):
            samples[:]=float(provider['actual_points'])
        elif live.get('state')=='in' and live.get('remaining_fraction') is not None:
            samples=float(provider['actual_points'])+conditional*live['remaining_fraction']
            warnings.append('Live estimate = current points plus clock-scaled remaining production; not play-by-play usage adjusted')
        else:
            samples[:]=float(provider['actual_points'])
            warnings.append('Live clock unavailable: only banked points are shown; remaining projection unavailable')

    q=np.quantile(samples,[.1,.25,.5,.75,.9]);mean=float(np.mean(samples))
    row={'id':player.id,'name':player.name,'position':player.position,'team':player.team,'mean':mean,'median':float(q[2]),'p10':float(q[0]),'p25':float(q[1]),'p75':float(q[3]),'p90':float(q[4]),'boom_threshold':max(1,center)*1.5,'bust_threshold':max(1,center)*.6,'boom':float(np.mean(samples>max(1,center)*1.5)),'bust':float(np.mean(samples<max(1,center)*.6)),'scoring':league.scoring,'independent_projection':baseline,'espn_projection':espn,'weights':weights,'injury_status':status,'play_probability':playing,'likely_to_play':likely,'points_if_active':points_if_active,'participation_weighted_points':points_if_active*playing,'participation_evidence':participation_evidence,'context':{'matchup':adjustment,'play_calling':calling,'replacement':replacement,'weather':weather,'advanced_usage':context.get('play_calling',{}).get('players',{}).get(player.ids.get('gsis',player.id),{})},'injury':injury,'bye':bye,'locked':locked,'eligible_slots':provider.get('eligible_slots',[]),'current_slot':provider.get('slot'),'game':game,'usage':usage,'usage_evidence':evidence,'warnings':warnings,'confidence':'limited' if warnings or status not in ('ACTIVE','NORMAL','HEALTHY') else 'moderate'}
    return row,samples

def choose_lineup(rows,samples,league,risk,current,utilities=None):
    slot_names=slots(league);count=len(rows);row_index={r['id']:i for i,r in enumerate(rows)}
    matrix=np.full((len(slot_names),count+len(slot_names)),-1e12)
    matrix[:,count:]=-1e8
    field={'balanced':'mean','floor':'p25','upside':'p75'}[risk]
    for i,slot in enumerate(slot_names):
        for j,row in enumerate(rows):
            if (slot in row['eligible_slots'] if row.get('eligible_slots') else row['position'] in ELIGIBLE[slot]) and row['current_slot']!='IR' and (row.get('likely_to_play',row['play_probability']>=.5) or row['locked']):matrix[i,j]=utilities[j] if utilities is not None else row[field]
    used_slots=set()
    for j,row in enumerate(rows):
        if not row['locked']:continue
        slot=row.get('current_slot')
        if slot in ('BE','IR',None):matrix[:,j]=-1e12;continue
        target=next((i for i,s in enumerate(slot_names) if s==slot and i not in used_slots),None)
        if target is None:raise ValueError('Locked player cannot be reconciled to current lineup slots; re-sync ESPN')
        used_slots.add(target);matrix[target,:]=-1e12;matrix[:,j]=-1e12;matrix[target,j]=row[field]
    a,b=linear_sum_assignment(matrix,maximize=True)
    if any(matrix[i,j]<-1e10 for i,j in zip(a,b)):raise ValueError('No legal lineup satisfies the synced game locks')
    lineup=[];total=np.zeros(samples.shape[1]);selected=set()
    for i,j in zip(a,b):
        row=rows[j] if j<count else None
        lineup.append({'slot':slot_names[i],'player':row})
        if row:total+=samples[j];selected.add(row['id'])
    current_ids={p['player_id'] for p in current};current_total=np.sum([samples[row_index[p]] for p in current_ids if p in row_index],axis=0) if current_ids else np.zeros(samples.shape[1])
    return {'starters':lineup,'bench':[r for r in rows if r['id'] not in selected],'projected_points':float(total.mean()),'p10':float(np.quantile(total,.1)),'p90':float(np.quantile(total,.9)),'current_projected_points':float(current_total.mean()) if current_ids else None,'improvement':float(total.mean()-current_total.mean()) if current_ids else None,'start':[r['name'] for r in rows if r['id'] in selected-current_ids],'sit':[r['name'] for r in rows if r['id'] in current_ids-selected],'empty_slots':sum(x['player'] is None for x in lineup)},total

def lineup_advice(players,league,roster_ids,meta,health,week,risk='balanced',cache_path=DATA/'cache',context=None,opponent_ids=None):
    if risk not in ('balanced','floor','upside','win'):raise ValueError('Unknown risk objective')
    if meta and (meta.get('week')!=week or meta.get('season')!=league.season):raise ValueError('Synced ESPN snapshot is for a different week/season. Re-sync first.')
    by_id={p.id:p for p in players};rows=[];samples=[];missing=[]
    for pid in roster_ids:
        player=by_id[pid];provider=meta.get('weekly',{}).get(pid,{}).copy()
        from app.scoring_support import player_scoring
        scoring_league,incomplete=player_scoring(league,player,meta,provider)
        provider['independent_scoring_incomplete']=incomplete
        injury=health.get('players',{}).get(player.ids.get('espn',''),{}) if health.get('season',league.season)==league.season else {}
        row,draw=project_week(player,scoring_league,week,provider,injury,health['status'],cache_path,context=context)
        if row is None:missing.append(player.name)
        else:rows.append(row);samples.append(draw)
    if not rows:raise ValueError('No roster projections. Sync your ESPN league or import current rosters first.')
    # Opponent optimization uses the same shared game/team draws as this roster.
    current=meta.get('current_lineups',{}).get(str(league.my_team),[])
    array=np.asarray(samples)
    advice,_=choose_lineup(rows,array,league,'balanced' if risk=='win' else risk,current)
    if opponent_ids:
        opponent_rows=[];opponent_samples=[]
        for pid in opponent_ids:
            if pid not in by_id:continue
            p=by_id[pid];provider=meta.get('weekly',{}).get(pid,{}).copy()
            scoring_league,incomplete=player_scoring(league,p,meta,provider);provider['independent_scoring_incomplete']=incomplete
            injury=health.get('players',{}).get(p.ids.get('espn',''),{})
            row,draw=project_week(p,scoring_league,week,provider,injury,health['status'],cache_path,context=context)
            if row is not None:opponent_rows.append(row);opponent_samples.append(draw)
        if opponent_rows:
            opponent_index=meta.get('opponents',{}).get(str(league.my_team))
            opponent_advice,_=choose_lineup(opponent_rows,np.asarray(opponent_samples),league,'balanced',meta.get('current_lineups',{}).get(str(opponent_index),[]))
            def uncertain(row,draw):
                if row['locked']:return draw
                random=np.random.default_rng(int.from_bytes(hashlib.sha256(f"availability:{row['id']}:{league.season}:{week}".encode()).digest()[:4],'little'))
                return draw*(random.random(len(draw))<row['play_probability'])
            their_ids={s['player']['id'] for s in opponent_advice['starters'] if s['player']}
            their_total=sum((uncertain(r,d) for r,d in zip(opponent_rows,opponent_samples) if r['id'] in their_ids),np.zeros(array.shape[1]))
            adjusted=np.asarray([uncertain(r,d) for r,d in zip(rows,array)])
            advice=matchup_optimize(rows,array,adjusted,league,current,advice,their_total,risk=='win')
            advice['opponent_projected_points']=opponent_advice['projected_points']
        elif risk=='win':raise ValueError('Opponent roster has no usable projections; re-sync before optimizing matchup probability')
    elif risk=='win':raise ValueError('Matchup optimization needs a synced opponent roster for this week')
    advice.update(week=week,risk=risk,generated_at=now(),league_name=league.name,roster_updated=meta.get('updated'),injury_source={k:v for k,v in health.items() if k!='players'},missing_projections=missing,notes=['Boom = above 150% of healthy weekly forecast; bust = below 60%.','Points are conditional on playing. Likely active (probability at least 50%) retains the forecast; unlikely/out/bye is zero. Report-language probabilities are heuristic and shown separately.','Context uses opponent-adjusted position performance, learned home/weather effects, depth-chart replacements and play-by-play evidence when available. Context changes affect the independent estimate only, avoiding duplicate ESPN adjustments.','Shared game/team simulation factors introduce correlation, but their magnitudes and weekly intervals remain provisional. ESPN forecasts may already include injury assumptions.'])
    return advice


def matchup_optimize(rows,samples,uncertain_samples,league,current,advice,opponent_total,optimize):
    index={r['id']:i for i,r in enumerate(rows)}
    def selected(a):return {s['player']['id'] for s in a['starters'] if s['player']}
    def probability(ids):
        total=sum((uncertain_samples[index[pid]] for pid in ids),np.zeros(samples.shape[1]))
        return float(np.mean(total>opponent_total)+.5*np.mean(total==opponent_total))
    chosen=selected(advice);initial=probability(chosen);best=initial
    if optimize:
        for _ in range(5):
            winner=None
            for out in chosen:
                if rows[index[out]]['locked']:continue
                for candidate in rows:
                    if candidate['id'] in chosen or candidate['locked'] or not candidate.get('likely_to_play',True):continue
                    wanted=(chosen-{out})|{candidate['id']}
                    trial,_=choose_lineup(rows,samples,league,'balanced',current,np.array([1e6 if r['id'] in wanted else -1e6 for r in rows]))
                    ids=selected(trial)
                    if ids!=wanted:continue
                    chance=probability(ids)
                    if chance>best+1e-6:best=chance;winner=trial
            if winner is None:break
            advice=winner;chosen=selected(advice)
    advice['estimated_win_probability']=best
    advice['baseline_win_probability']=initial
    advice['matchup_method']='Legal local-swap search' if optimize else 'Evaluate selected lineup'
    advice['matchup_assumptions']='Includes separate participation uncertainty and shared game/team factors; opponent uses its modeled best balanced lineup. Local search is not a guaranteed global optimum.'
    return advice
