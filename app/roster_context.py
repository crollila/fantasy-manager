"""Explicit depth-chart IDs, historical replacement cohorts and current unit changes."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from app.availability import participation
from app.game_model import kickoff,number,features,ridge_fit
from app.research_sources import read_frame


def group_position(pos):
    pos=str(pos).upper()
    if pos in ('QB','RB','WR','TE','K','PK'):return 'K' if pos=='PK' else pos
    if pos in ('LT','RT','LG','RG','C','T','G','OT','OG','OL'):return 'OL'
    if pos in ('DE','DT','NT','DL','LDE','RDE','LDT','RDT'):return 'DL'
    if 'LB' in pos:return 'LB'
    if pos in ('CB','LCB','RCB','NB','DB','S','SS','FS'):return 'DB'
    return 'OTHER'


def id_maps(cache_path):
    frame=read_frame(Path(cache_path)/'players.parquet');by_gsis={};by_espn={}
    for r in frame.fillna('').to_dict('records'):
        item={'gsis':str(r.get('gsis_id','')),'pfr':str(r.get('pfr_id','')),'espn':str(r.get('espn_id','')).removesuffix('.0')}
        if item['gsis']:by_gsis[item['gsis']]=item
        if item['espn']:by_espn[item['espn']]=item
    return by_gsis,by_espn


def replacement_study(cache_path,season,schedule,stats):
    gsis,_=id_maps(cache_path);examples=[];xs=[];ys=[];weights=[];counts={};performance={}
    stat_lookup={(r['game_id'],str(r['player_id'])):r for r in stats.to_dict('records')} if not stats.empty else {}
    for year in range(season-2,season):
        depth=read_frame(Path(cache_path)/f'depth_{year}.parquet');snaps=read_frame(Path(cache_path)/f'snaps_{year}.parquet')
        if depth.empty or snaps.empty:continue
        snap_lookup={(g,t):dict(zip(part.pfr_player_id,part.offense_snaps+part.defense_snaps)) for (g,t),part in snaps.groupby(['game_id','team'])}
        dated='dt' in depth
        if dated:
            depth=depth.copy();depth['stamp']=pd.to_datetime(depth.dt,utc=True)
            lookup={team:(sorted(part.stamp.unique()),{stamp:g for stamp,g in part.groupby('stamp')}) for team,part in depth.groupby('team')}
        else:lookup={(str(t),int(w)):g for (t,w),g in depth.dropna(subset=['week']).groupby(['club_code','week'])}
        for game in schedule.values():
            if int(game['season'])!=year:continue
            context={};detail=[]
            for side in ('home','away'):
                team=game[side+'_team'];day=kickoff(game)
                if dated:
                    stamps,parts=lookup.get(team,([],{}));before=[d for d in stamps if d<=day]
                    frame=parts[before[-1]] if before else pd.DataFrame()
                    if before and (day-before[-1]).days>14:frame=pd.DataFrame()
                else:frame=lookup.get((team,int(game['week'])),pd.DataFrame())
                unit={}
                if not frame.empty:
                    ranks=pd.to_numeric(frame['pos_rank' if dated else 'depth_team'],errors='coerce')
                    f=frame.assign(rank=ranks)
                    slot_column='pos_slot' if dated else 'depth_position'
                    # Formation/group distinguishes identically numbered offensive and defensive slots.
                    group_columns=['pos_grp',slot_column] if dated else ['formation',slot_column]
                    appearances=snap_lookup.get((game['game_id'],team),{})
                    seen=set()
                    for _,slot in f.groupby(group_columns):
                        starters=slot[slot['rank']==1]
                        if starters.empty:continue
                        first=starters.iloc[0];pid=str(first.get('gsis_id',''))
                        if not pid or pid in seen:continue
                        seen.add(pid);pfr=gsis.get(pid,{}).get('pfr')
                        if not pfr:continue
                        position=group_position(first.get('pos_abb') if dated else first.get('position'))
                        if position=='OTHER':continue
                        if appearances.get(pfr,0)>=5:continue
                        backups=slot[slot['rank']>=2].sort_values('rank')
                        active=[str(r.gsis_id) for r in backups.itertuples() if appearances.get(gsis.get(str(r.gsis_id),{}).get('pfr'),0)>=15]
                        if not active:continue
                        backup=active[0];unit[position]=unit.get(position,0)+1;counts[position]=counts.get(position,0)+1
                        row=stat_lookup.get((game['game_id'],backup))
                        if row:
                            performance.setdefault(position,[]).append({k:number(row.get(k)) for k in ('targets','carries','attempts','receiving_yards','rushing_yards','passing_yards')})
                        detail.append({'team':team,'position':position,'starter':pid,'replacement':backup})
                context[side]=unit
            for side in ('home','away'):
                other='away' if side=='home' else 'home';x=features(game,side)
                x.update({'missing_offense:'+p:float(n) for p,n in context[side].items() if p in ('QB','RB','WR','TE','OL','K')})
                x.update({'missing_defense:'+p:float(n) for p,n in context[other].items() if p in ('DL','LB','DB')})
                xs.append(x);ys.append(number(game[side+'_score']));weights.append(.7**(season-1-year))
            examples.extend(detail)
    model=ridge_fit(xs,ys,weights,alpha=35) if len(xs)>=100 else {'coefficients':{}}
    estimates={}
    for position,n in counts.items():
        side='defense' if position in ('DL','LB','DB') else 'offense'
        estimates[position]={'observations':n,'points_delta':float(np.clip(model['coefficients'].get('missing_'+side+':'+position,0),-3,3)) if n>=30 else 0.,'backup_average':{k:float(np.mean([r[k] for r in performance.get(position,[])])) for k in ('targets','carries','attempts','receiving_yards','rushing_yards','passing_yards')} if performance.get(position) else {},'performance_samples':len(performance.get(position,[]))}
    return {'positions':estimates,'observations':len(examples),'method':'Historical pregame depth charts joined by GSIS/PFR ID to actual participation; regularized offense/opponent/home/weather regression','limitation':'Observational replacement effects, not causal injury effects; historical in-game absences and depth-chart errors can remain. Fewer than 30 examples gives a neutral points adjustment.'}


def current_rosters(depth,health,study,stats,cache_path,as_of=None,season=None):
    as_of=as_of or datetime.now(timezone.utc);_,espn=id_maps(cache_path)
    averages={}
    prior_snaps=read_frame(Path(cache_path)/f'snaps_{(season or as_of.year)-1}.parquet')
    prior_units={};prior_shares={}
    if not prior_snaps.empty:
        prior_snaps=prior_snaps[prior_snaps.game_type=='REG'].copy()
        prior_snaps['unit']=prior_snaps.position.map(group_position)
        prior_snaps['share']=np.where(prior_snaps.unit.isin(['OL','QB','RB','WR','TE']),prior_snaps.offense_pct,prior_snaps.defense_pct)
        prior_shares=prior_snaps.groupby('pfr_player_id').share.mean().to_dict()
        for (team,unit),part in prior_snaps.groupby(['team','unit']):
            prior_units[(team,unit)]=part.groupby('pfr_player_id').share.sum().sort_values(ascending=False).index.tolist()
    if not stats.empty:
        latest=stats[stats.season==stats.season.max()]
        for pid,frame in latest.groupby('player_id'):
            averages[str(pid)]={k:float(frame[k].fillna(0).mean()) for k in ('targets','carries','attempts') if k in frame}
    result={}
    for team,data in depth.items():
        r={'changes':[],'notes':[],'availability':[],'offense_points_delta':0.,'defense_points_delta':0.,'opportunities':{},'source_url':data.get('source_url')};seen=set()
        for unit in data.get('units',[]):
            position=group_position(unit['position']);players=unit['players']
            if not players or position=='OTHER':continue
            first=players[0]
            if first['espn_id'] in seen:continue
            seen.add(first['espn_id'])
            first_report=health.get('players',{}).get(first['espn_id'],{})
            first_participation=participation(first_report.get('status','UNKNOWN'),first_report,as_of)
            r['availability'].append({'espn_id':first['espn_id'],'name':first['name'],'position':position,'status':first_report.get('status','UNKNOWN'),'probability':first_participation['probability'],'likely':first_participation['likely'],'reported_at':first_report.get('reported_at'),'source_url':first_report.get('source_url') or data.get('source_url')})
            def likely(p):
                report=health.get('players',{}).get(p['espn_id'],{})
                return participation(report.get('status','UNKNOWN'),report,as_of)['likely']
            active=next((p for p in players if likely(p)),None)
            if position=='QB' and active:r['qb_gsis']=espn.get(active['espn_id'],{}).get('gsis')
            if likely(first):continue
            evidence=study.get('positions',{}).get(position,{})
            delta=evidence.get('points_delta',0.)
            if position=='QB' and active and r.get('qb_gsis') in averages:delta=0. # learned QB identity already changes game score
            side='defense' if position in ('DL','LB','DB') else 'offense'
            r[side+'_points_delta']+=delta
            change={'position':position,'starter':first['name'],'replacement':active['name'] if active else None,'starter_id':first['espn_id'],'replacement_id':active['espn_id'] if active else None,'points_delta':delta,'historical_examples':evidence.get('observations',0),'backup_average':evidence.get('backup_average',{})}
            r['changes'].append(change);r['notes'].append(f'{position}: {first["name"]} unlikely/out; {active["name"] if active else "no verified replacement"}. {evidence.get("observations",0)} historical replacement examples.')
            if active and position in ('QB','RB','WR','TE'):
                key={'QB':'attempts','RB':'carries','WR':'targets','TE':'targets'}[position]
                starter=averages.get(espn.get(first['espn_id'],{}).get('gsis'),{})
                backup=averages.get(espn.get(active['espn_id'],{}).get('gsis'),{})
                learned=evidence.get('backup_average',{}).get(key)
                if learned is not None and evidence.get('performance_samples',0)>=20:
                    volume=.5*starter.get(key,learned)+.5*learned
                    r['opportunities'][active['espn_id']]={key:max(backup.get(key,0),volume),'method':'Blend of vacated starter usage and observed replacement workload'}
        # Established transfers are not automatically treated as backups. Low previous participation
        # is only a replacement-exposure proxy; unknown rookie history gets no invented penalty.
        active_units={};changed_units={c['position'] for c in r['changes']}
        for unit in data.get('units',[]):
            position=group_position(unit['position'])
            if position not in ('OL','DL','LB','DB') or position in changed_units:continue
            active=next((p for p in unit['players'] if participation(health.get('players',{}).get(p['espn_id'],{}).get('status','UNKNOWN'),health.get('players',{}).get(p['espn_id'],{}),as_of)['likely']),None)
            if active:active_units.setdefault(position,{})[active['espn_id']]=active
        r['lineup_turnover']={}
        for position,unit in active_units.items():
            prior=prior_units.get((team,position),[])[:len(unit)]
            current=[(p,espn.get(eid,{}).get('pfr')) for eid,p in unit.items()]
            new=[(p,pfr) for p,pfr in current if pfr and pfr not in prior]
            if not prior or not new:continue
            exposure=sum(max(0,.65-prior_shares[pfr])/.65 for _,pfr in new if pfr in prior_shares)
            delta=exposure*study.get('positions',{}).get(position,{}).get('points_delta',0.)
            side='offense' if position=='OL' else 'defense'
            r[side+'_points_delta']+=delta
            r['lineup_turnover'][position]={'new_starters':[p['name'] for p,_ in new],'backup_equivalent_exposure':exposure,'points_delta':delta,'method':'Prior-season snap participation proxy; established starters are not treated as backups'}
            r['notes'].append(f'{position}: {len(new)} starters differ from last season; {exposure:.1f} backup-equivalent exposures from prior snaps. Unknown histories receive no assumed penalty.')
        r['offense_points_delta']=float(np.clip(r['offense_points_delta'],-7,7));r['defense_points_delta']=float(np.clip(r['defense_points_delta'],-7,7))
        result[team]=r
    return result
