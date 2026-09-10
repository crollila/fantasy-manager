"""Independent, regularized team scoring model. Market lines are benchmarks, never labels."""
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import norm

MODEL_VERSION = '0.5.0-evidence-1'


def number(value, default=0.):
    try:
        n=float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):return default


def kickoff(row):
    from zoneinfo import ZoneInfo
    try:
        return datetime.fromisoformat(str(row['gameday'])+'T'+str(row['gametime'])).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
    except (KeyError, ValueError):
        # Unknown kickoff cannot be used to create a new official prediction.
        return None


def weather_features(team, weather):
    wind=max(0,number(weather.get('wind_mph'),0)-10)/10
    cold=max(0,40-number(weather.get('temperature_f'),60))/20
    heat=max(0,number(weather.get('temperature_f'),60)-85)/15
    wet=float(bool(weather.get('wet',number(weather.get('precipitation_mm'))>.1)))
    return {'wind':wind,'cold':cold,'heat':heat,'wet':wet,f'team_wind:{team}':wind,f'team_cold:{team}':cold,f'team_heat:{team}':heat,f'team_wet:{team}':wet}


def features(row, side, weather=None, qb=None):
    other='away' if side=='home' else 'home'
    team=str(row[f'{side}_team']);opponent=str(row[f'{other}_team'])
    neutral=str(row.get('location','Home')).lower()=='neutral'
    roof=str(row.get('roof','')).lower()
    if weather is None:
        weather={'wind_mph':number(row.get('wind')),'temperature_f':number(row.get('temp'),60),'wet':bool(row.get('wet',False))}
        if roof in ('dome','closed'):weather={'wind_mph':0,'temperature_f':70}
    result={f'offense:{team}':1.,f'defense:{opponent}':1.,'home':.5 if neutral else float(side=='home'), 'rest':max(-7,min(7,number(row.get(side+'_rest'),7)-number(row.get(other+'_rest'),7)))/7}
    qb=qb or row.get(side+'_qb_id')
    if isinstance(qb,str) and qb:result['qb:'+qb]=1.
    result.update(weather_features(team,weather))
    return result


def ridge_fit(xs,ys,weights=None,alpha=16):
    vector=DictVectorizer(sparse=True)
    matrix=vector.fit_transform(xs)
    model=Ridge(alpha=alpha,solver='lsqr')
    model.fit(matrix,np.asarray(ys),sample_weight=weights)
    return {'intercept':float(model.intercept_),'coefficients':dict(zip(vector.get_feature_names_out(),map(float,model.coef_))),'observations':len(ys)}


def predict(model, x):
    return model['intercept']+sum(model['coefficients'].get(k,0)*v for k,v in x.items())


def train_games(schedule, as_of):
    rows=[];xs=[];ys=[];weights=[]
    for row in schedule.to_dict('records'):
        date=kickoff(row)
        # The schedule has no reliable live/final flag. Exclude the game window so
        # a partially populated live score cannot become a completed training label.
        if not date or date+timedelta(hours=8)>=as_of or (as_of-date).days>4*366 or pd.isna(row.get('home_score')) or pd.isna(row.get('away_score')) or row.get('game_type')!='REG':continue
        rows.append(row)
        for side in ('home','away'):
            xs.append(features(row,side));ys.append(number(row[side+'_score']));weights.append(.5**((as_of-date).days/365))
    if len(rows)<80:raise ValueError('At least 80 completed historical games are required')
    model=ridge_fit(xs,ys,weights)
    # A held-out season provides empirical margin/total errors and a baseline comparison.
    season_counts={year:sum(int(r['season'])==year for r in rows) for year in {int(r['season']) for r in rows}}
    last_season=max((year for year,count in season_counts.items() if count>=200),default=max(season_counts))
    training=[r for r in rows if int(r['season'])<last_season]
    test=[r for r in rows if int(r['season'])==last_season]
    margins=[];totals=[];wins=[];market_wins=[];score_errors=[]
    if len(training)>=80 and test:
        cutoff=min(kickoff(r) for r in test)
        train_x=[features(r,s) for r in training for s in ('home','away')]
        train_y=[number(r[s+'_score']) for r in training for s in ('home','away')]
        train_w=[.5**((cutoff-kickoff(r)).days/365) for r in training for _ in range(2)]
        held=ridge_fit(train_x,train_y,train_w)
        for r in test:
            h=max(0,predict(held,features(r,'home')));a=max(0,predict(held,features(r,'away')))
            actual_h=number(r['home_score']);actual_a=number(r['away_score'])
            margins.append((actual_h-actual_a)-(h-a));totals.append((actual_h+actual_a)-(h+a));score_errors.extend([abs(actual_h-h),abs(actual_a-a)])
            if actual_h!=actual_a:wins.append((h>a)==(actual_h>actual_a))
            if pd.notna(r.get('spread_line')) and actual_h!=actual_a and number(r['spread_line'])!=0:market_wins.append((number(r['spread_line'])>0)==(actual_h>actual_a))
    model.update(version=MODEL_VERSION,trained_at=as_of.isoformat(),games=len(rows),margin_sd=max(10,float(np.std(margins))) if margins else 14.,total_sd=max(10,float(np.std(totals))) if totals else 15.,validation={'type':'held-out season; historical observed weather and starting QB, not archived pregame forecasts','season':last_season,'games':len(margins),'winner_accuracy':float(np.mean(wins)) if wins else None,'market_favorite_accuracy':float(np.mean(market_wins)) if market_wins else None,'score_mae':float(np.mean(score_errors)) if score_errors else None,'limitations':['Retrospective diagnostic, not the live pick record','Actual historical weather and starting QB can be more certain than pregame inputs','Roster replacement adjustments are excluded from this diagnostic','Market baseline availability can differ; compare live common games for a fair test']})
    return model


def game_prediction(model,row,weather=None,roster=None):
    roster=roster or {};points={};reasons=[];components={}
    for side in ('home','away'):
        team=row[side+'_team'];unit=roster.get(team,{})
        x=features(row,side,weather,unit.get('qb_gsis'))
        raw=predict(model,x)
        other=row['away_team' if side=='home' else 'home_team']
        offense_change=number(unit.get('offense_points_delta'))
        defense_change=number(roster.get(other,{}).get('defense_points_delta'))
        points[side]=max(0.,min(60.,raw+offense_change+defense_change))
        contributions={k:round(model['coefficients'].get(k,0)*v,3) for k,v in x.items() if k in model['coefficients']}
        components[side]={'baseline':model['intercept'],'factors':contributions,'inputs':x,'offensive_replacements':offense_change,'opposing_defensive_replacements':defense_change,'raw_points':raw}
        reasons.extend(f'{team}: {n}' for n in unit.get('notes',[]))
    margin=points['home']-points['away'];total=sum(points.values());sd=model['margin_sd']
    hp=float(norm.cdf((margin-.5)/sd));ap=float(norm.cdf((-margin-.5)/sd));tie=max(0,1-hp-ap)
    line=number(row.get('spread_line'),None);market_total=number(row.get('total_line'),None)
    # nflverse spread_line is the market-implied HOME winning margin (opposite home handicap).
    ats=None if line is None else ('home' if margin>line else 'away' if margin<line else 'pass')
    ou=None if market_total is None else ('over' if total>market_total else 'under' if total<market_total else 'pass')
    return {'model_version':MODEL_VERSION,'home_score':round(points['home'],2),'away_score':round(points['away'],2),'home_win_probability':hp,'away_win_probability':ap,'tie_probability':tie,'pick':row['home_team'] if hp>=ap else row['away_team'],'margin':margin,'total':total,'margin_p10':margin-1.28155*sd,'margin_p90':margin+1.28155*sd,'market_home_margin':line,'market_total':market_total,'market_home_moneyline':number(row.get('home_moneyline'),None),'market_away_moneyline':number(row.get('away_moneyline'),None),'ats_pick':ats,'total_pick':ou,'home_spread_odds':number(row.get('home_spread_odds'),None),'away_spread_odds':number(row.get('away_spread_odds'),None),'over_odds':number(row.get('over_odds'),None),'under_odds':number(row.get('under_odds'),None),'market_source':'nflverse schedule snapshot; not a guaranteed live or closing line','components':components,'weather':weather or {'status':'unavailable'},'reasons':reasons,'warning':'Experimental independent model; no demonstrated edge over the market'}
