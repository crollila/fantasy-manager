"""Pregame expectations and postgame observations with explicit missing coverage.

Public statistics are revised by their publishers. Only the expectations archived
before kickoff are scored; today's inputs never reconstruct yesterday's prediction.
"""
from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.game_model import features, kickoff, number, predict, ridge_fit
from app.research_sources import read_frame

# label, display unit, minimum historical scatter, valid range
METRICS = {
    'net_passing_yards': ('Net passing yards', 'yards', 45., -100., 800.),
    'rushing_yards': ('Rushing yards', 'yards', 30., -50., 600.),
    'plays': ('Offensive plays', 'plays', 7., 0., 150.),
    'turnovers': ('Turnovers lost', 'turnovers', .8, 0., 15.),
    'sacks_suffered': ('Sacks allowed', 'sacks', 1., 0., 20.),
    'epa_per_play': ('EPA per play', 'EPA/play', .12, -3., 3.),
    'success_rate': ('Successful plays', '%', .06, 0., 1.),
    'neutral_pass_rate': ('Pass rate in close situations', '%', .08, 0., 1.),
    'explosive_rate': ('Explosive plays (20+ yards)', '%', .03, 0., 1.),
    'red_zone_td_rate': ('Red-zone drives ending in a TD', '%', .15, 0., 1.),
    'ngs_cpoe': ('NGS completion rate above expectation', 'pp', 5., -100., 100.),
    'ngs_ryoe': ('NGS rushing yards over expectation', 'yards/carry', .6, -15., 30.),
    'ngs_separation': ('NGS receiver separation', 'yards', .3, 0., 15.),
    'ngs_yac_oe': ('NGS YAC above expectation', 'yards/catch', 1., -15., 30.),
}
NGS = {
    'passing': ('attempts', {'completion_percentage_above_expectation': 'ngs_cpoe'}),
    'rushing': ('rush_attempts', {'rush_yards_over_expected_per_att': 'ngs_ryoe'}),
    'receiving': ('targets', {'avg_separation': 'ngs_separation', 'avg_yac_above_expectation': 'ngs_yac_oe'}),
}


def normalize(team):
    return {'WSH': 'WAS', 'LA': 'LAR', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LAR'}.get(str(team), str(team))


def finite(value):
    return number(value, None)


def provenance(path):
    try:
        meta = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf-8'))
        return {k: meta[k] for k in ('url', 'fetched_at', 'sha256', 'status') if k in meta}
    except (OSError, ValueError):
        return {'file': path.name, 'status': 'local cache; retrieval metadata unavailable'}


def observations(cache_path, schedule, season):
    """One record per game/team. No missing-to-zero coercion, no week-zero NGS."""
    cache_path = Path(cache_path)
    games = {r['game_id']: r for r in schedule.to_dict('records') if r.get('game_type') == 'REG'}
    by_week = {(int(r['season']), int(r['week']), normalize(r[s+'_team'])): gid
               for gid, r in games.items() for s in ('home', 'away')}
    output = {}

    def record(gid, team):
        return output.setdefault((gid, normalize(team)), {'metrics': {}, 'sources': {}, 'players': {}})

    for year in range(season - 4, season + 1):
        path = cache_path / f'team_stats_{year}.parquet'
        source = provenance(path)
        for r in read_frame(path).to_dict('records'):
            if r.get('game_id') not in games: continue
            item = record(r['game_id'], r['team'])
            values = {k: finite(r.get(k)) for k in ('passing_yards', 'sack_yards_lost', 'rushing_yards', 'attempts', 'carries', 'sacks_suffered', 'passing_interceptions', 'fumbles_lost_total')}
            metrics = {k: values[k] for k in ('rushing_yards', 'sacks_suffered')}
            for key, cols, signs in [('net_passing_yards', ['passing_yards', 'sack_yards_lost'], [1, -1]), ('plays', ['attempts', 'carries', 'sacks_suffered'], [1, 1, 1]), ('turnovers', ['passing_interceptions', 'fumbles_lost_total'], [1, 1])]:
                if all(values[c] is not None for c in cols): metrics[key] = sum(values[c]*s for c, s in zip(cols, signs))
            for key, value in metrics.items():
                if value is not None: item['metrics'][key] = value; item['sources'][key] = source

        pbp_path = cache_path / f'pbp_{year}.parquet'
        pbp = read_frame(pbp_path)
        if pbp.empty or not {'game_id', 'posteam', 'play_type', 'epa'} <= set(pbp): continue
        # Kneels/spikes are excluded from efficiency and tendency statistics.
        pbp = pbp[pbp.game_id.isin(games) & pbp.play_type.isin(['pass', 'run'])].copy()
        for col in ('qb_kneel', 'qb_spike'):
            if col in pbp: pbp = pbp[pbp[col].fillna(0) != 1]
        source = provenance(pbp_path)
        for (gid, team), part in pbp.groupby(['game_id', 'posteam']):
            if len(part) < 20: continue # Never interpret a partial feed as a full box score.
            item = record(gid, team); metrics = {}
            valid = pd.to_numeric(part.epa, errors='coerce').dropna()
            if len(valid) >= 20:
                metrics['epa_per_play'] = float(valid.mean()); metrics['success_rate'] = float((valid > 0).mean())
            if 'yards_gained' in part:
                valid_yards = pd.to_numeric(part.yards_gained, errors='coerce').dropna()
                if len(valid_yards) >= 20: metrics['explosive_rate'] = float((valid_yards >= 20).mean())
            if {'score_differential', 'game_seconds_remaining'} <= set(part):
                neutral = part[(part.score_differential.abs() <= 7) & (part.game_seconds_remaining > 120)]
                if len(neutral) >= 10: metrics['neutral_pass_rate'] = float((neutral.play_type == 'pass').mean())
            if {'fixed_drive', 'yardline_100', 'pass_touchdown', 'rush_touchdown'} <= set(part):
                rz = part.groupby('fixed_drive').filter(lambda x: x.yardline_100.min() <= 20)
                if not rz.empty:
                    rz=rz.assign(offensive_td=rz[['pass_touchdown','rush_touchdown']].max(axis=1))
                    metrics['red_zone_td_rate'] = float(rz.groupby('fixed_drive').offensive_td.max().mean())
            for key, value in metrics.items():
                if finite(value) is not None: item['metrics'][key] = value; item['sources'][key] = source

    for kind, (weight_col, fields) in NGS.items():
        path = cache_path / f'ngs_{kind}.parquet'; frame = read_frame(path)
        if frame.empty or not {'season', 'season_type', 'week', 'team_abbr', weight_col} <= set(frame): continue
        frame = frame[(frame.season_type == 'REG') & (frame.week > 0) & (frame.season >= season-4) & (frame.season <= season)]
        source = provenance(path) | {'coverage': 'Players meeting NGS opportunity thresholds only; not a complete team census'}
        for (year, week, team), part in frame.groupby(['season', 'week', 'team_abbr']):
            gid = by_week.get((int(year), int(week), normalize(team)))
            if not gid: continue
            item = record(gid, team)
            for column, key in fields.items():
                if column not in part: continue
                # YAC averages have receptions as their denominator, not targets.
                denominator = 'receptions' if key == 'ngs_yac_oe' else weight_col
                if denominator not in part: continue
                valid = part[[column, denominator]].apply(pd.to_numeric, errors='coerce').dropna()
                valid = valid[valid[denominator] > 0]
                if not valid.empty:
                    item['metrics'][key] = float(np.average(valid[column], weights=valid[denominator]))
                    item['sources'][key] = source

    # A stat row/snaps can establish participation; an absent row cannot establish a DNP.
    from app.roster_context import id_maps
    _, espn = id_maps(cache_path)
    pfr_to_espn = {v.get('pfr'): eid for eid, v in espn.items() if v.get('pfr')}
    gsis_to_espn = {v.get('gsis'): eid for eid, v in espn.items() if v.get('gsis')}
    for year in range(season - 4, season + 1):
        for r in read_frame(cache_path / f'stats_{year}.parquet').to_dict('records'):
            eid = gsis_to_espn.get(str(r.get('player_id')))
            if not eid or r.get('game_id') not in games or not r.get('team'): continue
            # Actual recorded opportunities/defensive stats, not just membership in the file.
            used = [finite(r.get(k)) for k in ('attempts', 'carries', 'targets', 'def_tackles_solo', 'def_tackles_with_assist')]
            if any(v is not None and v > 0 for v in used):
                record(r['game_id'], r['team'])['players'][eid] = {'played': True, 'source': 'nflverse player statistics'}
        for r in read_frame(cache_path / f'snaps_{year}.parquet').to_dict('records'):
            eid = pfr_to_espn.get(str(r.get('pfr_player_id')))
            if not eid or r.get('game_id') not in games: continue
            counts = [finite(r.get(k)) for k in ('offense_snaps', 'defense_snaps', 'st_snaps')]
            if all(v is not None for v in counts):
                record(r['game_id'], r['team'])['players'][eid] = {'played': sum(counts) > 0, 'snaps': sum(counts), 'source': 'nflverse/PFR snap counts'}

    for (gid, _), item in output.items():
        row = games[gid]; roof = str(row.get('roof', '')).lower()
        item['weather'] = {k: v for k, col in [('temperature_f', 'temp'), ('wind_mph', 'wind')] if (v := finite(row.get(col))) is not None}
        if roof in ('dome', 'closed'): item['weather'] = {'indoors': True}
        item['weather']['source'] = 'nflverse historical game conditions; observations, not an archived weather forecast'
    return output


def fit_expectations(actuals, schedule, as_of):
    """Joint offense/opponent regression avoids rewarding a soft opponent schedule."""
    rows = {r['game_id']: r for r in schedule.to_dict('records') if r.get('game_type') == 'REG'
            and kickoff(r) and kickoff(r) + timedelta(hours=8) < as_of
            and finite(r.get('home_score')) is not None and finite(r.get('away_score')) is not None}
    models = {}
    for key, (_, _, floor, _, _) in METRICS.items():
        xs, ys, weights, ids = [], [], [], []
        for (gid, team), item in actuals.items():
            row = rows.get(gid); value = finite(item['metrics'].get(key))
            if row is None or value is None or (as_of-kickoff(row)).days > 3*366: continue
            side = 'home' if normalize(row['home_team']) == team else 'away'
            if normalize(row[side+'_team']) != team: continue
            xs.append(features(row, side)); ys.append(value); ids.append(gid)
            weights.append(.5**((as_of-kickoff(row)).days/180))
        if len(ys) < 160: continue
        model = ridge_fit(xs, ys, weights, alpha=25)
        residuals = np.array(ys) - np.array([predict(model, x) for x in xs])
        model.update(scatter=max(floor, float(np.sqrt(np.average(residuals**2, weights=weights)))), training_games=len(set(ids)))
        models[key] = model
    return {'version': 'process-context-1', 'models': models, 'as_of': as_of.isoformat(),
            'method': 'Recency-weighted joint team offense/opponent defense regression; 180-day half-life',
            'range_note': 'Reference bands use historical fitted residual scatter. They are not calibrated probability intervals.'}


def expectations(model, row, weather, roster):
    result = {}
    for side in ('home', 'away'):
        x = features(row, side, weather, roster.get(row[side+'_team'], {}).get('qb_gsis'))
        result[side] = {}
        for key, fitted in model.get('models', {}).items():
            label, unit, _, low, high = METRICS[key]
            mean = float(np.clip(predict(fitted, x), low, high)); scatter = fitted['scatter']
            result[side][key] = {'label': label, 'unit': unit, 'expected': mean, 'scatter': scatter,
                                 'low': max(low, mean-1.28155*scatter), 'high': min(high, mean+1.28155*scatter),
                                 'training_games': fitted['training_games']}
    return result


def review_game(forecast, game, actuals):
    right, missed, unknown, comparisons, players, conditions = [], [], [], [], [], []
    h, a = game['home_score'], game['away_score']
    winner = game['home_team'] if h > a else game['away_team'] if a > h else 'TIE'
    (right if winner == forecast['pick'] else missed).append(
        f"Winner: picked {forecast['pick']}; final winner {winner}.")
    deviations = []
    for side, score in [('home', h), ('away', a)]:
        team = game[side+'_team']; error = score-forecast[side+'_score']
        text = f'{team} scored {score:g} versus {forecast[side+"_score"]:.1f} projected ({error:+.1f}).'
        deviations.append(text)
        (right if abs(error) <= 3 else missed).append(text)
        observed = actuals.get((game['game_id'], normalize(team)), {})
        for key, expected in forecast.get('process_expectations', {}).get(side, {}).items():
            value = finite(observed.get('metrics', {}).get(key))
            item = expected | {'key': key, 'team': team, 'actual': value, 'source': observed.get('sources', {}).get(key)}
            if value is None: item.update(status='awaiting_data', error=None, standardized_error=None)
            else:
                z = (value-expected['expected'])/max(.001, expected['scatter'])
                item.update(status='within_range' if expected['low'] <= value <= expected['high'] else 'surprise', error=value-expected['expected'], standardized_error=z)
            comparisons.append(item)
        for p in forecast.get('personnel_snapshot', {}).get(side, []):
            actual = observed.get('players', {}).get(p['espn_id'], {})
            players.append(p | {'team': team, 'observed': actual.get('played'), 'snaps': actual.get('snaps'),
                                'source': actual.get('source'), 'matched': None if 'played' not in actual else actual['played'] == p['likely']})
        if side == 'home':
            for key, label, tolerance in [('temperature_f', 'Kickoff temperature', 8), ('wind_mph', 'Kickoff wind', 5)]:
                expected = finite(forecast.get('weather', {}).get(key)); actual = finite(observed.get('weather', {}).get(key))
                if expected is not None:
                    conditions.append({'label': label, 'expected': expected, 'actual': actual,
                                       'matched': None if actual is None else abs(actual-expected) <= tolerance,
                                       'source': observed.get('weather', {}).get('source')})
    if not forecast.get('process_expectations'):
        unknown.append('This older forecast did not save statistical expectations. They cannot be reconstructed using postgame information.')
    missing = sum(c['status'] == 'awaiting_data' for c in comparisons)
    if missing: unknown.append(f'{missing} expected statistics are waiting for source data. Reviews refresh when statistics arrive or are corrected.')
    surprises = sorted((c for c in comparisons if c['status'] == 'surprise'), key=lambda c: -abs(c['standardized_error']))
    within = [c for c in comparisons if c['status'] == 'within_range']
    for c in surprises[:4]: missed.append(f"{c['team']}: {c['label']} fell outside the saved reference range.")
    if within: right.append(f'{len(within)} of {len(comparisons)-missing} available statistical outcomes fell within their saved reference ranges.')
    discrepancies = [p for p in players if p['matched'] is False]
    for p in discrepancies[:4]: missed.append(f"{p['team']}: {p['name']} was {'expected to play' if p['likely'] else 'expected to miss the game'}, but recorded participation differed.")
    if any(p['observed'] is None for p in players): unknown.append('Missing player statistics are unknown participation, not an assumed absence.')
    return {'version': 1, 'observed_differences': deviations, 'pre_game_factors': forecast.get('reasons', []),
            'right': right, 'missed': missed, 'unknown': unknown, 'comparisons': comparisons, 'personnel': players, 'conditions': conditions,
            'score_errors': {side: game[side+'_score']-forecast[side+'_score'] for side in ('home', 'away')},
            'interpretation': 'These are measured differences and saved assumptions, not proof of causation. A surprise can be ordinary game variance. Reference bands are descriptive, not calibrated probabilities.',
            'learning_note': 'Final results update team history. New correction models train only on archived pregame inputs, wait for stat corrections, then face future games in shadow testing before promotion.'}
