"""Public research inputs with provenance, bounded refreshes and explicit missing coverage."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from app.data import Cache, BASE
from app.storage import DATA, now
from app.league_sync import TEAMS

SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'


def read_frame(path):
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, ValueError, OSError):
        return pd.DataFrame()


def json_source(cache, name, url, ttl=3600):
    path = cache.fetch(name + '.jsondata', url, ttl=ttl)
    return json.loads(path.read_text(encoding='utf-8'))


def refresh_inputs(season, cache_path=DATA/'cache'):
    cache = Cache(cache_path)
    specs = [('schedules.parquet', f'{BASE}/schedules/games.parquet', 900), ('players.parquet', f'{BASE}/players/players.parquet', 86400), (f'roster_{season}.parquet', f'{BASE}/rosters/roster_{season}.parquet', 21600)]
    for year in range(season-4, season+1):
        specs += [(f'stats_{year}.parquet', f'{BASE}/stats_player/stats_player_week_{year}.parquet', 1800 if year==season else 86400), (f'team_stats_{year}.parquet', f'{BASE}/stats_team/stats_team_week_{year}.parquet', 1800 if year==season else 86400), (f'snaps_{year}.parquet', f'{BASE}/snap_counts/snap_counts_{year}.parquet', 3600 if year==season else 86400)]
    for year in range(season-2, season+1):
        specs.append((f'depth_{year}.parquet', f'{BASE}/depth_charts/depth_charts_{year}.parquet', 21600))
    # Play-by-play adds red-zone usage, pace, neutral pass rate and expected-points metrics.
    for year in range(season-2, season+1):
        specs.append((f'pbp_{year}.parquet', f'{BASE}/pbp/play_by_play_{year}.parquet', 3600 if year==season else 86400))
    def download(spec):
        name, url, ttl = spec
        try:
            path = cache.fetch(name, url, ttl=ttl)
            metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text())
            return {'name': name, 'status': metadata.get('status'), 'fetched_at': metadata.get('fetched_at'), 'url': url, 'bytes': path.stat().st_size}
        except Exception as exc:
            return {'name': name, 'status': 'unavailable', 'error': str(exc), 'url': url}
    with ThreadPoolExecutor(max_workers=4) as pool:
        status = list(pool.map(download, specs))
    return status


def depth_charts(season, cache_path=DATA/'cache'):
    cache = Cache(cache_path)
    def one(item):
        team_id, team = item
        url = f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts'
        try:
            data = json_source(cache, 'depth-live-'+team, url, 1800)
            if data.get('season', {}).get('year') != season:
                raise ValueError('Depth chart season mismatch')
            units = []
            for chart in data.get('depthchart', []):
                for slot, position in chart.get('positions', {}).items():
                    athletes = position.get('athletes', [])
                    units.append({'slot': slot.upper(), 'position': position.get('position', {}).get('abbreviation', slot.upper()), 'players': [{'espn_id': str(p['id']), 'name': p.get('displayName', str(p['id']))} for p in athletes]})
            return team, {'units': units, 'source_url': url, 'reported_at': data.get('timestamp')}
        except Exception as exc:
            return team, {'units': [], 'error': str(exc), 'source_url': url}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(one, TEAMS.items()))


def scoreboard(season, week, cache_path=DATA/'cache'):
    data = json_source(Cache(cache_path), f'scoreboard-{season}-{week}', f'{SCOREBOARD}?dates={season}&seasontype=2&week={week}&limit=1000', 300)
    result = []
    for event in data.get('events', []):
        if event.get('season', {}).get('year') != season or event.get('season', {}).get('type') != 2:
            continue
        competition = event['competitions'][0]
        sides = {c['homeAway']: c for c in competition['competitors']}
        if not all(k in sides for k in ('home', 'away')):
            continue
        state = event.get('status', competition.get('status', {})).get('type', {})
        normalize = lambda t: {'WSH': 'WAS', 'LA': 'LAR'}.get(t, t)
        result.append({'espn_id': event['id'], 'season': season, 'week': int(event.get('week', {}).get('number', week)), 'kickoff': event['date'], 'home_team': normalize(sides['home']['team']['abbreviation']), 'away_team': normalize(sides['away']['team']['abbreviation']), 'home_score': float(sides['home'].get('score', 0)), 'away_score': float(sides['away'].get('score', 0)), 'home_name':sides['home']['team'].get('displayName'), 'away_name':sides['away']['team'].get('displayName'), 'completed': bool(state.get('completed')), 'remaining_fraction': max(0,min(1,((4-int(event.get('status',{}).get('period',0)))*900+float(event.get('status',{}).get('clock',0)))/3600)) if state.get('state')=='in' and 1<=int(event.get('status',{}).get('period',0))<=4 else None, 'state': state.get('state', 'pre'), 'venue': competition.get('venue', {}), 'source_url': f'https://www.espn.com/nfl/game/_/gameId/{event["id"]}'})
    return result


def weather_for(game, cache_path=DATA/'cache'):
    venue = game.get('venue', {})
    if venue.get('indoor'):
        return {'status': 'indoors', 'wind_mph': 0., 'temperature_f': 70., 'precipitation_mm': 0., 'source': 'ESPN venue roof', 'fetched_at': now()}
    try:
        kickoff = datetime.fromisoformat(game['kickoff'].replace('Z', '+00:00'))
        hours = (kickoff-datetime.now(timezone.utc)).total_seconds()/3600
        if not 0 <= hours <= 16*24:
            return {'status': 'outside forecast horizon'}
        city = venue.get('address', {}).get('city')
        if not city:
            return {'status': 'venue unavailable'}
        from urllib.parse import urlencode
        cache = Cache(cache_path)
        country = {'USA':'US','United States':'US','United Kingdom':'GB','Germany':'DE','Brazil':'BR','Spain':'ES','Mexico':'MX','Australia':'AU','Ireland':'IE'}.get(venue.get('address', {}).get('country'))
        query = {'name': city, 'count': 5, 'language': 'en', 'format': 'json'}
        if country:query['countryCode'] = country
        location = json_source(cache, 'venue-'+str(venue.get('id', city)), 'https://geocoding-api.open-meteo.com/v1/search?'+urlencode(query), 30*86400)
        places = [p for p in location.get('results', []) if not country or p.get('country_code')==country]
        state = venue.get('address', {}).get('state')
        # State is often an abbreviation; retain city/country match and expose approximate location.
        if not places:return {'status': 'venue coordinates unavailable'}
        place = places[0]
        params = {'latitude':place['latitude'],'longitude':place['longitude'],'hourly':'temperature_2m,precipitation,wind_speed_10m','temperature_unit':'fahrenheit','wind_speed_unit':'mph','forecast_days':16,'timezone':'UTC'}
        url = 'https://api.open-meteo.com/v1/forecast?'+urlencode(params)
        raw = json_source(cache, 'weather-'+str(venue.get('id', city)), url, 1800)
        hourly = raw['hourly'];index = min(range(len(hourly['time'])), key=lambda i:abs((datetime.fromisoformat(hourly['time'][i]).replace(tzinfo=timezone.utc)-kickoff).total_seconds()))
        distance = abs((datetime.fromisoformat(hourly['time'][index]).replace(tzinfo=timezone.utc)-kickoff).total_seconds())
        if distance > 3600:return {'status':'forecast does not cover kickoff'}
        meta_path=cache.path/('weather-'+str(venue.get('id',city))+'.jsondata.json')
        metadata=json.loads(meta_path.read_text())
        if metadata.get('status')!='ok':return {'status':'stale forecast; not applied','fetched_at':metadata.get('fetched_at')}
        return {'status':'forecast','temperature_f':hourly['temperature_2m'][index],'wind_mph':hourly['wind_speed_10m'][index],'precipitation_mm':hourly['precipitation'][index],'forecast_time':hourly['time'][index],'source_url':url,'location':place['name'],'approximate_location':True,'fetched_at':metadata['fetched_at']}
    except Exception as exc:
        return {'status':'unavailable','error':str(exc)}
