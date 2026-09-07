"""Fresh, attributed injury evidence. Unknown does not mean healthy."""
import json
import re
from datetime import datetime,timezone
from app.data import Cache

URL='https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'

def normalize_injuries(payload):
    if payload.get('status')!='success' or not isinstance(payload.get('injuries'),list):raise ValueError('Invalid ESPN injury response')
    result={}
    for team in payload['injuries']:
        for item in team.get('injuries',[]):
            athlete=item.get('athlete',{});eid=str(athlete.get('id') or '')
            urls=[l.get('href','') for l in athlete.get('links',[]) if l.get('href','').startswith('https://www.espn.com/')]
            if not eid:
                match=next((re.search(r'/id/(\d+)',u) for u in urls if re.search(r'/id/(\d+)',u)),None)
                eid=match.group(1) if match else ''
            if not eid:continue
            row={'espn_id':eid,'status':str(item.get('status','UNKNOWN')).upper().replace(' ','_'),'reported_at':item.get('date'),'description':item.get('shortComment',''),'details':item.get('details',{}),'source_url':urls[0] if urls else 'https://www.espn.com/nfl/injuries'}
            if eid not in result or (row['reported_at'] or '')>(result[eid]['reported_at'] or ''):result[eid]=row
    return result

def injury_report(cache=None,force=False):
    cache=cache or Cache()
    try:
        path=cache.fetch('espn-injuries.jsondata',URL,ttl=900,force=force)
        raw=json.loads(path.read_text(encoding='utf-8'));rows=normalize_injuries(raw)
        metadata=json.loads(path.with_suffix(path.suffix+'.json').read_text())
        stamp=datetime.fromisoformat(metadata['fetched_at'].replace('Z','+00:00'))
        age=(datetime.now(timezone.utc)-stamp).total_seconds()/3600
        return {'status':'fresh' if age<=6 and metadata.get('status')=='ok' else 'stale','fetched_at':metadata['fetched_at'],'source_time':raw.get('timestamp'),'season':raw.get('season',{}).get('year'),'age_hours':age,'players':rows,'source_url':URL}
    except Exception as exc:return {'status':'unavailable','players':{},'error':str(exc),'source_url':URL}

def availability(status):
    status=status.upper().replace(' ','_')
    if status in ('OUT','INACTIVE','INJURED_RESERVE','INJURY_RESERVE','IR','SUSPENSION','SUSPENDED','PUP','RESERVE/PUP','DOUBTFUL'):return 0 if status!='DOUBTFUL' else .15
    if status in ('QUESTIONABLE','Q'):return .7
    if status in ('ACTIVE','HEALTHY','NORMAL'):return 1.
    return .9
