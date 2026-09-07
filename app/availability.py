"""Report evidence describes participation, never an automatic fantasy-point haircut."""
import re
from datetime import datetime, timezone
from app.injuries import availability


def participation(status, report, as_of=None):
    as_of = as_of or datetime.now(timezone.utc)
    status = status.upper().replace(' ', '_')
    probability = availability(status)
    evidence = []
    text = ' '.join(str(report.get(k) or '') for k in ('description', 'shortComment'))
    details = report.get('details', {})
    if isinstance(details, dict):
        text += ' ' + str(details.get('detail', '')) + ' ' + str(details.get('returnDate', ''))
    stamp = report.get('reported_at')
    fresh = False
    if stamp:
        try:
            age = (as_of - datetime.fromisoformat(stamp.replace('Z', '+00:00'))).total_seconds() / 86400
            fresh = 0 <= age <= 7
        except (ValueError, TypeError):
            pass
    # Official unavailability always overrides optimistic prose.
    if probability == 0:
        return {'probability': 0., 'likely': False, 'evidence': ['Official unavailable designation'], 'method': 'official designation', 'calibrated': False}
    text=re.sub(r"\b(?:not(?: yet| been)?|hasn.t been) ruled out\b",'status unresolved',text,flags=re.I)
    rules = [
        (r'\b(?:ruled out|will not play|won.t play|not expected to play|not cleared to play)\b', .05, 'Report expects absence'),
        (r'\b(?:unlikely to play|doubtful to play)\b', .15, 'Report says unlikely to play'),
        (r'\b(?:game.time decision|true game.time call)\b', .5, 'Game-time decision'),
        (r'\b(?:expected to play|expects to play|will play|cleared to play|good to go)\b', .9, 'Report expects participation'),
        (r'\b(?:full participant|practiced fully|full practice)\b', .9, 'Full practice reported'),
        (r'\b(?:limited participant|limited practice|limited in practice)\b', .7, 'Limited practice reported'),
        (r'\b(?:did not practice|missed practice|didn.t practice)\b', .55, 'Missed practice reported'),
    ]
    if fresh:
        matches = [(p, label, m.group()) for pattern, p, label in rules for m in [re.search(pattern, text, re.I)] if m]
        if matches:
            # Negative/contradictory evidence wins; do not match "expected to play" inside "not expected to play" optimistically.
            probability = min(p for p, _, _ in matches)
            evidence = [f'{label}: “{quote}”' for _, label, quote in matches]
    else:
        evidence.append('No recent dated report; designation prior only')
    return {'probability': probability, 'likely': probability >= .5, 'evidence': evidence, 'method': 'transparent report-language rules and designation priors', 'calibrated': False, 'source_url': report.get('source_url'), 'reported_at': stamp, 'limitation': 'Heuristic participation estimate, not a trained sentiment model or medical prognosis'}
