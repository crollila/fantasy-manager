"""Auditable, prospective score correction. A candidate never grades its own training games.

The foundation refits public history on refresh. This separate correction layer is
frozen, scored in shadow on future archived predictions, and promoted once per test.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np
from scipy.stats import norm
from app.game_model import number, predict, ridge_fit

FAMILY = 'game-evidence-1'
MIN_TRAIN = 80
MIN_TEST = 32
MIN_WEEKS = 3
CORRECTION_LAG_DAYS = 4
MAX_CORRECTION = 3.
CENTERS = {'net_passing_yards': (220, 80), 'rushing_yards': (115, 50), 'plays': (64, 10),
           'turnovers': (1.3, 1), 'sacks_suffered': (2.5, 2), 'epa_per_play': (0, .2),
           'success_rate': (.45, .1), 'neutral_pass_rate': (.55, .15), 'explosive_rate': (.07, .04),
           'red_zone_td_rate': (.55, .2), 'ngs_cpoe': (0, 5), 'ngs_ryoe': (0, 1),
           'ngs_separation': (3, .5), 'ngs_yac_oe': (0, 1)}


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def initialize(store):
    from app.tracking import initialize as init_tracking
    init_tracking(store)
    with store.connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS game_model_artifacts(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,kind TEXT NOT NULL,body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS game_learning_events(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,body TEXT NOT NULL);
        ''')


def archive_artifact(store, kind, body, as_of):
    initialize(store); artifact_id = digest(body)
    with store.connect() as c:
        c.execute('INSERT OR IGNORE INTO game_model_artifacts VALUES(?,?,?,?)',
                  (artifact_id, as_of.isoformat(), kind, json.dumps(body, allow_nan=False)))
    return artifact_id


def artifact(store, artifact_id):
    if not artifact_id: return None
    with store.connect() as c:
        row = c.execute('SELECT body FROM game_model_artifacts WHERE id=?', (artifact_id,)).fetchone()
    return json.loads(row[0]) if row else None


def initial_state():
    return {'family': FAMILY, 'active': None, 'pending': None, 'last_training_count': 0,
            'eligible_games': 0, 'validation_games': 0, 'validation_weeks': 0,
            'phase': 'collecting', 'monitor_through': None, 'last_evaluation': None}


def state(store):
    initialize(store)
    with store.connect() as c:
        row = c.execute("SELECT value FROM meta WHERE key='game-learning'").fetchone()
    value = json.loads(row[0]) if row else initial_state()
    return value if value.get('family') == FAMILY else initial_state()


def add_event(store, kind, as_of, **details):
    value = {'kind': kind, **details}
    with store.connect() as c:
        c.execute('INSERT INTO game_learning_events(created_at,body) VALUES(?,?)', (as_of.isoformat(), json.dumps(value, allow_nan=False)))


def correction_features(forecast, side):
    other = 'away' if side == 'home' else 'home'
    component = forecast.get('components', {}).get(side, {})
    x = {k: float(v) for k, v in component.get('inputs', {}).items()
         if k in ('home', 'rest', 'wind', 'cold', 'heat', 'wet') or k.startswith(('offense:', 'defense:'))}
    x['base_points'] = (forecast[side+'_score']-23)/10
    x['offensive_replacements'] = component.get('offensive_replacements', 0)/3
    x['opposing_defensive_replacements'] = component.get('opposing_defensive_replacements', 0)/3
    for prefix, which in [('own', side), ('opponent', other)]:
        for key, (center, scale) in CENTERS.items():
            mean = number(forecast.get('process_expectations', {}).get(which, {}).get(key, {}).get('expected'), None)
            if mean is not None: x[prefix+':'+key] = float(np.clip((mean-center)/scale, -4, 4))
            else: x['missing:'+prefix+':'+key] = 1.
    return x


def scores(home, away, sd, forecast):
    h, a = round(float(np.clip(home, 0, 60)), 2), round(float(np.clip(away, 0, 60)), 2)
    margin, total = h-a, h+a
    hp = float(norm.cdf((margin-.5)/sd)); ap = float(norm.cdf((-margin-.5)/sd))
    line, market_total = forecast.get('market_home_margin'), forecast.get('market_total')
    return {'home_score': h, 'away_score': a, 'margin': margin, 'total': total,
            'home_win_probability': hp, 'away_win_probability': ap, 'tie_probability': max(0, 1-hp-ap),
            'pick': forecast['home_team'] if hp >= ap else forecast['away_team'], 'pick_win_probability': max(hp, ap),
            'margin_p10': margin-1.28155*sd, 'margin_p90': margin+1.28155*sd,
            'ats_pick': None if line is None else 'home' if margin > line else 'away' if margin < line else 'pass',
            'total_pick': None if market_total is None else 'over' if total > market_total else 'under' if total < market_total else 'pass'}


def corrected(model, base, xs, forecast):
    delta = {s: float(np.clip(predict(model, xs[s]), -MAX_CORRECTION, MAX_CORRECTION)) for s in ('home', 'away')}
    return scores(base['home_score']+delta['home'], base['away_score']+delta['away'], base['margin_sd'], forecast), delta


def apply_learning(forecast, learning, active=None, pending=None):
    """Archive both champion and shadow scores using exactly the same pregame inputs."""
    base = {k: forecast[k] for k in ('home_score', 'away_score')}
    base['margin_sd'] = forecast.get('margin_sd', 14.)
    xs = {s: correction_features(forecast, s) for s in ('home', 'away')}
    saved = {'family': FAMILY, 'features': xs, 'base': base, 'active_artifact': learning.get('active'),
             'candidate': None, 'adjustments': {'home': 0., 'away': 0.}}
    if active and active.get('family') == FAMILY:
        updated, delta = corrected(active, base, xs, forecast)
        forecast.update(updated); saved['adjustments'] = delta
        forecast['model_version'] += '+learn-'+learning['active'][:8]
    if pending and pending.get('family') == FAMILY:
        candidate, _ = corrected(pending, base, xs, forecast)
        saved['candidate'] = {'artifact': learning['pending'], 'scores': candidate}
    forecast['learning'] = saved
    return forecast


def mature_examples(store, as_of):
    from app.tracking import saved_games
    forecasts = saved_games(store)
    with store.connect() as c:
        results = {r['game_id']: json.loads(r['body']) for r in c.execute('SELECT * FROM game_results')}
    periods = defaultdict(list)
    for f in forecasts.values(): periods[(f['season'], f['week'])].append(f)
    eligible = []
    for period, rows in sorted(periods.items()):
        # Keep the complete archived NFL week together and allow source corrections.
        if any(f['game_id'] not in results for f in rows): continue
        if max(stamp(f['kickoff']) for f in rows) + timedelta(days=CORRECTION_LAG_DAYS) > as_of: continue
        for f in rows:
            if f.get('learning', {}).get('family') != FAMILY: continue
            if stamp(f['saved_at']) >= stamp(f['kickoff']): continue
            r = results[f['game_id']]
            if r['forecast_id'] != f['forecast_id']: continue
            eligible.append({'forecast': f, 'result': r, 'period': period})
    return sorted(eligible, key=lambda e: (e['period'], e['forecast']['kickoff'], e['forecast']['game_id']))


def fit_candidate(examples, as_of):
    xs, ys = [], []
    for e in examples:
        f, r = e['forecast'], e['result']; saved = f['learning']
        for s in ('home', 'away'):
            xs.append(saved['features'][s]); ys.append(r[s+'_score']-saved['base'][s+'_score'])
    model = ridge_fit(xs, ys, alpha=40)
    model.update(family=FAMILY, created_at=as_of.isoformat(), training_games=len(examples),
                 training_ids=[e['forecast']['forecast_id'] for e in examples],
                 training_fingerprint=digest([{'id': e['forecast']['forecast_id'], 'h': e['result']['home_score'], 'a': e['result']['away_score']} for e in examples]),
                 max_adjustment=MAX_CORRECTION, method='Ridge correction of saved foundation score residuals; alpha=40; bounded to 3 points per team')
    return model


def losses(prediction, result):
    h, a = result['home_score'], result['away_score']
    mae = (abs(h-prediction['home_score'])+abs(a-prediction['away_score']))/2
    probs = [prediction['home_win_probability'], prediction['away_win_probability'], prediction.get('tie_probability', 0.)]
    brier = sum((p-y)**2 for p, y in zip(probs, [int(h>a), int(a>h), int(h==a)]))
    return mae, brier


def test_batch(examples):
    groups = defaultdict(list)
    for e in examples: groups[e['period']].append(e)
    result = []
    for i, (_, rows) in enumerate(sorted(groups.items())):
        result.extend(rows)
        if len(result) >= MIN_TEST and i+1 >= MIN_WEEKS: return result
    return []


def compare(examples, challenger):
    """Paired same-game comparisons; resampling keeps both teams and each week together."""
    pairs, groups = [], defaultdict(list)
    for e in examples:
        f, r = e['forecast'], e['result']; alt = challenger(e)
        champion_loss, champion_brier = losses(f, r); candidate_loss, candidate_brier = losses(alt, r)
        pairs.append((champion_loss, candidate_loss, champion_brier, candidate_brier))
        groups[e['period']].append(champion_loss-candidate_loss)
    array = np.array(pairs); blocks = list(groups.values()); rng = np.random.default_rng(74051)
    bootstrap = [float(np.mean([v for i in rng.integers(0, len(blocks), len(blocks)) for v in blocks[i]])) for _ in range(2000)]
    lower, upper = np.quantile(bootstrap, [.025, .975])
    gain = float(np.mean(array[:, 0]-array[:, 1]))
    brier_delta = float(np.mean(array[:, 3]-array[:, 2]))
    return {'games': len(examples), 'weeks': len(blocks), 'champion_mae': float(array[:, 0].mean()),
            'candidate_mae': float(array[:, 1].mean()), 'gain': gain, 'gain_ci95': [float(lower), float(upper)],
            'champion_brier': float(array[:, 2].mean()), 'candidate_brier': float(array[:, 3].mean()),
            'passed': gain >= .15 and lower > 0 and brier_delta <= .01,
            'forecast_ids': [e['forecast']['forecast_id'] for e in examples],
            'first_kickoff': min(e['forecast']['kickoff'] for e in examples), 'last_kickoff': max(e['forecast']['kickoff'] for e in examples),
            'rule': 'At least 32 future games across 3 closed NFL weeks; score MAE improves by 0.15+, week-bootstrap 95% lower bound > 0, and Brier worsens by no more than 0.01. One evaluation per candidate.'}


def update_learning(store, as_of=None):
    as_of = as_of or datetime.now(timezone.utc)
    current = state(store); examples = mature_examples(store, as_of)
    current['eligible_games'] = len(examples)
    pending = artifact(store, current['pending'])
    if pending:
        future = [e for e in examples if (e['forecast'].get('learning', {}).get('candidate') or {}).get('artifact') == current['pending']
                  and e['forecast']['forecast_id'] not in pending['training_ids']
                  and stamp(e['forecast']['saved_at']) >= stamp(pending['created_at'])
                  and stamp(e['forecast']['kickoff']) > stamp(pending['created_at'])]
        current['validation_games'] = len(future); current['validation_weeks'] = len({e['period'] for e in future})
        batch = test_batch(future)
        if batch:
            comparison = compare(batch, lambda e: e['forecast']['learning']['candidate']['scores'])
            kind = 'promoted' if comparison['passed'] else 'rejected'
            add_event(store, kind, as_of, artifact=current['pending'], previous=current['active'], comparison=comparison)
            if comparison['passed']:
                current['active'] = current['pending']; current['promoted_at'] = as_of.isoformat(); current['monitor_through'] = None
            current['last_evaluation'] = comparison | {'decision': kind, 'evaluated_at': as_of.isoformat()}
            current['pending'] = None
            # The tested candidate is retired. Never re-test it after seeing more outcomes.
            current['last_training_count'] = len(examples)
    # Fixed batches monitor a promoted model against the saved unadjusted foundation.
    if current['active']:
        monitored = [e for e in examples if e['forecast']['learning'].get('active_artifact') == current['active']
                     and (not current['monitor_through'] or e['forecast']['kickoff'] > current['monitor_through'])]
        batch = test_batch(monitored)
        if batch:
            baseline = lambda e: scores(e['forecast']['learning']['base']['home_score'], e['forecast']['learning']['base']['away_score'], e['forecast']['learning']['base']['margin_sd'], e['forecast'])
            comparison = compare(batch, baseline)
            current['monitor_through'] = max(e['forecast']['kickoff'] for e in batch)
            add_event(store, 'rolled_back' if comparison['passed'] else 'monitor_pass', as_of, artifact=current['active'], comparison=comparison)
            if comparison['passed']:
                current['active'] = None
                current['pending'] = None # A shadow trained against a retired champion gets a fresh test.
                current['last_training_count'] = len(examples)
                current['last_evaluation'] = comparison | {'decision': 'rolled_back', 'evaluated_at': as_of.isoformat()}
    if not current['pending'] and len(examples) >= MIN_TRAIN and len(examples)-current['last_training_count'] >= MIN_TEST:
        candidate = fit_candidate(examples, as_of)
        current['pending'] = archive_artifact(store, 'score-correction', candidate, as_of)
        current['last_training_count'] = len(examples); current['validation_games'] = 0; current['validation_weeks'] = 0
        add_event(store, 'candidate_started', as_of, artifact=current['pending'], training_games=len(examples))
    current['phase'] = 'shadow_testing' if current['pending'] else 'active' if current['active'] else 'collecting'
    current['updated_at'] = as_of.isoformat()
    with store.connect() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES('game-learning',?)", (json.dumps(current, allow_nan=False),))
    return current


def dashboard(store):
    current = state(store)
    with store.connect() as c:
        events = [json.loads(r['body']) | {'created_at': r['created_at']} for r in c.execute('SELECT * FROM game_learning_events ORDER BY id DESC LIMIT 40')]
    return current | {'events': events, 'minimum_training_games': MIN_TRAIN, 'minimum_test_games': MIN_TEST,
                      'next_training_target': max(MIN_TRAIN,current['last_training_count']+MIN_TEST),
                      'minimum_test_weeks': MIN_WEEKS, 'correction_lag_days': CORRECTION_LAG_DAYS, 'max_adjustment': MAX_CORRECTION,
                      'method': 'The foundation refits completed game history on refresh. A separate correction model learns from archived score misses and saved weather, personnel and statistical expectations. It shadows future games before it can affect published picks.',
                      'limits': 'Promotion is evidence on a finite sample, not a guarantee. Repeated experiments can still overfit. Historical diagnostics and postgame explanations are never counted as live predictions.'}


def probability(value):
    odds = number(value, None)
    if odds is None or -100 < odds < 100: return None
    return -odds/(100-odds) if odds < 0 else 100/(100+odds)


def diagnostics(forecasts, results):
    matched = [(forecasts[r['game_id']], r) for r in results if r['game_id'] in forecasts]
    bins, conditions, market = defaultdict(list), defaultdict(list), []
    for f, r in matched:
        h, a = r['home_score'], r['away_score']
        if h != a:
            p = f['home_win_probability']/max(.0001, f['home_win_probability']+f['away_win_probability'])
            bins[min(4, int(p*5))].append((p, int(h>a)))
            implied_h, implied_a = probability(f.get('market_home_moneyline')), probability(f.get('market_away_moneyline'))
            if implied_h is not None and implied_a is not None:
                mp = implied_h/(implied_h+implied_a)
                market.append({'model_brier': (p-int(h>a))**2, 'market_brier': (mp-int(h>a))**2})
        weather = f.get('weather', {})
        labels = ['All games']
        if weather.get('status') == 'indoors': labels.append('Indoors')
        elif weather.get('status') == 'forecast':
            if number(weather.get('wind_mph')) >= 15: labels.append('Wind 15+ mph')
            if number(weather.get('temperature_f'), 60) < 40: labels.append('Below 40°F')
            if number(weather.get('precipitation_mm')) > .1: labels.append('Rain / snow forecast')
        if any(p.get('likely') is False for ps in f.get('personnel_snapshot', {}).values() for p in ps): labels.append('Starter absence expected')
        if any(c.get('inputs', {}).get('rest', 0) <= -3/7 for c in f.get('components', {}).values()): labels.append('Rest disadvantage 3+ days')
        for label in labels: conditions[label].append(r)
    reliability = [{'lower': b/5, 'upper': (b+1)/5, 'games': len(rows), 'predicted': float(np.mean([p for p, _ in rows])), 'actual': float(np.mean([y for _, y in rows]))} for b, rows in sorted(bins.items())]
    from app.tracking import rates
    condition_rows = [{'condition': name, 'games': len(rows), 'score_mae': float(np.mean([r['score_mae'] for r in rows])),
                       'mean_total_error': float(np.mean([r['total_error'] for r in rows])), 'record': rates(rows, 'winner_result')} for name, rows in conditions.items()]
    margin_pairs = [(abs(r['margin_error']), abs(r['home_score']-r['away_score']-f['market_home_margin'])) for f, r in matched if f.get('market_home_margin') is not None]
    total_pairs = [(abs(r['total_error']), abs(r['home_score']+r['away_score']-f['market_total'])) for f, r in matched if f.get('market_total') is not None]
    pair_summary = lambda pairs: {'games': len(pairs), 'model_mae': float(np.mean([p[0] for p in pairs])) if pairs else None, 'market_mae': float(np.mean([p[1] for p in pairs])) if pairs else None}
    return {'reliability': reliability, 'conditions': condition_rows, 'market_margin': pair_summary(margin_pairs), 'market_total': pair_summary(total_pairs),
            'market_probability': {'games': len(market), **{k: float(np.mean([r[k] for r in market])) if market else None for k in ('model_brier', 'market_brier')}},
            'note': 'All comparisons use the same archived pregame games. Moneylines are normalized to remove the two-sided overround; ties are excluded from this binary probability comparison. Conditions are descriptive, overlap, and do not establish causation. Lines are saved snapshots, not certified closing lines.'}
