import re
import unicodedata
from app.domain import Player


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())


HISTORY_FIELDS = ('stats', 'games', 'workload_cv', 'efficiency_cv', 'bye', 'source', 'warnings')
MARKET_FIELDS = ('adp', 'adp_sd', 'espn_rank', 'ecr', 'market_stats', 'market_weight')


def merge_duplicates(players: list[Player], prefer=()) -> list[Player]:
    """Collapse catalog entries that share an external ID into one entry.

    Duplicates appear when the same person arrives from two sources (an ESPN roster entry and
    the nflverse history) before their IDs were linked. A single conflict makes every
    :class:`Identity` check fail, which blocks ESPN imports and catalog rebuilds indefinitely,
    so the catalog is repaired instead. The survivor is an ID in ``prefer`` (rosters and drafts
    point at it), else one with stat history, else the non-ESPN id. Entries at different
    positions are never merged; the shared ID is removed from all but the survivor.
    """
    prefer = set(prefer)
    parent = list(range(len(players)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, p in enumerate(players):
        for namespace, value in p.ids.items():
            key = (namespace, str(value))
            if key in owner:
                parent[root(i)] = root(owner[key])
            else:
                owner[key] = i
    groups = {}
    for i in range(len(players)):
        groups.setdefault(root(i), []).append(i)
    if all(len(g) == 1 for g in groups.values()):
        return players
    dropped = set()
    for members in groups.values():
        if len(members) == 1:
            continue
        group = [players[i] for i in members]
        keeper = min(group, key=lambda p: (p.id not in prefer, not p.stats, p.id.startswith('espn:')))
        for other in group:
            if other is keeper:
                continue
            if other.position != keeper.position:
                shared = {k for k, v in other.ids.items() if keeper.ids.get(k) == v}
                other.ids = {k: v for k, v in other.ids.items() if k not in shared}
                continue
            if not keeper.stats and other.stats:
                for field in HISTORY_FIELDS:
                    setattr(keeper, field, getattr(other, field))
            for field in MARKET_FIELDS:
                if getattr(keeper, field) in (None, 0) and getattr(other, field) not in (None, 0):
                    setattr(keeper, field, getattr(other, field))
            keeper.ids = other.ids | keeper.ids
            if keeper.team == 'FA' and other.team != 'FA':
                keeper.team = other.team
            dropped.add(id(other))
    out = [p for p in players if id(p) not in dropped]
    # A survivor may have inherited an ID that a different-position entry still holds.
    seen = {}
    for p in out:
        for key in list(p.ids.items()):
            holder = seen.setdefault((key[0], str(key[1])), p.id)
            if holder != p.id:
                del p.ids[key[0]]
    return out


def merge_catalog(existing: list[Player], built: list[Player], prefer=()) -> list[Player]:
    """Fold freshly built nflverse players into the saved catalog without creating duplicates.

    Saved entries keep their canonical IDs (rosters, drafts and archived forecasts point at
    them) and their market fields; stat history is replaced by the new build. A built player
    matches a saved one by canonical ID or by any shared external ID.
    """
    existing = merge_duplicates([p.model_copy(deep=True) for p in existing], prefer)
    merged = {p.id: p for p in existing}
    by_external = {(k, str(v)): p for p in existing for k, v in p.ids.items()}
    for b in built:
        target = merged.get(b.id) or next((by_external[(k, str(v))] for k, v in b.ids.items() if (k, str(v)) in by_external), None)
        if target is None:
            merged[b.id] = b
            for k, v in b.ids.items(): by_external.setdefault((k, str(v)), b)
            continue
        if target.position != b.position: continue
        for field in HISTORY_FIELDS:
            setattr(target, field, getattr(b, field))
        target.ids = b.ids | target.ids
        for k, v in target.ids.items(): by_external.setdefault((k, str(v)), target)
        if b.team and b.team != 'FA': target.team = b.team
    return merge_duplicates(list(merged.values()), prefer)


def referenced_ids(store) -> set[str]:
    """Player IDs that saved rosters, ESPN snapshots or drafts point at."""
    import json
    refs = set()
    with store.connect() as c:
        for key, value in c.execute("SELECT key,value FROM meta WHERE key LIKE 'ownership:%' OR key LIKE 'espn-season:%'"):
            try:
                body = json.loads(value)
            except ValueError:
                continue
            if key.startswith('ownership:'):
                refs.update(pid for ids in body.get('rosters', {}).values() for pid in ids)
            else:
                refs.update(body.get('weekly', {}))
        for (body,) in c.execute('SELECT body FROM drafts'):
            try:
                refs.update(p['player_id'] for p in json.loads(body))
            except (ValueError, KeyError, TypeError):
                continue
    return refs


class Identity:
    def __init__(self, players: list[Player]):
        self.players = {p.id: p for p in players}
        self.external = {}
        for p in players:
            for namespace, value in p.ids.items():
                key = (namespace, str(value))
                if key in self.external and self.external[key] != p.id:
                    raise ValueError(f"Conflicting identity: {key}")
                self.external[key] = p.id

    def resolve(self, namespace, value):
        if namespace == "canonical" and str(value) in self.players:
            return str(value)
        key = (namespace, str(value))
        if key not in self.external:
            raise ValueError(f"Unmapped {namespace} ID {value}; explicit mapping required")
        return self.external[key]

    def suggestions(self, name, position=None):
        # Suggestions only. Callers must explicitly confirm a mapping.
        return [p.id for p in self.players.values() if normalized(p.name) == normalized(name) and (position is None or p.position == position)]
