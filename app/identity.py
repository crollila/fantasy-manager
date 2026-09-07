import re
import unicodedata
from app.domain import Player


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())


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
