"""Keeps each visitor's game separate, keyed by a random id their browser generates.

Everything is in memory: a restart forgets all games, which is fine for a toy app.
Idle games expire and the total is capped, so a public URL can't grow memory forever.
"""
import time
from collections import OrderedDict
from collections.abc import Callable


class SessionStore[T]:
    def __init__(self, factory: Callable[[], T], max_sessions: int = 200, ttl_seconds: float = 3600) -> None:
        self._factory = factory
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()  # least recently used first

    def get(self, session_id: str) -> T:
        """The session for `session_id`, created if new or expired."""
        now = time.monotonic()
        while self._items and next(iter(self._items.values()))[0] < now - self.ttl_seconds:
            self._items.popitem(last=False)

        item = self._items.pop(session_id, None)
        session = item[1] if item else self._factory()
        self._items[session_id] = (now, session)
        while len(self._items) > self.max_sessions:
            self._items.popitem(last=False)
        return session

    def __len__(self) -> int:
        return len(self._items)
