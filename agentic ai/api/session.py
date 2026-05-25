"""
Session Store — Simple in-memory dict for conversation history.

Maps session_id → list of conversation turns.
Each turn: {"role": "user"|"assistant", "content": "..."}.

Intentionally ephemeral — all sessions lost on server restart.
Replace with Redis/SQLite for persistence without changing the interface.
"""

from typing import Dict, List


# session_id → chronological list of conversation turns
_sessions: Dict[str, List[Dict[str, str]]] = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history for a session."""
    return _sessions.get(session_id, [])


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append a single turn to session history."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
