from __future__ import annotations


class TIDState:
    __slots__ = ("qualified", "enroll_left", "locked_sid", "new_sid")

    def __init__(self):
        self.qualified: int = 0
        self.enroll_left: int = 0
        self.locked_sid: int | None = None
        self.new_sid: int | None = None


class TIDStateManager:
    def __init__(self):
        self._states: dict[int, TIDState] = {}

    def get(self, tid: int) -> TIDState:
        state = self._states.get(tid)
        if state is None:
            state = TIDState()
            self._states[tid] = state
        return state

    def forget(self, alive: set[int]) -> None:
        for tid in list(self._states.keys()):
            if tid not in alive:
                self._states.pop(tid, None)

