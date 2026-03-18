from __future__ import annotations
# receiver/downlink_state.py — Thread-safe downlink progress tracker
#
# Singleton that tracks the current image transfer state. Updated by
# listener.py on every recv() chunk. Read by dashboard SSE endpoint.

import threading
import time


class DownlinkState:
    """Thread-safe tracker for active image downlink progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"           # idle | receiving | validating | processing | complete | failed
        self._filename = ""
        self._declared_size = 0
        self._bytes_received = 0
        self._start_time = 0.0
        self._end_time = 0.0
        self._rate_bps = 0.0
        self._eta_sec = 0.0
        self._error = ""
        self._seq = 0                   # incremented on every update (SSE change detection)
        self._history: list[dict] = []  # last N completed transfers

    def start_transfer(self, filename: str, declared_size: int):
        with self._lock:
            self._status = "receiving"
            self._filename = filename
            self._declared_size = declared_size
            self._bytes_received = 0
            self._start_time = time.monotonic()
            self._end_time = 0.0
            self._rate_bps = 0.0
            self._eta_sec = 0.0
            self._error = ""
            self._seq += 1

    def update_progress(self, bytes_received: int):
        with self._lock:
            self._bytes_received = bytes_received
            elapsed = time.monotonic() - self._start_time
            if elapsed > 0:
                self._rate_bps = bytes_received / elapsed
                remaining = self._declared_size - bytes_received
                self._eta_sec = remaining / self._rate_bps if self._rate_bps > 0 else 0
            self._seq += 1

    def set_status(self, status: str, error: str = ""):
        with self._lock:
            self._status = status
            self._error = error
            if status in ("complete", "failed"):
                self._end_time = time.monotonic()
                elapsed = self._end_time - self._start_time if self._start_time else 0
                self._history.append({
                    "filename": self._filename,
                    "size": self._declared_size,
                    "status": status,
                    "duration_sec": round(elapsed, 2),
                    "rate_bps": round(self._rate_bps, 1),
                    "error": error,
                    "timestamp": time.time(),
                })
                # Keep last 20 transfers
                self._history = self._history[-20:]
            self._seq += 1

    def get_snapshot(self) -> dict:
        with self._lock:
            elapsed = 0.0
            if self._start_time and self._status == "receiving":
                elapsed = time.monotonic() - self._start_time
            elif self._end_time:
                elapsed = self._end_time - self._start_time

            pct = 0.0
            if self._declared_size > 0:
                pct = round(self._bytes_received / self._declared_size * 100, 1)

            return {
                "status": self._status,
                "filename": self._filename,
                "declared_size": self._declared_size,
                "bytes_received": self._bytes_received,
                "pct": pct,
                "rate_bps": round(self._rate_bps, 1),
                "eta_sec": round(self._eta_sec, 1),
                "elapsed_sec": round(elapsed, 2),
                "error": self._error,
                "seq": self._seq,
            }

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq


# Module-level singleton
_state = DownlinkState()


def get_state() -> DownlinkState:
    return _state
