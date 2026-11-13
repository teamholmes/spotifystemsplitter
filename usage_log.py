"""Append-only application usage history and current-week summaries."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).resolve().with_name("usage_log.jsonl")


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {log_path} at line {line_number}."
                ) from error
            if not isinstance(event, dict):
                raise ValueError(
                    f"Invalid usage event in {log_path} at line {line_number}."
                )
            event_type = event.get("event")
            timestamp = event.get("timestamp")
            session_id = event.get("session_id")
            valid_event = (
                isinstance(event_type, str)
                and event_type in {"app_opened", "app_closed", "song_played"}
                and isinstance(timestamp, str)
                and isinstance(session_id, str)
            )
            if valid_event:
                try:
                    datetime.fromisoformat(timestamp)
                except ValueError:
                    valid_event = False
            if event_type == "song_played":
                valid_event = valid_event and isinstance(event.get("song"), str)
                valid_event = valid_event and isinstance(event.get("artist"), str)
            if not valid_event:
                raise ValueError(
                    f"Invalid usage event in {log_path} at line {line_number}."
                )
            events.append(event)
    return events


class UsageLog:
    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self.session_id = str(uuid.uuid4())
        self._started_monotonic = time.monotonic()
        self._closed = False

        _read_events(self.log_path)
        self._append(
            {
                "event": "app_opened",
                "session_id": self.session_id,
                "timestamp": _local_now().isoformat(timespec="seconds"),
            }
        )

    def _append(self, event: dict[str, Any]) -> None:
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")
                log_file.flush()
                os.fsync(log_file.fileno())

    def record_song(self, song: str, artist: str) -> None:
        if self._closed:
            raise RuntimeError("Cannot record a song after the app session has closed.")
        self._append(
            {
                "event": "song_played",
                "session_id": self.session_id,
                "timestamp": _local_now().isoformat(timespec="seconds"),
                "song": song,
                "artist": artist,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        duration_seconds = round(time.monotonic() - self._started_monotonic, 3)
        self._append(
            {
                "event": "app_closed",
                "session_id": self.session_id,
                "timestamp": _local_now().isoformat(timespec="seconds"),
                "duration_seconds": duration_seconds,
            }
        )
        self._closed = True


def weekly_summary(
    log_path: Path = LOG_PATH,
    now: datetime | None = None,
    active_session_id: str | None = None,
) -> dict[str, Any]:
    current_time = (now or _local_now()).astimezone()
    current_date = current_time.date()
    week_start_date = current_date - timedelta(days=current_date.weekday())
    week_end_date = week_start_date + timedelta(days=7)
    week_start = datetime.combine(week_start_date, datetime_time.min).astimezone()
    week_end = datetime.combine(week_end_date, datetime_time.min).astimezone()
    previous_week_start_date = week_start_date - timedelta(days=7)
    previous_week_start = datetime.combine(
        previous_week_start_date, datetime_time.min
    ).astimezone()

    sessions: dict[str, dict[str, datetime | None]] = {}
    song_events: list[tuple[datetime, str, str]] = []
    for event in _read_events(Path(log_path)):
        try:
            event_time = datetime.fromisoformat(event["timestamp"]).astimezone()
            session_id = event["session_id"]
            event_type = event["event"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Usage log contains an event with invalid fields.") from error
        if not isinstance(session_id, str) or not isinstance(event_type, str):
            raise ValueError("Usage log contains an event with invalid fields.")

        if event_type == "app_opened":
            sessions[session_id] = {"opened": event_time, "closed": None}
        elif event_type == "app_closed":
            session = sessions.get(session_id)
            if session is not None:
                session["closed"] = event_time
        elif event_type == "song_played":
            try:
                song = event["song"]
                artist = event["artist"]
            except KeyError as error:
                raise ValueError(
                    "Usage log contains a song event without a song or artist."
                ) from error
            if not isinstance(song, str) or not isinstance(artist, str):
                raise ValueError(
                    "Usage log contains a song event with invalid song or artist."
                )
            song_events.append((event_time, song, artist))

    def summarize_period(
        period_start: datetime, period_end: datetime
    ) -> dict[str, Any]:
        session_count = 0
        usage_seconds = 0.0
        for session_id, session in sessions.items():
            opened = session["opened"]
            if opened is None:
                continue
            closed = session["closed"]
            if closed is None and session_id == active_session_id:
                closed = current_time
            effective_end = closed or opened
            interval_start = max(opened, period_start)
            interval_end = min(effective_end, period_end, current_time)
            if opened < period_end and effective_end >= period_start:
                session_count += 1
            if interval_end > interval_start:
                usage_seconds += (interval_end - interval_start).total_seconds()

        period_songs = [
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "song": song,
                "artist": artist,
            }
            for timestamp, song, artist in song_events
            if period_start <= timestamp < period_end
        ]
        return {
            "week_start": period_start.date().isoformat(),
            "week_end": (period_end.date() - timedelta(days=1)).isoformat(),
            "sessions": session_count,
            "duration_seconds": int(usage_seconds),
            "songs": period_songs,
        }

    current_summary = summarize_period(week_start, week_end)
    previous_summary = summarize_period(previous_week_start, week_start)
    song_counts: dict[tuple[str, str], int] = {}
    for timestamp, song, artist in song_events:
        if week_start <= timestamp < week_end:
            key = (song, artist)
            song_counts[key] = song_counts.get(key, 0) + 1
    top_songs = [
        {"song": song, "artist": artist, "plays": count}
        for (song, artist), count in sorted(
            song_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:5]
    ]

    return {
        **current_summary,
        "top_songs": top_songs,
        "previous_week": previous_summary,
    }
