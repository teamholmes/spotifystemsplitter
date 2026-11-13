"""Monitor Spotify playback and select AmpliTube/Tone3000 patches over MIDI."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import mido
import requests
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"
TOKEN_CACHE_PATH = PROJECT_DIR / ".spotify-token-cache"
NTFY_URL = "https://ntfy.sh/spotify-jacauavh1735gsibHVJhjnnn"


def post_now_playing(song_name: str, artist: str) -> None:
    """Publish a track change to the configured ntfy topic."""
    response = requests.post(
        NTFY_URL,
        data=f"Song: {song_name}\nArtist: {artist}".encode("utf-8"),
        headers={"Title": "Spotify now playing"},
        timeout=5,
    )
    response.raise_for_status()


def _normalize_mapping(mapping: Any, name: str) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise ValueError(f"'{name}' must be an object mapping song names to program numbers")

    normalized: dict[str, int] = {}
    for song, program in mapping.items():
        try:
            program_number = int(program)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Program number for '{song}' must be an integer") from error
        if not 0 <= program_number <= 127:
            raise ValueError(f"Program number for '{song}' must be between 0 and 127")
        normalized[song] = program_number
    return normalized


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate playlist and MIDI program mappings."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object")
    playlist_name = config.get("playlist_name")
    if not isinstance(playlist_name, str) or not playlist_name.strip():
        raise ValueError("Config file must include a non-empty 'playlist_name'")

    default_program = config.get("default_program_change", 16)
    if not isinstance(default_program, int) or not 0 <= default_program <= 127:
        raise ValueError("'default_program_change' must be an integer between 0 and 127")

    return {
        "playlist_name": playlist_name,
        "default_program_change": default_program,
        "song_to_patch": _normalize_mapping(
            config.get("song_to_patch"), "song_to_patch"
        ),
        "tone3000_song_to_patch": _normalize_mapping(
            config.get("tone3000_song_to_patch", {}), "tone3000_song_to_patch"
        ),
    }


def get_patch_for_song(
    song_name: str, mapping: dict[str, int], default_program: int
) -> int:
    normalized_name = song_name.casefold()
    for mapped_song, program in mapping.items():
        if mapped_song.casefold() == normalized_name:
            return program
    return default_program


def _open_midi_output(port_name: str) -> mido.ports.BaseOutput:
    try:
        outputs = mido.get_output_names()
        if port_name in outputs:
            return mido.open_output(port_name)
        return mido.open_output(port_name, virtual=True)
    except Exception as error:
        raise RuntimeError(f"Unable to open MIDI output '{port_name}': {error}") from error


def run_spotify_midi(
    midi_port: str,
    config_path: str | Path,
    stop_event: threading.Event,
    report_status: Callable[[str], None],
    report_track: Callable[[str], None],
) -> None:
    """Run the Spotify monitor until stopped, reporting progress to the UI."""
    load_dotenv(PROJECT_DIR / ".env")
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to the app's .env file."
        )

    config = load_config(config_path)
    report_status("Connecting to Spotify...")
    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=os.getenv(
            "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"
        ),
        scope="user-read-currently-playing playlist-read-private",
        cache_path=str(TOKEN_CACHE_PATH),
    )
    spotify = spotipy.Spotify(auth_manager=auth)

    report_status(f"Looking for playlist: {config['playlist_name']}")
    playlists = spotify.current_user_playlists()
    playlist_found = False
    while playlists:
        if any(
            item["name"] == config["playlist_name"] for item in playlists["items"]
        ):
            playlist_found = True
            break
        playlists = spotify.next(playlists) if playlists["next"] else None
    if not playlist_found:
        raise RuntimeError(f"Playlist '{config['playlist_name']}' was not found.")

    midi_output = _open_midi_output(midi_port)
    previous_track_id: str | None = None
    report_status(f"Monitoring Spotify; MIDI output: {midi_port}")
    try:
        while not stop_event.is_set():
            playback = spotify.current_user_playing_track()
            track = playback.get("item") if playback else None
            if track:
                track_id = track.get("id") or track["name"]
                if track_id != previous_track_id:
                    track_name = track["name"]
                    artist = (
                        track["artists"][0]["name"] if track["artists"] else "Unknown"
                    )
                    amplitube_program = get_patch_for_song(
                        track_name,
                        config["song_to_patch"],
                        config["default_program_change"],
                    )
                    tone3000_program = get_patch_for_song(
                        track_name, config["tone3000_song_to_patch"], 0
                    )
                    midi_output.send(
                        mido.Message(
                            "program_change", program=amplitube_program, channel=0
                        )
                    )
                    midi_output.send(
                        mido.Message(
                            "program_change", program=tone3000_program, channel=5
                        )
                    )
                    report_track(track_name)
                    previous_track_id = track_id
                    try:
                        post_now_playing(track_name, artist)
                    except requests.RequestException as error:
                        report_status(f"Ntfy notification failed: {error}")
                    else:
                        report_status(
                            f"Now playing: {track_name} by {artist} "
                            f"(AmpliTube {amplitube_program}, "
                            f"Tone3000 {tone3000_program}); ntfy notification sent."
                        )
            stop_event.wait(5)
    finally:
        midi_output.close()
    report_status("Spotify MIDI monitor stopped.")
