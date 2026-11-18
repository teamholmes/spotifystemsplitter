#!/usr/bin/env python3
"""Play Spotify through a stem-reduction model using BlackHole as input."""

from __future__ import annotations

import colorsys
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from io import BytesIO
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import psutil
import sounddevice as sd
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from PIL import Image, ImageTk

from spotify_midi import (
    DEFAULT_CONFIG_PATH,
    is_song_configured,
    load_config,
    run_spotify_midi,
)
from usage_log import LOG_PATH, UsageLog, weekly_summary


SAMPLE_RATE = 44_100
CHANNELS = 2
WINDOW_SECONDS = 6
HOP_SECONDS = 3
BLOCK_SIZE = 1_024
QUALITY_SETTINGS = {
    "Balanced": (1, 0.25),
    "Higher quality (slower)": (2, 0.5),
    "Highest quality (slowest)": (4, 0.75),
}


def apply_playback_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    boosted = samples * gain
    if gain <= 1.0:
        return boosted

    threshold = 1.0 / gain
    over_threshold = np.abs(samples) > threshold
    magnitude = np.abs(samples[over_threshold])
    limited_magnitude = threshold + (1.0 - threshold) * (
        1.0 - np.exp(-gain * (magnitude - threshold) / (1.0 - threshold))
    )
    boosted[over_threshold] = np.sign(samples[over_threshold]) * limited_magnitude
    return boosted


def format_elapsed_time(elapsed_seconds: int) -> str:
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class GuitarRemoverApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        for font_name in tkfont.names(root):
            app_font = tkfont.nametofont(font_name, root=root)
            font_size = app_font.cget("size")
            if font_size > 1:
                app_font.configure(size=font_size - 1)
            elif font_size < -1:
                app_font.configure(size=font_size + 1)
        self.root.title("Spotify Stem Remover + AmpliTube MIDI")
        self.root.minsize(520, 740)

        self.usage_log = UsageLog()
        self.last_usage_summary_update = 0.0
        self.input_devices: dict[str, int] = {}
        self.output_devices: dict[str, int] = {}
        self.metronome_output_devices: dict[str, int] = {}
        self.remove_percent = tk.IntVar(value=100)
        self.removal_amount = 1.0
        self.remove_vocals_percent = tk.IntVar(value=0)
        self.vocal_removal_amount = 0.0
        self.remove_bass_percent = tk.IntVar(value=0)
        self.bass_removal_amount = 0.0
        self.remove_drums_percent = tk.IntVar(value=0)
        self.drums_removal_amount = 0.0
        self.volume_percent = tk.IntVar(value=180)
        self.playback_gain = 1.8
        self.status = tk.StringVar(value="Choose audio devices, then start.")
        self.cpu_usage = tk.StringVar(value="System CPU usage: measuring...")
        psutil.cpu_percent(interval=None)
        self.last_cpu_update = time.monotonic()
        self.stop_event = threading.Event()
        self.stream_active = threading.Event()
        self.processing_active = threading.Event()
        self.meter_lock = threading.Lock()
        self.input_level = 0.0
        self.output_level = 0.0
        self.animation_frame = 0
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.midi_status = tk.StringVar(value="Stopped")
        self.midi_port = tk.StringVar(value="IAC Driver Bus 1")
        self.now_playing = tk.StringVar(value="")
        self.midi_status_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.album_artwork: ImageTk.PhotoImage | None = None
        self.weekly_daily: list[dict[str, Any]] = []
        self.weekly_song_colors: dict[tuple[str, str], str] = {}
        self.selected_week_offset = 0
        self.metronome_bpm = tk.DoubleVar(value=90)
        self.metronome_output_device = tk.StringVar(value="")
        self.metronome_blackhole_enabled = tk.BooleanVar(value=False)
        self.metronome_running = False
        self.metronome_after_id: str | None = None
        self.metronome_next_beat = 0.0
        self.metronome_beat_count = 0
        self.metronome_click_generation = 0
        self.metronome_streams: list[sd.OutputStream] = []
        self.midi_stop_event = threading.Event()
        self.midi_worker: threading.Thread | None = None
        self.input_overflow = threading.Event()
        self.worker: threading.Thread | None = None

        self._build_ui()
        self.refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll_status)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.player_page = ttk.Frame(notebook, padding=10)
        self.history_page = ttk.Frame(notebook, padding=10)
        self.metronome_page = ttk.Frame(notebook, padding=10)
        notebook.add(self.player_page, text="Player")
        notebook.add(self.history_page, text="Weekly History")
        notebook.add(self.metronome_page, text="Metronome")

        frame = self.player_page
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Spotify input").grid(row=0, column=0, sticky="w", pady=2)
        self.input_choice = ttk.Combobox(frame, state="readonly")
        self.input_choice.grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=2)

        ttk.Label(frame, text="Headphones / output").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.output_choice = ttk.Combobox(frame, state="readonly")
        self.output_choice.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=2)

        ttk.Button(frame, text="Refresh devices", command=self.refresh_devices).grid(
            row=2, column=1, sticky="e", pady=(0, 4)
        )

        ttk.Label(frame, text="Guitar removal").grid(row=3, column=0, sticky="w")
        slider_frame = ttk.Frame(frame)
        slider_frame.grid(row=3, column=1, sticky="ew", padx=(12, 0))
        slider_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            variable=self.remove_percent,
            command=self._update_percent_label,
        ).grid(row=0, column=0, sticky="ew")
        self.percent_label = ttk.Label(slider_frame, text="100%", width=5, anchor="e")
        self.percent_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Vocal removal").grid(row=4, column=0, sticky="w")
        vocal_slider_frame = ttk.Frame(frame)
        vocal_slider_frame.grid(row=4, column=1, sticky="ew", padx=(12, 0))
        vocal_slider_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            vocal_slider_frame,
            from_=0,
            to=100,
            variable=self.remove_vocals_percent,
            command=self._update_vocal_percent_label,
        ).grid(row=0, column=0, sticky="ew")
        self.vocal_percent_label = ttk.Label(
            vocal_slider_frame, text="0%", width=5, anchor="e"
        )
        self.vocal_percent_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Bass guitar removal").grid(
            row=5, column=0, sticky="w"
        )
        bass_slider_frame = ttk.Frame(frame)
        bass_slider_frame.grid(row=5, column=1, sticky="ew", padx=(12, 0))
        bass_slider_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            bass_slider_frame,
            from_=0,
            to=100,
            variable=self.remove_bass_percent,
            command=self._update_bass_percent_label,
        ).grid(row=0, column=0, sticky="ew")
        self.bass_percent_label = ttk.Label(
            bass_slider_frame, text="0%", width=5, anchor="e"
        )
        self.bass_percent_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Drums removal").grid(row=6, column=0, sticky="w")
        drums_slider_frame = ttk.Frame(frame)
        drums_slider_frame.grid(row=6, column=1, sticky="ew", padx=(12, 0))
        drums_slider_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            drums_slider_frame,
            from_=0,
            to=100,
            variable=self.remove_drums_percent,
            command=self._update_drums_percent_label,
        ).grid(row=0, column=0, sticky="ew")
        self.drums_percent_label = ttk.Label(
            drums_slider_frame, text="0%", width=5, anchor="e"
        )
        self.drums_percent_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Spotify volume").grid(row=7, column=0, sticky="w")
        volume_frame = ttk.Frame(frame)
        volume_frame.grid(row=7, column=1, sticky="ew", padx=(12, 0))
        volume_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            volume_frame,
            from_=100,
            to=200,
            variable=self.volume_percent,
            command=self._update_volume,
        ).grid(row=0, column=0, sticky="ew")
        self.volume_label = ttk.Label(volume_frame, text="180%", width=5, anchor="e")
        self.volume_label.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Separation quality").grid(
            row=8, column=0, sticky="w", pady=2
        )
        self.quality_choice = ttk.Combobox(
            frame,
            state="readonly",
            values=tuple(QUALITY_SETTINGS),
            width=30,
        )
        self.quality_choice.set("Higher quality (slower)")
        self.quality_choice.grid(row=8, column=1, sticky="w", padx=(12, 0), pady=2)

        visualizer = ttk.LabelFrame(frame, text="Live audio activity", padding=8)
        visualizer.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        visualizer.columnconfigure(0, weight=1)
        self.visualizer = tk.Canvas(
            visualizer,
            height=76,
            background="#101722",
            highlightthickness=0,
        )
        self.visualizer.grid(row=0, column=0, sticky="ew")

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=10, column=0, sticky="w", pady=(4, 2))
        self.stop_button = ttk.Button(
            frame, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=10, column=1, sticky="e", pady=(4, 2))

        ttk.Label(
            frame,
            text="Uses a 6-second AI audio window with a 3-second overlap.",
            wraplength=430,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(2, 4))
        ttk.Label(frame, textvariable=self.status, wraplength=430).grid(
            row=12, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, textvariable=self.cpu_usage).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        midi_frame = ttk.LabelFrame(frame, text="Spotify to AmpliTube MIDI", padding=8)
        midi_frame.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        midi_frame.columnconfigure(2, weight=1)
        self.artwork_canvas = tk.Canvas(
            midi_frame,
            width=96,
            height=96,
            background="#101722",
            highlightthickness=0,
        )
        self.artwork_canvas.grid(row=0, column=0, rowspan=4, sticky="nw", padx=(0, 8))
        self.artwork_canvas.create_text(
            48,
            48,
            text="ALBUM ART",
            fill="#9aa9bd",
            font=("TkDefaultFont", 8),
        )
        ttk.Label(midi_frame, text="Song title (Ctrl+C to copy into config)").grid(
            row=0, column=1, sticky="w"
        )
        self.now_playing_entry = ttk.Entry(
            midi_frame, textvariable=self.now_playing, state="readonly", width=42
        )
        self.now_playing_entry.grid(row=0, column=2, sticky="ew", padx=(12, 0))
        self.now_playing_entry.bind("<Control-c>", self._copy_now_playing)
        self.now_playing_entry.bind("<Control-C>", self._copy_now_playing)
        ttk.Label(midi_frame, text="MIDI output port").grid(
            row=1, column=1, sticky="w", pady=(4, 0)
        )
        ttk.Entry(midi_frame, textvariable=self.midi_port).grid(
            row=1, column=2, sticky="ew", padx=(12, 0), pady=(4, 0)
        )
        self.midi_start_button = ttk.Button(
            midi_frame, text="Start Spotify MIDI", command=self.start_midi
        )
        self.midi_start_button.grid(row=2, column=1, sticky="w", pady=(4, 2))
        self.midi_stop_button = ttk.Button(
            midi_frame,
            text="Stop MIDI",
            command=self.stop_midi,
            state="disabled",
        )
        self.midi_stop_button.grid(row=2, column=2, sticky="e", pady=(4, 2))
        ttk.Label(midi_frame, textvariable=self.midi_status, wraplength=460).grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(2, 0)
        )

        self._build_history_ui()
        self._build_metronome_ui()
        self._refresh_weekly_summary()

    def _build_metronome_ui(self) -> None:
        page = self.metronome_page
        page.columnconfigure(0, weight=1)

        ttk.Label(
            page,
            text="Metronome",
            font=("TkDefaultFont", 18, "bold"),
        ).grid(row=0, column=0, pady=(48, 12))
        self.metronome_tempo_label = tk.StringVar(value="90 BPM")
        ttk.Label(
            page,
            textvariable=self.metronome_tempo_label,
            font=("TkDefaultFont", 24, "bold"),
        ).grid(row=1, column=0, pady=(0, 16))

        ttk.Scale(
            page,
            from_=70,
            to=120,
            variable=self.metronome_bpm,
            command=self._update_metronome_bpm,
        ).grid(row=2, column=0, sticky="ew", padx=32)
        range_frame = ttk.Frame(page)
        range_frame.grid(row=3, column=0, sticky="ew", padx=32)
        range_frame.columnconfigure(0, weight=1)
        range_frame.columnconfigure(1, weight=1)
        ttk.Label(range_frame, text="70 BPM").grid(row=0, column=0, sticky="w")
        ttk.Label(range_frame, text="120 BPM").grid(row=0, column=1, sticky="e")

        ttk.Label(page, text="Headphones / speaker").grid(
            row=4, column=0, sticky="w", padx=32, pady=(20, 2)
        )
        self.metronome_output_choice = ttk.Combobox(
            page,
            textvariable=self.metronome_output_device,
            state="readonly",
        )
        self.metronome_output_choice.grid(row=5, column=0, sticky="ew", padx=32)
        self.metronome_blackhole_check = ttk.Checkbutton(
            page,
            text="Also play through BlackHole",
            variable=self.metronome_blackhole_enabled,
        )
        self.metronome_blackhole_check.grid(
            row=6, column=0, sticky="w", padx=32, pady=(8, 0)
        )

        controls = ttk.Frame(page)
        controls.grid(row=7, column=0, pady=24)
        self.metronome_start_button = ttk.Button(
            controls,
            text="Start",
            command=self._start_metronome,
        )
        self.metronome_start_button.pack(side="left", padx=6)
        self.metronome_stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self._stop_metronome,
            state="disabled",
        )
        self.metronome_stop_button.pack(side="left", padx=6)

        self.metronome_status = tk.StringVar(value="Stopped")
        ttk.Label(
            page,
            textvariable=self.metronome_status,
            font=("TkDefaultFont", 12),
        ).grid(row=8, column=0)
        ttk.Label(
            page,
            text="Choose headphones or speakers; optionally send the click to BlackHole too.",
            wraplength=460,
        ).grid(row=9, column=0, pady=(12, 0))

    def _update_metronome_bpm(self, value: str) -> None:
        bpm = int(round(float(value)))
        self.metronome_tempo_label.set(f"{bpm} BPM")

    def _start_metronome(self) -> None:
        if self.metronome_running:
            return
        output_name = self.metronome_output_device.get()
        if output_name not in self.metronome_output_devices:
            self.metronome_status.set("Choose a headphones or speaker output.")
            return

        output_indices = [self.metronome_output_devices[output_name]]
        blackhole_name = next(
            (name for name in self.output_devices if "blackhole" in name.lower()),
            None,
        )
        if (
            self.metronome_blackhole_enabled.get()
            and blackhole_name is not None
            and self.output_devices[blackhole_name] not in output_indices
        ):
            output_indices.append(self.output_devices[blackhole_name])

        self.metronome_click_generation = 0
        streams: list[sd.OutputStream] = []
        try:
            for device_index in output_indices:
                device = sd.query_devices(device_index)
                sample_rate = int(device["default_samplerate"])
                stream = sd.OutputStream(
                    device=device_index,
                    samplerate=sample_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._make_metronome_callback(sample_rate),
                )
                streams.append(stream)
                stream.start()
        except (sd.PortAudioError, ValueError, OSError) as error:
            cleanup_errors: list[str] = []
            for stream in streams:
                try:
                    if stream.active:
                        stream.stop()
                except sd.PortAudioError as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
                try:
                    stream.close()
                except sd.PortAudioError as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            status = f"Unable to start metronome output: {error}"
            if cleanup_errors:
                status += f" Cleanup failed: {'; '.join(cleanup_errors)}"
            self.metronome_status.set(status)
            return

        self.metronome_streams = streams
        self.metronome_running = True
        self.metronome_beat_count = 0
        self.metronome_next_beat = time.monotonic()
        self.metronome_start_button.configure(state="disabled")
        self.metronome_stop_button.configure(state="normal")
        self._metronome_tick()

    def _make_metronome_callback(self, sample_rate: int):
        click_frames = int(sample_rate * 0.035)
        positions = np.arange(click_frames)
        click_wave = 0.75 * np.exp(-positions / (sample_rate * 0.008)) * np.sin(
            2 * np.pi * 1000 * positions / sample_rate
        )
        click_generation = self.metronome_click_generation
        click_position = click_frames

        def callback(
            outdata: np.ndarray,
            frames: int,
            _time_info: Any,
            _status: Any,
        ) -> None:
            nonlocal click_generation, click_position
            outdata.fill(0)
            if click_generation != self.metronome_click_generation:
                click_generation = self.metronome_click_generation
                click_position = 0

            active_frames = min(frames, max(0, click_frames - click_position))
            if active_frames:
                active_click = click_wave[
                    click_position : click_position + active_frames
                ]
                outdata[:active_frames, :] = active_click[:, None]
                click_position += active_frames

        return callback

    def _stop_metronome(self) -> None:
        self.metronome_running = False
        if self.metronome_after_id is not None:
            self.root.after_cancel(self.metronome_after_id)
            self.metronome_after_id = None
        streams, self.metronome_streams = self.metronome_streams, []
        stop_errors: list[str] = []
        for stream in streams:
            try:
                if stream.active:
                    stream.stop()
            except sd.PortAudioError as error:
                stop_errors.append(str(error))
            try:
                stream.close()
            except sd.PortAudioError as error:
                stop_errors.append(str(error))
        self.metronome_start_button.configure(state="normal")
        self.metronome_stop_button.configure(state="disabled")
        if stop_errors:
            self.metronome_status.set(
                f"Metronome output stop failed: {'; '.join(stop_errors)}"
            )
        else:
            self.metronome_status.set("Stopped")

    def _metronome_tick(self) -> None:
        if not self.metronome_running:
            return
        self.metronome_after_id = None
        self.metronome_click_generation += 1
        self.metronome_beat_count += 1
        bpm = int(round(self.metronome_bpm.get()))
        self.metronome_status.set(f"Beat {self.metronome_beat_count} · {bpm} BPM")

        interval = 60.0 / bpm
        self.metronome_next_beat += interval
        now = time.monotonic()
        if self.metronome_next_beat <= now:
            self.metronome_next_beat = now + interval
        delay_ms = max(1, round((self.metronome_next_beat - now) * 1000))
        self.metronome_after_id = self.root.after(delay_ms, self._metronome_tick)

    def _build_history_ui(self) -> None:
        page = self.history_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(3, weight=1)

        navigation = ttk.Frame(page)
        navigation.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.previous_week_button = ttk.Button(
            navigation,
            text="← Previous week",
            command=lambda: self._change_history_week(-1),
        )
        self.previous_week_button.pack(side="left")
        self.history_week_label = tk.StringVar(value="This week")
        ttk.Label(navigation, textvariable=self.history_week_label).pack(
            side="left", expand=True
        )
        self.next_week_button = ttk.Button(
            navigation,
            text="Next week →",
            command=lambda: self._change_history_week(1),
            state="disabled",
        )
        self.next_week_button.pack(side="right")
        self.this_week_button = ttk.Button(
            navigation,
            text="This week",
            command=self._show_current_week,
            state="disabled",
        )
        self.this_week_button.pack(side="right", padx=(0, 8))

        summary_frame = ttk.LabelFrame(page, text="Selected week", padding=8)
        summary_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        summary_frame.columnconfigure(0, weight=1)
        self.weekly_stats = tk.StringVar(value="Loading weekly summary...")
        ttk.Label(
            summary_frame, textvariable=self.weekly_stats, justify="left"
        ).grid(
            row=0, column=0, sticky="w"
        )
        self.weekly_top_songs = tk.StringVar(
            value="Top 5 songs for selected week: loading..."
        )
        ttk.Label(
            summary_frame,
            textvariable=self.weekly_top_songs,
            justify="left",
            wraplength=480,
        ).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )

        chart_frame = ttk.LabelFrame(
            page,
            text="Daily song time (estimated from track changes)",
            padding=6,
        )
        chart_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0)
        )
        chart_frame.columnconfigure(0, weight=1)
        self.weekly_chart = tk.Canvas(
            chart_frame,
            height=205,
            background="#101722",
            highlightthickness=0,
        )
        self.weekly_chart.grid(row=0, column=0, sticky="ew")
        self.weekly_chart.bind("<Configure>", self._draw_weekly_chart)
        self.weekly_chart_info = tk.StringVar(
            value="Hover over a colored bar segment to see the song and artist."
        )
        ttk.Label(
            chart_frame,
            textvariable=self.weekly_chart_info,
            wraplength=480,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        songs_frame = ttk.LabelFrame(
            page, text="Song time for selected week", padding=8
        )
        songs_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        songs_frame.columnconfigure(0, weight=1)
        songs_frame.rowconfigure(0, weight=1)
        self.weekly_songs = tk.Text(
            songs_frame,
            height=8,
            width=64,
            wrap="word",
            state="disabled",
        )
        self.weekly_songs.grid(row=0, column=0, sticky="nsew")
        weekly_scrollbar = ttk.Scrollbar(
            songs_frame, orient="vertical", command=self.weekly_songs.yview
        )
        weekly_scrollbar.grid(row=0, column=1, sticky="ns")
        self.weekly_songs.configure(yscrollcommand=weekly_scrollbar.set)
        ttk.Label(
            page,
            text=f"History file: {LOG_PATH} (JSON Lines)",
            wraplength=480,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _change_history_week(self, offset: int) -> None:
        self.selected_week_offset = min(0, self.selected_week_offset + offset)
        self._refresh_weekly_summary()

    def _show_current_week(self) -> None:
        self.selected_week_offset = 0
        self._refresh_weekly_summary()

    def _refresh_weekly_summary(self) -> None:
        try:
            summary = weekly_summary(
                active_session_id=self.usage_log.session_id,
                week_offset=self.selected_week_offset,
            )
        except (OSError, ValueError) as error:
            self.weekly_stats.set(f"Usage summary unavailable: {error}")
            return

        if self.selected_week_offset == 0:
            self.history_week_label.set("This week")
        else:
            self.history_week_label.set(
                f"Week of {summary['week_start']} to {summary['week_end']}"
            )
        self.next_week_button.configure(
            state="normal" if self.selected_week_offset < 0 else "disabled"
        )
        self.this_week_button.configure(
            state="normal" if self.selected_week_offset < 0 else "disabled"
        )
        self.weekly_stats.set(
            f"{summary['week_start']} to {summary['week_end']}  |  "
            f"{summary['sessions']} sessions  |  "
            f"{format_elapsed_time(summary['duration_seconds'])} app-session time  |  "
            f"{len(summary['songs'])} track changes\n"
            f"Previous week ({summary['previous_week']['week_start']} to "
            f"{summary['previous_week']['week_end']}): "
            f"{summary['previous_week']['sessions']} sessions  |  "
            f"{format_elapsed_time(summary['previous_week']['duration_seconds'])} "
            f"app-session time  |  "
            f"{len(summary['previous_week']['songs'])} track changes"
        )
        self.weekly_daily = summary["daily"]
        song_totals = summary["song_totals"]
        self.weekly_song_colors = {}
        for index, song in enumerate(song_totals):
            hue = index / max(len(song_totals), 1)
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.94)
            self.weekly_song_colors[(song["song"], song["artist"])] = (
                f"#{round(red * 255):02x}{round(green * 255):02x}"
                f"{round(blue * 255):02x}"
            )
        self._draw_weekly_chart()
        if summary["top_songs"]:
            top_songs = "\n".join(
                f"{index}. {song['song']} — {song['artist']} "
                f"({song['plays']} plays)"
                for index, song in enumerate(summary["top_songs"], start=1)
            )
            self.weekly_top_songs.set(f"Top 5 songs for selected week:\n{top_songs}")
        else:
            self.weekly_top_songs.set(
                "Top 5 songs for selected week:\nNo songs recorded."
            )
        self.weekly_songs.configure(state="normal")
        self.weekly_songs.delete("1.0", "end")
        for index, song in enumerate(song_totals):
            tag = f"song-{index}"
            color = self.weekly_song_colors[(song["song"], song["artist"])]
            self.weekly_songs.tag_configure(tag, foreground=color)
            self.weekly_songs.insert(
                "end",
                f"■ {format_elapsed_time(song['duration_seconds'])}  "
                f"{song['song']} — {song['artist']}\n",
                tag,
            )
        if not song_totals:
            self.weekly_songs.insert(
                "end", "No song time recorded for this week.\n"
            )
        self.weekly_songs.configure(state="disabled")
        self.weekly_songs.yview_moveto(0.0)
        self.last_usage_summary_update = time.monotonic()

    def _draw_weekly_chart(self, _event: tk.Event | None = None) -> None:
        canvas = self.weekly_chart
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        days = self.weekly_daily
        max_duration = max(
            (int(day["track_duration_seconds"]) for day in days), default=0
        )
        baseline = height - 32
        chart_height = height - 64
        slot_width = width / max(len(days), 1)
        for index, day in enumerate(days):
            total_seconds = int(day["track_duration_seconds"])
            bar_height = (
                chart_height * total_seconds / max_duration if max_duration else 0
            )
            center_x = slot_width * (index + 0.5)
            bar_width = min(42, slot_width * 0.58)
            segment_bottom = baseline
            for song in day["songs"]:
                segment_height = (
                    chart_height * int(song["duration_seconds"]) / max_duration
                    if max_duration
                    else 0
                )
                segment_top = segment_bottom - segment_height
                color = self.weekly_song_colors.get(
                    (song["song"], song["artist"]), "#45d6b2"
                )
                if segment_height > 0:
                    segment = canvas.create_rectangle(
                        center_x - bar_width / 2,
                        segment_top,
                        center_x + bar_width / 2,
                        segment_bottom,
                        fill=color,
                        outline="#101722",
                    )
                    canvas.tag_bind(
                        segment,
                        "<Enter>",
                        lambda event, day=day, song=song: (
                            self._show_weekly_chart_info(day, song)
                        ),
                    )
                    canvas.tag_bind(
                        segment,
                        "<Leave>",
                        lambda _event: self.weekly_chart_info.set(
                            "Hover over a colored bar segment to see the song and artist."
                        ),
                    )
                segment_bottom = segment_top
            day_name = datetime.strptime(str(day["date"]), "%Y-%m-%d").strftime("%a")
            canvas.create_text(
                center_x,
                baseline + 13,
                text=day_name,
                fill="#d5deea",
                font=("TkDefaultFont", 8),
            )
            duration_label = format_elapsed_time(total_seconds)
            canvas.create_text(
                center_x,
                max(10, baseline - bar_height - 10),
                text=duration_label,
                fill="#9aa9bd",
                font=("TkDefaultFont", 8),
            )

    def _show_weekly_chart_info(
        self, day: dict[str, Any], song: dict[str, Any]
    ) -> None:
        day_label = datetime.strptime(str(day["date"]), "%Y-%m-%d").strftime(
            "%A, %b %d"
        )
        self.weekly_chart_info.set(
            f"{day_label}: {song['song']} — {song['artist']} "
            f"({format_elapsed_time(int(song['duration_seconds']))})"
        )

    def _show_album_artwork(self, artwork: bytes | None) -> None:
        self.artwork_canvas.delete("all")
        self.album_artwork = None
        if artwork is None:
            self._draw_album_artwork_placeholder()
            return

        try:
            with Image.open(BytesIO(artwork)) as image:
                cover = image.convert("RGB")
            cover.thumbnail((96, 96), Image.Resampling.LANCZOS)
            self.album_artwork = ImageTk.PhotoImage(cover, master=self.root)
        except (OSError, ValueError) as error:
            self._draw_album_artwork_placeholder()
            self.midi_status.set(f"Unable to display album artwork: {error}")
            return

        self.artwork_canvas.create_image(
            48, 48, image=self.album_artwork, anchor="center"
        )

    def _draw_album_artwork_placeholder(self) -> None:
        self.artwork_canvas.create_text(
            48,
            48,
            text="ALBUM ART",
            fill="#9aa9bd",
            font=("TkDefaultFont", 8),
        )

    def _copy_now_playing(self, _event: tk.Event) -> str:
        title = self.now_playing.get()
        if title:
            self.root.clipboard_clear()
            self.root.clipboard_append(title)
            self.midi_status.set("Copied song title to clipboard.")
        return "break"

    def _update_percent_label(self, _value: str) -> None:
        amount = self.remove_percent.get()
        self.removal_amount = amount / 100.0
        self.percent_label.configure(text=f"{amount}%")

    def _update_vocal_percent_label(self, _value: str) -> None:
        amount = self.remove_vocals_percent.get()
        self.vocal_removal_amount = amount / 100.0
        self.vocal_percent_label.configure(text=f"{amount}%")

    def _update_bass_percent_label(self, _value: str) -> None:
        amount = self.remove_bass_percent.get()
        self.bass_removal_amount = amount / 100.0
        self.bass_percent_label.configure(text=f"{amount}%")

    def _update_drums_percent_label(self, _value: str) -> None:
        amount = self.remove_drums_percent.get()
        self.drums_removal_amount = amount / 100.0
        self.drums_percent_label.configure(text=f"{amount}%")

    def _update_volume(self, value: str) -> None:
        amount = int(round(float(value)))
        self.playback_gain = amount / 100.0
        self.volume_label.configure(text=f"{amount}%")

    def refresh_devices(self) -> None:
        try:
            devices = sd.query_devices()
        except sd.PortAudioError as error:
            messagebox.showerror(
                "Audio devices unavailable", str(error), parent=self.root
            )
            return

        self.input_devices.clear()
        self.output_devices.clear()
        for index, device in enumerate(devices):
            name = device["name"]
            if device["max_input_channels"] >= CHANNELS:
                self.input_devices[f"{index}: {name}"] = index
            if device["max_output_channels"] >= CHANNELS:
                self.output_devices[f"{index}: {name}"] = index

        self.input_choice["values"] = tuple(self.input_devices)
        self.output_choice["values"] = tuple(self.output_devices)
        self.metronome_output_devices = {
            name: index
            for name, index in self.output_devices.items()
            if "blackhole" not in name.lower()
        }
        self.metronome_output_choice["values"] = tuple(
            self.metronome_output_devices
        )
        if self.input_devices:
            blackhole = next(
                (
                    name
                    for name in self.input_devices
                    if "blackhole 2ch" in name.lower()
                ),
                next(
                    (
                        name
                        for name in self.input_devices
                        if "blackhole" in name.lower()
                    ),
                    next(iter(self.input_devices)),
                ),
            )
            self.input_choice.set(blackhole)
        if self.output_devices:
            headphones = next(
                (
                    name
                    for name in self.output_devices
                    if any(
                        keyword in name.lower()
                        for keyword in ("headphones", "headphone", "airpods", "headset")
                    )
                ),
                None,
            )
            default_output = sd.default.device[1]
            default_name = next(
                (
                    name
                    for name, index in self.output_devices.items()
                    if index == default_output
                ),
                next(iter(self.output_devices)),
            )
            self.output_choice.set(headphones or default_name)
        selected_metronome_output = self.metronome_output_device.get()
        if selected_metronome_output not in self.metronome_output_devices:
            selected_metronome_output = self.output_choice.get()
        if selected_metronome_output in self.metronome_output_devices:
            self.metronome_output_choice.set(selected_metronome_output)
        blackhole_output_available = any(
            "blackhole" in name.lower() for name in self.output_devices
        )
        self.metronome_blackhole_check.configure(
            text=(
                "Also play through BlackHole"
                if blackhole_output_available
                else "BlackHole output unavailable"
            )
        )
        self.metronome_blackhole_check.configure(
            state="normal" if blackhole_output_available else "disabled"
        )
        if not blackhole_output_available:
            self.metronome_blackhole_enabled.set(False)
        if not self.input_devices or not self.output_devices:
            self.status.set(
                "No stereo input or output devices found. Check macOS audio settings."
            )

    def start(self) -> None:
        input_name = self.input_choice.get()
        output_name = self.output_choice.get()
        if (
            input_name not in self.input_devices
            or output_name not in self.output_devices
        ):
            messagebox.showerror(
                "Choose audio devices",
                "Select a stereo input and output device before starting.",
                parent=self.root,
            )
            return

        self.stop_event.clear()
        self.stream_active.clear()
        self.processing_active.clear()
        self.input_overflow.clear()
        with self.meter_lock:
            self.input_level = 0.0
            self.output_level = 0.0
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=4)
        self.status.set("Loading the separation model (first run downloads it)...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker = threading.Thread(
            target=self._run_audio,
            args=(
                self.input_devices[input_name],
                self.output_devices[output_name],
                *QUALITY_SETTINGS[self.quality_choice.get()],
            ),
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping after the current audio window...")

    def start_midi(self) -> None:
        midi_port = self.midi_port.get().strip()
        if not midi_port:
            messagebox.showerror(
                "Choose a MIDI port",
                "Enter the MIDI output port connected to AmpliTube.",
                parent=self.root,
            )
            return

        self.midi_stop_event.clear()
        self.midi_status.set("Connecting to Spotify...")
        self.midi_start_button.configure(state="disabled")
        self.midi_stop_button.configure(state="normal")
        self.midi_worker = threading.Thread(
            target=self._run_spotify_midi,
            args=(midi_port,),
            daemon=True,
        )
        self.midi_worker.start()

    def stop_midi(self) -> None:
        self.midi_stop_event.set()
        self.midi_stop_button.configure(state="disabled")
        self.midi_status.set("Stopping Spotify MIDI monitor...")

    def _run_spotify_midi(self, midi_port: str) -> None:
        try:
            run_spotify_midi(
                midi_port,
                DEFAULT_CONFIG_PATH,
                self.midi_stop_event,
                lambda text: self.midi_status_queue.put(("status", text)),
                lambda title, artist: self.midi_status_queue.put(
                    ("track", (title, artist))
                ),
                lambda artwork: self.midi_status_queue.put(("artwork", artwork)),
            )
        except Exception as error:
            self.midi_status_queue.put(("status", f"Error: {error}"))
        finally:
            self.midi_status_queue.put(("finished", ""))

    def _run_audio(
        self,
        input_device: int,
        output_device: int,
        shifts: int,
        overlap: float,
    ) -> None:
        try:
            self._process_audio(input_device, output_device, shifts, overlap)
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            sd.PortAudioError,
            queue.Full,
        ) as error:
            self._notify(f"Error: {error}")
        else:
            self._notify("Stopped. Choose devices and start to play again.")
        finally:
            self.stream_active.clear()
            self.processing_active.clear()

    def _process_audio(
        self,
        input_device: int,
        output_device: int,
        shifts: int,
        overlap: float,
    ) -> None:
        model = get_model(name="htdemucs_6s")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            model.to(device)
        except RuntimeError:
            if device.type != "mps":
                raise
            device = torch.device("cpu")
            model.to(device)
        model.eval()
        source_indices: dict[str, int] = {}
        for source in ("guitar", "vocals", "bass", "drums"):
            try:
                source_indices[source] = model.sources.index(source)
            except ValueError as error:
                raise RuntimeError(
                    f"The selected Demucs model has no {source} stem."
                ) from error

        self._notify("Model loaded. Buffering audio...")
        callback_state: dict[str, np.ndarray | int | None] = {
            "current": None,
            "offset": 0,
        }

        def audio_callback(indata, outdata, frames, _time_info, status) -> None:
            if status and (status.input_overflow or status.output_underflow):
                self.input_overflow.set()
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                self.input_overflow.set()

            input_rms = float(np.sqrt(np.mean(np.square(indata))))
            outdata.fill(0)
            written = 0
            while written < frames:
                current = callback_state["current"]
                offset = int(callback_state["offset"] or 0)
                if current is None or offset >= len(current):
                    try:
                        current = self.output_queue.get_nowait()
                    except queue.Empty:
                        callback_state["current"] = None
                        break
                    callback_state["current"] = current
                    callback_state["offset"] = 0
                    offset = 0

                count = min(frames - written, len(current) - offset)
                outdata[written : written + count] = current[offset : offset + count]
                callback_state["offset"] = offset + count
                written += count

            output_rms = float(np.sqrt(np.mean(np.square(outdata))))
            with self.meter_lock:
                self.input_level = input_rms
                self.output_level = output_rms

        window_frames = SAMPLE_RATE * WINDOW_SECONDS
        hop_frames = SAMPLE_RATE * HOP_SECONDS
        chunks: list[np.ndarray] = []
        buffered_frames = 0
        previous_tail: np.ndarray | None = None

        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=(input_device, output_device),
            channels=(CHANNELS, CHANNELS),
            dtype="float32",
            callback=audio_callback,
        ):
            self.stream_active.set()
            self._notify("Running. Play Spotify through BlackHole.")
            while not self.stop_event.is_set():
                if self.input_overflow.is_set():
                    raise RuntimeError(
                        "Audio processing fell behind. Stop playback and try again."
                    )
                try:
                    chunk = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                chunks.append(chunk)
                buffered_frames += len(chunk)
                if buffered_frames < window_frames:
                    continue

                audio = np.concatenate(chunks, axis=0)
                window = audio[:window_frames]
                chunks = [audio[hop_frames:]]
                buffered_frames = len(chunks[0])

                tensor = torch.from_numpy(window.T.copy()).unsqueeze(0).to(device)
                self.processing_active.set()
                try:
                    try:
                        with torch.no_grad():
                            stems = apply_model(
                                model,
                                tensor,
                                device=device,
                                shifts=shifts,
                                split=True,
                                overlap=overlap,
                                progress=False,
                            )
                    except RuntimeError:
                        if device.type != "mps":
                            raise
                        device = torch.device("cpu")
                        model.to(device)
                        tensor = torch.from_numpy(window.T.copy()).unsqueeze(0)
                        with torch.no_grad():
                            stems = apply_model(
                                model,
                                tensor,
                                device=device,
                                shifts=shifts,
                                split=True,
                                overlap=overlap,
                                progress=False,
                            )
                finally:
                    self.processing_active.clear()
                guitar = stems[0, source_indices["guitar"]].T.cpu().numpy()
                vocals = stems[0, source_indices["vocals"]].T.cpu().numpy()
                bass = stems[0, source_indices["bass"]].T.cpu().numpy()
                drums = stems[0, source_indices["drums"]].T.cpu().numpy()
                result = (
                    window
                    - self.removal_amount * guitar
                    - self.vocal_removal_amount * vocals
                    - self.bass_removal_amount * bass
                    - self.drums_removal_amount * drums
                )
                result = apply_playback_gain(result, self.playback_gain)

                if previous_tail is None:
                    output = result[:hop_frames]
                else:
                    ramp = np.linspace(0.0, 1.0, hop_frames, dtype=np.float32)[:, None]
                    output = previous_tail * (1.0 - ramp) + result[:hop_frames] * ramp
                previous_tail = result[hop_frames:]
                self.output_queue.put(
                    output.astype(np.float32, copy=False), timeout=1.0
                )

    def _notify(self, text: str) -> None:
        self.status_queue.put(text)

    def _poll_status(self) -> None:
        while True:
            try:
                self.status.set(self.status_queue.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                message_type, text = self.midi_status_queue.get_nowait()
            except queue.Empty:
                break
            if message_type == "status":
                self.midi_status.set(str(text))
            elif message_type == "track":
                if (
                    isinstance(text, tuple)
                    and len(text) == 2
                    and isinstance(text[0], str)
                    and isinstance(text[1], str)
                ):
                    title, artist = text
                    self.now_playing.set(title)
                    try:
                        config = load_config(DEFAULT_CONFIG_PATH)
                    except (OSError, ValueError) as error:
                        self.midi_status.set(
                            f"Unable to check song configuration; history not "
                            f"recorded: {error}"
                        )
                    else:
                        if is_song_configured(title, config):
                            try:
                                self.usage_log.record_song(title, artist)
                            except OSError as error:
                                self.midi_status.set(
                                    f"Usage log write failed: {error}"
                                )
                        else:
                            self.midi_status.set(
                                f"{title} is not in config.json; not added to history."
                            )
                    self._refresh_weekly_summary()
            elif message_type == "artwork":
                if isinstance(text, bytes) or text is None:
                    self._show_album_artwork(text)
            elif message_type == "finished":
                try:
                    self.usage_log.record_song_stop()
                except OSError as error:
                    self.midi_status.set(f"Usage log write failed: {error}")
                self._refresh_weekly_summary()
                self.midi_start_button.configure(state="normal")
                self.midi_stop_button.configure(state="disabled")
            else:
                self.midi_start_button.configure(state="normal")
                self.midi_stop_button.configure(state="disabled")
        if self.worker is not None and not self.worker.is_alive():
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if self.status.get().startswith(
                ("Loading", "Model loaded", "Running", "Stopping")
            ):
                self.status.set(
                    "Processing stopped unexpectedly. Check the terminal for details."
                )
        with self.meter_lock:
            input_level = self.input_level
            output_level = self.output_level
        if self.stream_active.is_set():
            if self.processing_active.is_set():
                self.status.set("AI analyzing the next audio window...")
            elif output_level > 0.002:
                self.status.set(
                    "Playing processed audio; listening for the next window."
                )
            elif input_level > 0.002:
                self.status.set("Buffering Spotify audio for AI separation...")
            else:
                self.status.set("Waiting for Spotify audio through BlackHole...")
        now = time.monotonic()
        if now - self.last_usage_summary_update >= 60:
            self._refresh_weekly_summary()
        if now - self.last_cpu_update >= 1.0:
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(f"System CPU usage: {cpu_percent:.0f}%")
            self.last_cpu_update = now
        self._draw_visualizer(input_level, output_level)
        self.root.after(100, self._poll_status)

    def _draw_visualizer(self, input_level: float, output_level: float) -> None:
        canvas = self.visualizer
        width = max(canvas.winfo_width(), 360)
        canvas.delete("all")
        canvas.create_text(
            8, 18, text="INPUT", anchor="w", fill="#9aa9bd", font=("TkDefaultFont", 7)
        )
        canvas.create_text(
            8,
            56,
            text="OUTPUT",
            anchor="w",
            fill="#9aa9bd",
            font=("TkDefaultFont", 7),
        )
        canvas.create_line(70, 38, width - 8, 38, fill="#293545")

        bars = 24
        left, right = 76, width - 10
        gap = 3
        bar_width = max(2, (right - left - gap * (bars - 1)) / bars)
        input_strength = min(1.0, input_level * 8.0)
        output_strength = min(1.0, output_level * 8.0)
        analyzing = self.processing_active.is_set()
        self.animation_frame += 1

        for index in range(bars):
            phase = self.animation_frame * 0.32 - index * 0.72
            variation = 0.35 + 0.65 * abs(np.sin(phase))
            input_height = max(2, input_strength * variation * 30)
            output_height = max(2, output_strength * variation * 30)
            if analyzing:
                pulse = max(0.0, np.sin(phase + self.animation_frame * 0.12))
                input_height = max(input_height, 3 + pulse * 8)

            x0 = left + index * (bar_width + gap)
            x1 = x0 + bar_width
            canvas.create_rectangle(
                x0,
                25 - input_height / 2,
                x1,
                25 + input_height / 2,
                fill="#45d6b2" if input_strength > 0.01 else "#28534f",
                outline="",
            )
            canvas.create_rectangle(
                x0,
                60 - output_height / 2,
                x1,
                60 + output_height / 2,
                fill="#72a7ff" if output_strength > 0.01 else "#30415d",
                outline="",
            )
        if analyzing:
            canvas.create_text(
                width - 10,
                7,
                text="AI WORKING",
                anchor="ne",
                fill="#ffca72",
                font=("TkDefaultFont", 7, "bold"),
            )

    def close(self) -> None:
        try:
            self.usage_log.close()
        except OSError as error:
            messagebox.showerror(
                "Unable to save usage log",
                f"The app could not record its close time in {LOG_PATH}:\n{error}",
                parent=self.root,
            )
            return
        self.stop_event.set()
        self.midi_stop_event.set()
        self._stop_metronome()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        GuitarRemoverApp(root)
    except (OSError, ValueError) as error:
        messagebox.showerror(
            "Unable to start usage logging",
            f"The app could not open its usage log:\n{error}",
            parent=root,
        )
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
