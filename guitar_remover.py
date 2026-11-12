#!/usr/bin/env python3
"""Play Spotify through a stem-reduction model using BlackHole as input."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import psutil
import sounddevice as sd
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


SAMPLE_RATE = 44_100
CHANNELS = 2
WINDOW_SECONDS = 6
HOP_SECONDS = 3
BLOCK_SIZE = 1_024
QUALITY_SETTINGS = {
    "Balanced": (1, 0.25),
    "Higher quality (slower)": (2, 0.5),
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


class GuitarRemoverApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Spotify Stem Remover")
        self.root.minsize(500, 640)

        self.input_devices: dict[str, int] = {}
        self.output_devices: dict[str, int] = {}
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
        self.input_overflow = threading.Event()
        self.worker: threading.Thread | None = None

        self._build_ui()
        self.refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll_status)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Spotify input").grid(row=0, column=0, sticky="w", pady=6)
        self.input_choice = ttk.Combobox(frame, state="readonly")
        self.input_choice.grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=6)

        ttk.Label(frame, text="Headphones / output").grid(
            row=1, column=0, sticky="w", pady=6
        )
        self.output_choice = ttk.Combobox(frame, state="readonly")
        self.output_choice.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=6)

        ttk.Button(frame, text="Refresh devices", command=self.refresh_devices).grid(
            row=2, column=1, sticky="e", pady=(2, 12)
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
            row=8, column=0, sticky="w", pady=6
        )
        self.quality_choice = ttk.Combobox(
            frame,
            state="readonly",
            values=tuple(QUALITY_SETTINGS),
            width=24,
        )
        self.quality_choice.set("Balanced")
        self.quality_choice.grid(row=8, column=1, sticky="w", padx=(12, 0), pady=6)

        visualizer = ttk.LabelFrame(frame, text="Live audio activity", padding=8)
        visualizer.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        visualizer.columnconfigure(0, weight=1)
        self.visualizer = tk.Canvas(
            visualizer,
            height=112,
            background="#101722",
            highlightthickness=0,
        )
        self.visualizer.grid(row=0, column=0, sticky="ew")

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=10, column=0, sticky="w", pady=(12, 8))
        self.stop_button = ttk.Button(
            frame, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=10, column=1, sticky="e", pady=(12, 8))

        ttk.Label(
            frame,
            text="Uses a 6-second AI audio window, so expect several seconds of delay.",
            wraplength=430,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 8))
        ttk.Label(frame, textvariable=self.status, wraplength=430).grid(
            row=12, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, textvariable=self.cpu_usage).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

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
            8, 24, text="INPUT", anchor="w", fill="#9aa9bd", font=("TkDefaultFont", 8)
        )
        canvas.create_text(
            8,
            76,
            text="OUTPUT",
            anchor="w",
            fill="#9aa9bd",
            font=("TkDefaultFont", 8),
        )
        canvas.create_line(70, 50, width - 8, 50, fill="#293545")

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
                36 - input_height / 2,
                x1,
                36 + input_height / 2,
                fill="#45d6b2" if input_strength > 0.01 else "#28534f",
                outline="",
            )
            canvas.create_rectangle(
                x0,
                88 - output_height / 2,
                x1,
                88 + output_height / 2,
                fill="#72a7ff" if output_strength > 0.01 else "#30415d",
                outline="",
            )
        if analyzing:
            canvas.create_text(
                width - 10,
                10,
                text="AI WORKING",
                anchor="ne",
                fill="#ffca72",
                font=("TkDefaultFont", 8, "bold"),
            )

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GuitarRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
