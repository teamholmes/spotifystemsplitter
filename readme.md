# Spotify Stem Remover for macOS

This app listens to audio routed into **BlackHole**, estimates guitar, vocal,
bass, and drum stems with Demucs' six-stem AI model, and plays the mix back to a
selected output device. Set removal for each stem independently from 0%
(unchanged) to 100% (subtract the estimated stem). Vocal, bass, and drums
removal default to 0%, preserving the existing guitar-only behavior. This is
live processing with a buffer, not a zero-latency effect: expect several
seconds of delay, and sound quality depends on the recording and model estimate.

The same app can also monitor Spotify's current track and send MIDI program
changes to AmpliTube 5 and Tone3000 using the song mappings in `config.json`.
Audio separation and Spotify-to-MIDI monitoring have independent Start/Stop
controls.

## Requirements

- macOS, Spotify desktop app, and BlackHole 2ch installed.
- Python 3.11 recommended. The Python at `/opt/homebrew/bin/python3` on the
  setup machine is 3.14.6; the PyTorch/Demucs audio stack may not yet install
  cleanly on that version.
- Internet access on first launch to download the pretrained Demucs model.
- Headphones connected to the Mac. Do not select BlackHole as the app's output.

## Install

If Python 3.11 is not installed, install it with Homebrew:

```sh
brew install python@3.11
```

From this project folder, create an isolated environment and install the app:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch and model packages are large; allow time for installation. If
Homebrew installed Python 3.11 somewhere else, use that interpreter's path for
the `venv` command.

## Spotify-to-MIDI setup

1. Create a Spotify app in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and add `http://127.0.0.1:8888/callback` as a Redirect URI.
2. Copy `.env.example` to `.env` and add the app's client ID and client secret.
   Keep `.env` private; it is ignored by Git.
3. Configure an IAC virtual MIDI port in **Audio MIDI Setup** (Window >
   Show MIDI Studio > IAC Driver) and enable that port as an input in
   AmpliTube's MIDI preferences. Configure AmpliTube to map MIDI Program
   Changes to its presets. Enter the port name in the app, defaulting to
   `IAC Driver Bus 1`.
4. Edit `config.json` to set the playlist name and song-to-program mappings.
   AmpliTube program numbers are sent on MIDI channel 1, and Tone3000 program
   numbers on MIDI channel 6. Unmapped songs use the configured AmpliTube
   default and Tone3000 program 0.

Click **Start Spotify MIDI** in the app. On first run, Spotify opens a browser
window for authorization. The app checks the selected playlist and then changes
the patches when the currently playing track changes. Click **Stop MIDI** to
end monitoring. The **Song title** field shows the exact Spotify track name;
focus it and press **Ctrl+C** to copy the title for a `config.json` mapping.
The album artwork is displayed beside the title when Spotify provides it.

The app also appends each launch, graceful close, and Spotify track change
(including song, artist, and local timestamp) to `usage_log.jsonl` in this
folder. Each line is a separate JSON object, so the file can be read one event
at a time. The **This week's app usage** panel summarizes Monday through Sunday,
shows the five most-played songs this week, compares session time and track
changes with the previous week, and lists this week's track changes. The log is
excluded from Git and remains on this computer. A crash or forced termination
cannot write a close event, so only gracefully closed sessions have a recorded
end time.

Each track change is also posted to the configured ntfy topic with the song
title and artist. The topic is public to anyone who knows its name, so avoid
using it for private listening information. A notification/network error is
shown in the app but does not stop MIDI patch changes.

## Route Spotify audio through BlackHole

1. Open **Audio MIDI Setup** (in Applications > Utilities).
2. Confirm **BlackHole 2ch** is available. If Spotify also needs to be audible
   directly from the Mac speakers, create a **Multi-Output Device** containing
   BlackHole 2ch and the speakers/headphones, then select that device as
   Spotify's output. Otherwise, set Spotify's output to BlackHole 2ch.
3. Keep the app's output device set to your headphones or speakers. Using
   headphones avoids feeding the processed sound back into the microphone.
4. macOS may ask permission for the terminal or Python to access audio. Allow
   it in **System Settings > Privacy & Security > Microphone**.

Device labels and available routing options can vary between macOS versions.

## Run

With the virtual environment active:

```sh
python guitar_remover.py
```

You can also run `./run.sh`; it uses this folder's `.venv` when available.

1. The app defaults to **BlackHole 2ch** for Spotify input when available. It
   prefers an output device whose name identifies it as headphones, AirPods, or
   a headset; if none is detected, it uses macOS's default output. Confirm the
   selected devices before starting.
2. Set **Guitar removal**, **Vocal removal**, **Bass guitar removal**, and
   **Drums removal** independently to the desired percentages. Guitar removal
   defaults to 100%; the others default to 0%.
3. **Spotify volume** defaults to 180% and can be adjusted from 100% to 200%.
   This controls the app's output level, not Spotify's system-wide volume. A
   soft limiter helps control peaks at higher settings; lower the slider if
   you hear distortion.
4. The **CPU usage** readout shows the Mac's overall CPU utilization and
   refreshes once per second.
5. Start Spotify playback, then click **Start** in the app. Audio is analyzed
   in 6-second windows with a 3-second overlap. The first run
   downloads the `htdemucs_6s` model and can take a while.
6. Use **Stop** before changing routing. The initial audio buffer and model
   processing introduce several seconds of delay.
7. To also switch instrument patches automatically, configure Spotify API and
   MIDI as described above, then click **Start Spotify MIDI**. This monitor can
   run independently of audio separation.

At 100%, the app subtracts each selected AI-estimated stem from the stereo mix.
Separation is approximate: distorted guitar, dense arrangements, vocals mixed
with other sounds, and sounds similar to the selected stem can leave artifacts
or remove other audio. Lower percentages retain more of the selected stem. The
live activity display shows the incoming and processed signal levels; the AI
indicator pulses while a window is being analyzed.

For a cleaner separation, choose **Higher quality (slower)** or **Highest
quality (slowest)** before starting. These settings ask Demucs to average more
shifted analyses with increased overlap, which can reduce separation artifacts
at the cost of more processing time. Start with **Balanced** if playback becomes
choppy; not every Mac can sustain the higher-quality modes in real time. These
settings cannot fix every source of muddiness, and the results depend on the
song and how clearly its guitar, vocals, bass, and drums can be separated. If
100% removal sounds too hollow, try a lower percentage to retain some of the
original mix. If processing falls behind, the app reports an error; stop
playback and try Balanced.

## Troubleshooting

- **No BlackHole device:** check that BlackHole 2ch is installed, then click
  **Refresh devices**. If it was just installed, restarting macOS may be
  necessary.
- **Spotify is silent:** check that Spotify's output is routed to BlackHole
  (or a Multi-Output Device containing BlackHole), and that the app's output is
  set to headphones/speakers.
- **Spotify MIDI won't start:** check that `.env` contains the Spotify client
  ID and secret, the Redirect URI matches the Spotify Developer Dashboard, and
  `config.json` names a playlist available to your account.
- **AmpliTube patches do not change:** confirm that the selected MIDI port is
  enabled as an input in AmpliTube and its Program Change mappings are set up.
- **Permission or device error:** allow audio access for the terminal/Python in
  macOS Privacy & Security settings and verify the selected devices are still
  connected.
- **Installation fails on Python 3.14:** use Python 3.11 as described above.
- **Choppy or delayed playback:** close other demanding applications and retry.
  The model is computationally intensive, and not every Mac can keep up with
  live playback.
