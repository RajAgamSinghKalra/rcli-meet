"""Audio capture helper for rcli-meet.

Two sources, selected by --source:
  loopback  WASAPI loopback on the default speaker -- what OTHERS say in the
            meeting (no virtual audio driver needed, native to Windows).
  mic       A real microphone -- what YOU say.

Downmixes to mono, resamples to 16kHz, and streams raw float32 PCM to stdout.

Windows/WASAPI note: recording with channels=1 is a known soundcard bug
(\"records garbage\"). We always open the device in its native channel count
(usually stereo) and downmix to mono ourselves.
"""
import argparse
import queue
import sys
import threading
import warnings

import numpy as np
import soundcard as sc

warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    category=RuntimeWarning,
)

SAMPLE_RATE = 16000
SAMPLES_PER_BLOCK = 800  # 50ms
MAX_QUEUED_BLOCKS = 300
FLUSH_EVERY_BLOCKS = 4
# If the first ~0.5s of mic audio is below this, the boom is almost certainly
# muted / privacy-blocked / wrong device -- warn loudly once.
SILENCE_RMS = 1e-5


def list_mics():
    return list(sc.all_microphones(include_loopback=False))


def resolve_mic(name_substr: str | None):
    """Pick a microphone by substring (case-insensitive), else the default."""
    mics = list_mics()
    if not mics:
        raise RuntimeError("no microphones found")

    if name_substr:
        needle = name_substr.strip().lower()
        matches = [m for m in mics if needle in m.name.lower()]
        if not matches:
            available = "\n".join(f"  - {m.name}" for m in mics)
            raise RuntimeError(
                f'no microphone matching "{name_substr}". Available:\n{available}'
            )
        # Prefer exact-ish / shorter names when multiple match (e.g. "Razer").
        matches.sort(key=lambda m: len(m.name))
        return matches[0]

    return sc.default_microphone()


def get_device(source, mic_name=None):
    if source == "loopback":
        speaker = sc.default_speaker()
        return sc.get_microphone(id=str(speaker.name), include_loopback=True), speaker.name
    if source == "mic":
        mic = resolve_mic(mic_name)
        return mic, mic.name
    raise ValueError(f"unknown source: {source}")


def downmix_to_mono(block: np.ndarray) -> np.ndarray:
    """block: (n,) or (n, ch) float32 → (n,) mono."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32, copy=False)
    # Mean across channels -- works for stereo headset mics.
    return block.mean(axis=1).astype(np.float32)


def record_into(q, mic, stop, channels, gain):
    with mic.recorder(
        samplerate=SAMPLE_RATE,
        channels=channels,
        blocksize=SAMPLES_PER_BLOCK,
    ) as rec:
        while not stop.is_set():
            block = rec.record(numframes=SAMPLES_PER_BLOCK)
            mono = downmix_to_mono(np.asarray(block))
            if gain != 1.0:
                mono = np.clip(mono * gain, -1.0, 1.0)
            try:
                q.put_nowait(mono)
            except queue.Full:
                drop = min(32, q.maxsize // 4)
                for _ in range(drop):
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                try:
                    q.put_nowait(mono)
                except queue.Full:
                    pass


def probe_rms(mic, channels, seconds=0.5) -> float:
    nblocks = max(1, int(seconds * SAMPLE_RATE / SAMPLES_PER_BLOCK))
    with mic.recorder(
        samplerate=SAMPLE_RATE,
        channels=channels,
        blocksize=SAMPLES_PER_BLOCK,
    ) as rec:
        chunks = [downmix_to_mono(np.asarray(rec.record(numframes=SAMPLES_PER_BLOCK))) for _ in range(nblocks)]
    audio = np.concatenate(chunks)
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["loopback", "mic"], required=False)
    parser.add_argument(
        "--device",
        default=None,
        help='Substring to select a mic (e.g. "Razer" or "Kraken"). Ignored for loopback.',
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Software gain multiplier applied after downmix (mic path).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print microphones and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        default = sc.default_microphone()
        for m in list_mics():
            mark = "  (default)" if m.name == default.name else ""
            print(f"{m.name}{mark}", file=sys.stderr)
        return

    if not args.source:
        print("error: --source is required (unless --list-devices)", file=sys.stderr)
        sys.exit(2)

    device, name = get_device(args.source, mic_name=args.device)
    # Native channel count (Razer Kraken reports 2). Never open as channels=1
    # on WASAPI -- soundcard docs: single-channel capture is broken/garbage.
    channels = getattr(device, "channels", None) or 2
    if isinstance(channels, int) and channels < 1:
        channels = 2

    print(f"on: {name} (channels={channels})", file=sys.stderr, flush=True)

    if args.source == "mic":
        try:
            rms = probe_rms(device, channels)
            print(f"mic-level rms={rms:.6f}", file=sys.stderr, flush=True)
            if rms < SILENCE_RMS:
                print(
                    "WARNING: microphone is returning silence (all zeros).\n"
                    "  Check:\n"
                    "  1) Razer Kraken mute dial / boom-mic raised (hardware mute)\n"
                    "  2) Razer Synapse mic mute / volume\n"
                    "  3) Windows Settings → System → Sound → Input — unmute & raise volume\n"
                    "  4) Windows Settings → Privacy → Microphone — allow desktop apps\n"
                    "  List devices:  python capture_audio.py --list-devices\n"
                    "  Pick one:      set RCLI_MEET_MIC=Razer",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as err:
            print(f"mic probe failed: {err}", file=sys.stderr, flush=True)

    gain = args.gain if args.source == "mic" else 1.0
    q = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
    stop = threading.Event()
    thread = threading.Thread(
        target=record_into,
        args=(q, device, stop, channels, gain),
        daemon=True,
        name=f"capture-{args.source}",
    )
    thread.start()

    stdout = sys.stdout.buffer
    pending_flush = 0
    try:
        while True:
            block = q.get()
            stdout.write(np.asarray(block, dtype=np.float32).tobytes())
            pending_flush += 1
            if pending_flush >= FLUSH_EVERY_BLOCKS or q.empty():
                stdout.flush()
                pending_flush = 0
    finally:
        stop.set()
        try:
            stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, OSError):
        pass
    except Exception as err:
        print(f"fatal: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
