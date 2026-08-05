"""WASAPI loopback capture helper for rcli-meet.

Captures whatever the default output device is currently playing (no virtual
audio driver needed -- WASAPI loopback is native to Windows), downmixes to
mono, resamples to 16kHz, and streams raw float32 PCM samples to stdout for
the Node process to consume. One frame = SAMPLES_PER_BLOCK * 4 bytes.
"""
import sys

import soundcard as sc

SAMPLE_RATE = 16000
SAMPLES_PER_BLOCK = 1600  # 100ms per block


def main():
    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    print(f"[capture] loopback on: {speaker.name}", file=sys.stderr, flush=True)

    stdout = sys.stdout.buffer
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=SAMPLES_PER_BLOCK) as rec:
        while True:
            block = rec.record(numframes=SAMPLES_PER_BLOCK)
            stdout.write(block.reshape(-1).tobytes())
            stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
