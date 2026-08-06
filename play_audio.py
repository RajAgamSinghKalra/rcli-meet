"""Play raw float32 PCM (mono) from stdin through the default speaker.

Usage: play_audio.py <sample_rate>
"""
import sys

import numpy as np
import soundcard as sc


def main():
    sample_rate = int(sys.argv[1])
    raw = sys.stdin.buffer.read()
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return
    speaker = sc.default_speaker()
    speaker.play(samples, samplerate=sample_rate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
