"""WASAPI loopback capture helper for rcli-meet.

Captures whatever the default output device is currently playing (no virtual
audio driver needed -- WASAPI loopback is native to Windows), downmixes to
mono, resamples to 16kHz, and streams raw float32 PCM samples to stdout for
the Node process to consume.

The recorder runs on its own thread feeding a queue, because the consumer
(ONNX speech recognition) decodes synchronously and periodically stalls its
end of the pipe. When stdout blocked, the WASAPI recorder stopped being
drained and dropped audio -- surfacing as "data discontinuity in recording"
and silently losing words from the transcript.
"""
import queue
import sys
import threading

import soundcard as sc

SAMPLE_RATE = 16000
SAMPLES_PER_BLOCK = 1600  # 100ms per block
# Bounded so a permanently stalled consumer can't grow memory without limit;
# ~30s of audio is far more slack than any decode hiccup needs.
MAX_QUEUED_BLOCKS = 300


def record_into(q, mic, stop):
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=SAMPLES_PER_BLOCK) as rec:
        while not stop.is_set():
            block = rec.record(numframes=SAMPLES_PER_BLOCK)
            try:
                q.put_nowait(block)
            except queue.Full:
                # Consumer is badly behind. Drop the oldest block rather than
                # stalling the recorder, which is what causes WASAPI
                # discontinuities in the first place.
                try:
                    q.get_nowait()
                    q.put_nowait(block)
                except queue.Empty:
                    pass


def main():
    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    print(f"[capture] loopback on: {speaker.name}", file=sys.stderr, flush=True)

    q = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
    stop = threading.Event()
    thread = threading.Thread(target=record_into, args=(q, mic, stop), daemon=True)
    thread.start()

    stdout = sys.stdout.buffer
    try:
        while True:
            block = q.get()
            stdout.write(block.reshape(-1).tobytes())
            stdout.flush()
    finally:
        stop.set()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, OSError):
        # Parent went away (normal shutdown); exit quietly.
        pass
