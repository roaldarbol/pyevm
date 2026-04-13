# Usage

## Python API

### Batch mode

Loads all frames into memory at once. Best for short clips or when you need random access.

```python
from pyevm import ColorMagnifier, MotionMagnifier, PhaseMagnifier
from pyevm.io.video import VideoReader, VideoWriter

reader = VideoReader("input.mp4")
frames, fps = reader.read()

# Colour magnification — amplifies pulse/blood-flow
magnifier = ColorMagnifier(alpha=50, freq_low=0.4, freq_high=3.0)
result = magnifier.process(frames, fps)

VideoWriter("output.mp4", fps=fps).write(result)
```

### Streaming mode

Processes video in chunks with O(1) memory. Recommended for long or high-resolution video.

```python
from pyevm import MotionMagnifier
from pyevm.io.video import VideoReader, VideoWriter

reader = VideoReader("input.mp4")
magnifier = MotionMagnifier(alpha=20, freq_low=0.4, freq_high=3.0)

frame_stream, fps, n_frames = reader.stream()
output_stream = magnifier.process_stream(frame_stream, fps, n_frames=n_frames)

VideoWriter("output.mp4", fps=fps).write_stream(output_stream)
```

### Choosing an algorithm

| Algorithm | Class | Best for |
|-----------|-------|----------|
| Colour | `ColorMagnifier` | Pulse detection, blood-flow visualisation |
| Motion | `MotionMagnifier` | Breathing, structural vibration |
| Phase | `PhaseMagnifier` | Artifact-free motion magnification |

### Notch filters

Exclude specific frequencies (e.g. 50 Hz fluorescent light flicker):

```python
magnifier = ColorMagnifier(
    alpha=50, freq_low=0.4, freq_high=3.0,
    notch_freqs=[50.0], notch_width=2.0,
)
```

### Large-motion attenuation (phase only)

Suppress camera shake while preserving subtle motion:

```python
import math
from pyevm import PhaseMagnifier

magnifier = PhaseMagnifier(
    factor=10, freq_low=0.4, freq_high=3.0,
    attenuate_motion=True,
    attenuate_mag=math.pi,  # threshold in radians
)
```

---

## CLI

### Colour magnification

```bash
pyevm color input.mp4 output.mp4 --alpha 50 --freq-low 0.4 --freq-high 3.0
```

### Motion magnification

```bash
pyevm motion input.mp4 output.mp4 --alpha 20 --freq-low 0.4 --freq-high 3.0
```

### Phase-based magnification

```bash
pyevm phase input.mp4 output.mp4 --factor 10 --freq-low 0.4 --freq-high 3.0
```

### Exclude a frequency (notch filter)

```bash
# Remove 50 Hz light flicker
pyevm color input.mp4 output.mp4 --notch 50.0 --notch-width 2.0
```

### Attenuate large motions

```bash
pyevm phase input.mp4 output.mp4 --factor 10 --freq-low 0.4 --freq-high 3.0 --attenuate
```

### Inspect compute device

```bash
pyevm info
```

### Interactive dashboard

```bash
pyevm dashboard
```

Add `--debug` to any command for verbose per-chunk timing output.

---

## Tips

- **Frequency range**: set `--freq-low` and `--freq-high` to the expected signal band. For a human heartbeat at rest, 0.4–3.0 Hz (24–180 bpm) is a good starting point.
- **Nyquist limit**: frequencies must be below `fps / 2`. A 30 fps video supports at most 15 Hz.
- **Alpha / factor**: start low and increase until you see the effect without visible artefacts. Values that are too high produce ringing or colour bleed.
- **Chunk size**: reduce `--chunk-size` if you run out of VRAM. The default (64 frames) uses ~10 GB at 1080p.
