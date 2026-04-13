<p align="center">
  <img src="assets/logo.png" alt="pyevm logo" width="600"/>
</p>

#

Eulerian Video Magnification — reveal invisible motion and colour changes in video.

---

Eulerian Video Magnification (EVM) amplifies subtle, otherwise invisible variations in video —
such as the colour flush of a heartbeat, the micro-vibrations of a bridge, or the barely
perceptible breathing of a sleeping animal. pyevm provides clean, GPU-accelerated Python
implementations of the three canonical EVM algorithms, plus a CLI and an interactive Streamlit app.

## Algorithms

| Method | Best for | Reference |
|--------|----------|-----------|
| **Color** | Pulse detection, blood-flow visualisation | Wu et al. (2012) |
| **Motion** | Breathing, structural vibration | Wu et al. (2012) |
| **Phase** | Artifact-free motion magnification | Wadhwa et al. (2013) |

## Installation

```bash
# PyPI
pip install pyeulervid

# conda-forge
conda install -c conda-forge pyevm
```

See [Installation](installation.md) for GPU-accelerated video I/O options.

## Quick start

```python
from pyevm import ColorMagnifier, MotionMagnifier, PhaseMagnifier
from pyevm.io.video import VideoReader, VideoWriter

reader = VideoReader("input.mp4")
frames, fps = reader.read()

magnifier = ColorMagnifier(alpha=50, freq_low=0.4, freq_high=3.0)
result = magnifier.process(frames, fps)

VideoWriter("output.mp4", fps=fps).write(result)
```

Or from the CLI:

```bash
pyevm color input.mp4 output.mp4 --alpha 50 --freq-low 0.4 --freq-high 3.0
```

## References

- Wu, H.-Y. et al. (2012). **Eulerian Video Magnification for Revealing Subtle Changes in the World.** *ACM TOG*, 31(4).
- Wadhwa, N. et al. (2013). **Phase-Based Video Motion Processing.** *ACM TOG*, 32(4).
