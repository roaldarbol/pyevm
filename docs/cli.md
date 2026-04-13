# CLI Reference

The `pyevm` command-line tool provides three magnification commands plus utilities.

## pyevm color

Amplifies subtle colour changes (e.g. skin-tone flush from pulse).

```bash
pyevm color INPUT OUTPUT [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--alpha` | `50.0` | Luminance amplification factor |
| `--freq-low` | `0.4` | Lower bandpass frequency (Hz) |
| `--freq-high` | `3.0` | Upper bandpass frequency (Hz) |
| `--n-levels` | `6` | Gaussian pyramid levels |
| `--chrom-attenuation` | `0.1` | Chrominance attenuation (0–1) |
| `--pyramid-level` | auto | Pyramid level to filter |
| `--filter` | `ideal` | Filter type (`ideal` or `butterworth`) |
| `--notch` | — | Notch frequency to exclude (repeatable) |
| `--notch-width` | `1.0` | Width of each notch (Hz) |
| `--chunk-size` | `64` | Frames per GPU batch |
| `--max-frames` | — | Limit frames read |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--debug` | off | Verbose logging |

## pyevm motion

Amplifies subtle physical motion (e.g. breathing, structural vibration).

```bash
pyevm motion INPUT OUTPUT [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--alpha` | `20.0` | Amplification factor |
| `--freq-low` | `0.4` | Lower bandpass frequency (Hz) |
| `--freq-high` | `3.0` | Upper bandpass frequency (Hz) |
| `--n-levels` | `6` | Laplacian pyramid levels |
| `--lambda-c` | `16.0` | Spatial wavelength cutoff (px) |
| `--filter` | `butterworth` | Filter type (`butterworth` or `ideal`) |
| `--notch` | — | Notch frequency to exclude (repeatable) |
| `--notch-width` | `1.0` | Width of each notch (Hz) |
| `--chunk-size` | `64` | Frames per GPU batch |
| `--max-frames` | — | Limit frames read |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--debug` | off | Verbose logging |

## pyevm phase

Artifact-free motion magnification via steerable pyramid phase decomposition.

```bash
pyevm phase INPUT OUTPUT [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--factor` | `10.0` | Phase amplification factor |
| `--freq-low` | `0.4` | Lower bandpass frequency (Hz) |
| `--freq-high` | `3.0` | Upper bandpass frequency (Hz) |
| `--n-scales` | `6` | Pyramid scales |
| `--n-orientations` | `8` | Orientation bands per scale |
| `--sigma` | `0.0` | Spatial phase smoothing (0 = off) |
| `--filter` | `ideal` | Filter type for batch mode |
| `--attenuate` | off | Attenuate large motions instead of amplifying |
| `--attenuate-mag` | `π` | Attenuation threshold (radians) |
| `--notch` | — | Notch frequency to exclude (repeatable) |
| `--notch-width` | `1.0` | Width of each notch (Hz) |
| `--chunk-size` | `64` | Frames per GPU batch |
| `--max-frames` | — | Limit frames read |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--debug` | off | Verbose logging |

## pyevm dashboard

Opens the interactive Streamlit dashboard in a browser.

```bash
pyevm dashboard [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max-upload-size` | `5000` | Maximum video upload size (MB) |
| `--debug` | off | Enable verbose DEBUG logging |

## pyevm info

Prints detected compute device, PyTorch version, and GPU details.

```bash
pyevm info
```
