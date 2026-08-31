# Localizer

The most minimal localization pipeline. It globalizes logged `deviceMotion` with raw GNSS (GPS+GLONASS), stable calibration, and wheel speed. Runs a small KF/RTS smoother. Adjusts GPS fix delays.

## Use

```bash
uv sync 
uv run python -m localizer.run
```

The first run may download precise orbit and ionosphere data into Laika's cache.

Measured against the comma1M reference `localizer.safetensors` for the included segment:

| Metric | error |
| --- | ---: |
| Median 3D position  | `0.449 m` |
| Median absolute longitudinal | `0.396 m` |
| Median absolute lateral | `0.071 m` |
| Median velocity  | `0.117 m/s` |
