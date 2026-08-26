# Localizer

The most minimal localization pipeline. It globalizes logged `deviceMotion` with external GPS, stable calibration, and wheel speed. Runs a small KF/RTS smoother. Adjusts GPS fix delays.

```text
GPS p/v (measurement time) ----.
deviceMotion velocity + wheel speed +-> 6-state p/v KF -> RTS -> ECEF p/v --------.
                                                                                |
deviceMotion orientation + calibration + GPS course -> yaw -> ECEF quaternion ------+-> resample -> Localizer arrays
deviceMotion angular rate + acceleration -----------> passthrough ------------------'
```

## Use

```bash
uv sync --dev
uv run python -m localizer.run
```

Measured against [TODO]


| Step | Error without -> with |
| --- | --- |
| Rolling GPS correction | position: `7.968 -> 1.842 m`; velocity: `0.344 -> 0.126 m/s` |
| GPS fix timestamp | position: `2.854 -> 1.842 m`; velocity: `0.135 -> 0.126 m/s` |
| Wheel-speed forward correction | position: `1.902 -> 1.842 m`; velocity: `0.232 -> 0.126 m/s` |
| RTS backward pass | position: `2.426 -> 1.842 m`; velocity: `0.374 -> 0.126 m/s` |
| `0.3 m/s` correlated-velocity noise floor | position: `2.319 -> 1.842 m`; velocity: `0.198 -> 0.126 m/s` |
| GPS velocity updates | position: `1.850 -> 1.842 m`; velocity: `0.141 -> 0.126 m/s` |
