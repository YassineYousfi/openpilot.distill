import laika.raw_gnss as raw
import numpy as np
from laika.astro_dog import AstroDog
from laika.ephemeris import EphemerisType
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, get_DOP

GPS_WEEK_SECONDS = 604_800.0
GPS_MAX_PRN = 32
GLONASS_MIN_PRN = 65
GLONASS_MAX_PRN = 96
STATE_SIZE = 11
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
CLOCK_BIAS = 6
CLOCK_DRIFT = 7
CLOCK_ACCELERATION = 8
GLONASS_BIAS = 9
GLONASS_FREQ_SLOPE = 10


def _gnss_epochs(raw_t: np.ndarray, raw_measurements: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
  raw_t = np.asarray(raw_t, dtype=np.float64)
  raw_measurements = np.asarray(raw_measurements, dtype=np.float64)
  if raw_measurements.ndim != 2 or raw_measurements.shape[1] != 10 or len(raw_t) != len(raw_measurements):
    raise RuntimeError("raw UBlox measurements have an invalid shape")

  prn = raw_measurements[:, GNSSMeasurement.PRN]
  gps = prn <= GPS_MAX_PRN
  glonass = (prn >= GLONASS_MIN_PRN) & (prn <= GLONASS_MAX_PRN)
  measurements = raw_measurements[gps | glonass]
  log_t = raw_t[gps | glonass]
  if not np.any(gps):
    raise RuntimeError("no raw GPS measurements found")

  receiver_t = (
    measurements[:, GNSSMeasurement.RECV_TIME_WEEK] * GPS_WEEK_SECONDS
    + measurements[:, GNSSMeasurement.RECV_TIME_SEC]
  )
  order = np.argsort(receiver_t, kind="stable")
  receiver_t, log_t, measurements = receiver_t[order], log_t[order], measurements[order]
  epoch_receiver_t, starts = np.unique(receiver_t, return_index=True)
  ends = np.r_[starts[1:], len(measurements)]
  epochs = [measurements[start:end] for start, end in zip(starts, ends, strict=True)]

  # Receiver time is the measurement time; logMonoTime can include delivery jitter.
  epoch_t = log_t[starts[0]] + epoch_receiver_t - epoch_receiver_t[0]
  if len(epoch_t) < 2 or np.any(np.diff(epoch_t) <= 0.0):
    raise RuntimeError("raw GNSS timestamps must be strictly increasing")
  return epoch_t, epochs


def _normal_measurements(rows: np.ndarray) -> list[GNSSMeasurement]:
  return [raw.normal_meas_from_array(row) for row in rows]


def _correct_epochs(dog: AstroDog, initial_position: np.ndarray, epochs: list[np.ndarray]) -> list[np.ndarray]:
  corrected_epochs = []
  for rows in epochs:
    processed = raw.process_measurements(_normal_measurements(rows), dog)
    corrected = raw.correct_measurements(processed, initial_position, dog)
    values = np.asarray([measurement.as_array() for measurement in corrected], dtype=np.float64).reshape(-1, 14)
    corrected_epochs.append(values)
  if not any(len(epoch) >= 4 for epoch in corrected_epochs):
    raise RuntimeError("fewer than four corrected GPS satellites are available")
  return corrected_epochs


def _update(
  state: np.ndarray,
  covariance: np.ndarray,
  measurement: float,
  expected: float,
  observation: np.ndarray,
  variance: float,
) -> tuple[np.ndarray, np.ndarray]:
  innovation_variance = observation @ covariance @ observation + variance
  gain = covariance @ observation / innovation_variance
  state = state + gain * (measurement - expected)
  residual_projection = np.eye(STATE_SIZE) - np.outer(gain, observation)
  covariance = (
    residual_projection @ covariance @ residual_projection.T
    + np.outer(gain, gain) * variance
  )
  return state, (covariance + covariance.T) * 0.5


def _observe_epoch(
  state: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  for measurement in measurements:
    satellite_position = measurement[GNSSMeasurement.SAT_POS]
    delta = satellite_position - state[POSITION]
    distance = np.linalg.norm(delta)
    line_of_sight = delta / distance
    observation = np.zeros(STATE_SIZE)
    observation[POSITION] = -line_of_sight
    observation[CLOCK_BIAS] = 1.0
    expected = distance + state[CLOCK_BIAS]
    if GLONASS_MIN_PRN <= measurement[GNSSMeasurement.PRN] <= GLONASS_MAX_PRN:
      frequency = measurement[GNSSMeasurement.GLONASS_FREQ]
      observation[GLONASS_BIAS] = 1.0
      observation[GLONASS_FREQ_SLOPE] = frequency
      expected += state[GLONASS_BIAS] + frequency * state[GLONASS_FREQ_SLOPE]
    state, covariance = _update(
      state,
      covariance,
      measurement[GNSSMeasurement.PR],
      expected,
      observation,
      measurement[GNSSMeasurement.PR_STD] ** 2,
    )

  for measurement in measurements:
    satellite_position = measurement[GNSSMeasurement.SAT_POS]
    delta = satellite_position - state[POSITION]
    distance = np.linalg.norm(delta)
    line_of_sight = delta / distance
    relative_velocity = measurement[GNSSMeasurement.SAT_VEL] - state[VELOCITY]
    observation = np.zeros(STATE_SIZE)
    observation[POSITION] = -(
      (np.eye(3) - np.outer(line_of_sight, line_of_sight)) @ relative_velocity
    ) / distance
    observation[VELOCITY] = -line_of_sight
    observation[CLOCK_DRIFT] = 1.0
    state, covariance = _update(
      state,
      covariance,
      measurement[GNSSMeasurement.PRR],
      line_of_sight @ relative_velocity + state[CLOCK_DRIFT],
      observation,
      measurement[GNSSMeasurement.PRR_STD] ** 2,
    )
  return state, covariance


def _smooth_positions(
  epoch_t: np.ndarray, measurements: list[np.ndarray], initial_position: np.ndarray
) -> np.ndarray:
  state = np.zeros(STATE_SIZE)
  state[POSITION] = initial_position
  initial_std = np.array([1_000.0] * 3 + [10.0] * 3 + [1e7, 100.0, 0.2, 10.0, 1.0])
  covariance = np.diag(initial_std**2)
  process_std = np.array([0.03] * 3 + [3.0] * 3 + [100.0, 0.0, 0.005, 0.1, 0.01])
  process_noise = np.diag(process_std**2)

  count = len(epoch_t)
  filtered_states = np.empty((count, STATE_SIZE))
  filtered_covariances = np.empty((count, STATE_SIZE, STATE_SIZE))
  predicted_states = np.empty_like(filtered_states)
  predicted_covariances = np.empty_like(filtered_covariances)
  transitions = np.empty_like(filtered_covariances)

  previous_t = epoch_t[0]
  for index, (time, epoch) in enumerate(zip(epoch_t, measurements, strict=True)):
    dt = time - previous_t
    transition = np.eye(STATE_SIZE)
    transition[POSITION, VELOCITY] = np.eye(3) * dt
    transition[CLOCK_BIAS, CLOCK_DRIFT] = dt
    transition[CLOCK_DRIFT, CLOCK_ACCELERATION] = dt
    state = transition @ state
    covariance = transition @ covariance @ transition.T + process_noise * dt
    predicted_states[index] = state
    predicted_covariances[index] = covariance
    transitions[index] = transition

    state, covariance = _observe_epoch(state, covariance, epoch)
    filtered_states[index] = state
    filtered_covariances[index] = covariance
    previous_t = time

  smoothed_states = filtered_states.copy()
  for index in range(count - 2, -1, -1):
    smoothing_gain = np.linalg.solve(
      predicted_covariances[index + 1],
      transitions[index + 1] @ filtered_covariances[index].T,
    ).T
    smoothed_states[index] += smoothing_gain @ (
      smoothed_states[index + 1] - predicted_states[index + 1]
    )
  return smoothed_states


def localize_raw_gnss(
  raw_t: np.ndarray, raw_measurements: np.ndarray, initial_position: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return smoothed ECEF position and velocity at receiver epochs."""
  epoch_t, epochs = _gnss_epochs(raw_t, raw_measurements)
  initial_position = np.asarray(initial_position, dtype=np.float64)
  if initial_position.shape != (3,) or not np.all(np.isfinite(initial_position)):
    raise RuntimeError("the initial GPS position must be a finite ECEF vector")
  gps_epochs = [epoch[epoch[:, GNSSMeasurement.PRN] <= GPS_MAX_PRN] for epoch in epochs]
  dog_options = {
    "dgps": False,
    "valid_ephem_types": EphemerisType.all_orbits(),
    "default_delays": False,
  }
  dog = AstroDog(
    valid_const=(ConstellationId.GPS,),
    **dog_options,
  )
  corrected = _correct_epochs(dog, initial_position, gps_epochs)
  gps_dop = np.nanmedian([
    get_DOP(initial_position, epoch[:, GNSSMeasurement.SAT_POS])
    for epoch in corrected if len(epoch) >= 4
  ])
  gps_satellites = np.median([len(epoch) for epoch in corrected])
  has_glonass = any(np.any(epoch[:, GNSSMeasurement.PRN] >= GLONASS_MIN_PRN) for epoch in epochs)
  if has_glonass and (gps_dop > 3.0 or gps_satellites < 6):
    dog = AstroDog(
      valid_const=(ConstellationId.GPS, ConstellationId.GLONASS),
      **dog_options,
    )
    corrected = _correct_epochs(dog, initial_position, epochs)
  states = _smooth_positions(epoch_t, corrected, initial_position)
  return epoch_t, states[:, POSITION], states[:, VELOCITY]
