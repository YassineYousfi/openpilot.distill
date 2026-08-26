from dataclasses import dataclass

import numpy as np

from .math import (
  circular_offset,
  ecef_to_geodetic,
  euler_to_rot,
  geodetic_to_ecef,
  interpolate,
  ned_from_ecef_matrix,
  nlerp_quaternions,
  rot_to_quat,
)

STATE_SIZE = 43
DEVICE_MOTION_VELOCITY_NOISE_FLOOR = 0.3  # m/s; deviceMotion samples are correlated filter outputs


@dataclass
class LocalizationInput:
  pose_t: np.ndarray
  orientation: np.ndarray
  velocity: np.ndarray
  velocity_std: np.ndarray
  acceleration_device: np.ndarray
  angular_velocity_device: np.ndarray
  gps_unix_ms: np.ndarray
  gps_geodetic: np.ndarray
  gps_velocity_ned: np.ndarray
  gps_position_std: np.ndarray
  clock: np.ndarray
  calibration: np.ndarray
  wide_from_device_euler: np.ndarray
  frame_t: np.ndarray
  car_state: np.ndarray


def _kalman_update(state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray,
                   observation: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  innovation_cov = observation @ covariance @ observation.T + noise
  gain = np.linalg.solve(innovation_cov, observation @ covariance).T
  state = state + gain @ (measurement - observation @ state)
  eye = np.eye(6)
  covariance = (eye - gain @ observation) @ covariance @ (eye - gain @ observation).T + gain @ noise @ gain.T
  return state, covariance


def _rts_trajectory(pose_t: np.ndarray, velocity: np.ndarray, velocity_covariance: np.ndarray,
                    gps_t: np.ndarray, gps_position: np.ndarray, gps_position_std: np.ndarray,
                    gps_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  events = [(float(t), "pose", i) for i, t in enumerate(pose_t)]
  events += [(float(t), "gps", i) for i, t in enumerate(gps_t)]
  events.sort()

  first_velocity = interpolate(gps_t[:1], pose_t, velocity)[0]
  initial_position = gps_position[0] - first_velocity * (gps_t[0] - events[0][0])
  state = np.concatenate((initial_position, velocity[0]))
  covariance = np.diag([100.0, 100.0, 100.0, 4.0, 4.0, 4.0]) ** 2

  filtered_states, filtered_covariances = [], []
  predicted_states, predicted_covariances, transitions = [], [], []
  event_pose_indices = []
  previous_t = events[0][0]
  for event_index, (event_t, kind, index) in enumerate(events):
    dt = event_t - previous_t if event_index else 0.0
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    noise_gain = np.vstack((np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt))
    process_noise = noise_gain @ (np.eye(3) * 2.0**2) @ noise_gain.T + np.eye(6) * 1e-9
    predicted_state = transition @ state
    predicted_covariance = transition @ covariance @ transition.T + process_noise

    if kind == "pose":
      observation = np.column_stack((np.zeros((3, 3)), np.eye(3)))
      measurement = velocity[index]
      noise = velocity_covariance[index] + np.eye(3) * DEVICE_MOTION_VELOCITY_NOISE_FLOOR**2
      event_pose_indices.append(event_index)
    else:
      observation = np.eye(6)
      measurement = np.concatenate((gps_position[index], gps_velocity[index]))
      std = np.concatenate((np.maximum(gps_position_std[index], 1.0), np.full(3, 0.5)))
      noise = np.diag(std**2)

    state, covariance = _kalman_update(predicted_state, predicted_covariance, measurement, observation, noise)
    predicted_states.append(predicted_state)
    predicted_covariances.append(predicted_covariance)
    filtered_states.append(state)
    filtered_covariances.append(covariance)
    transitions.append(transition)
    previous_t = event_t

  smoothed = np.asarray(filtered_states)
  for i in range(len(events) - 2, -1, -1):
    gain = np.linalg.solve(predicted_covariances[i + 1], transitions[i + 1] @ filtered_covariances[i]).T
    smoothed[i] += gain @ (smoothed[i + 1] - predicted_states[i + 1])

  pose_states = smoothed[event_pose_indices]
  return pose_states[:, :3], pose_states[:, 3:]


def _globalize_live_pose(
  data: LocalizationInput, gps_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  calibration = data.calibration
  device_from_calibrated = euler_to_rot(calibration[None])[0]
  calibrated_from_device = device_from_calibrated.T
  local_from_device = euler_to_rot(data.orientation)

  gps_orientation = interpolate(gps_t, data.pose_t, np.unwrap(data.orientation, axis=0))
  local_from_calibrated = euler_to_rot(gps_orientation) @ device_from_calibrated
  local_heading = np.arctan2(local_from_calibrated[:, 1, 0], local_from_calibrated[:, 0, 0])
  gps_heading = np.arctan2(data.gps_velocity_ned[:, 1], data.gps_velocity_ned[:, 0])
  gps_speed = np.linalg.norm(data.gps_velocity_ned[:, :2], axis=1)
  moving = np.isfinite(gps_heading) & (gps_speed > 5.0)
  if np.count_nonzero(moving) < 2 or np.ptp(gps_t[moving]) < 0.5:
    raise RuntimeError("global yaw needs at least two moving GPS fixes spanning 0.5 seconds")
  yaw_offset = circular_offset(gps_heading[moving], local_heading[moving])
  global_ned_from_local = euler_to_rot(np.array([[0.0, 0.0, yaw_offset]]))[0]
  ned_from_device = global_ned_from_local @ local_from_device

  velocity_device = data.velocity.copy()
  velocity_covariance_device = np.eye(3) * data.velocity_std[:, :, None] ** 2
  if len(data.car_state):
    car_t, car_speed = data.car_state.T
    car_speed = interpolate(data.pose_t, car_t, car_speed)
    velocity_calibrated = np.einsum("ij,kj->ki", calibrated_from_device, velocity_device)
    velocity_calibrated[:, 0] = np.copysign(car_speed, velocity_calibrated[:, 0])
    velocity_device = np.einsum("ij,kj->ki", device_from_calibrated, velocity_calibrated)

    velocity_covariance_calibrated = (
      calibrated_from_device @ velocity_covariance_device @ device_from_calibrated
    )
    velocity_covariance_calibrated[:, 0, :] = 0.0
    velocity_covariance_calibrated[:, :, 0] = 0.0
    velocity_covariance_calibrated[:, 0, 0] = 0.2**2
    velocity_covariance_device = (
      device_from_calibrated @ velocity_covariance_calibrated @ calibrated_from_device
    )
  velocity_ned = np.einsum("kij,kj->ki", ned_from_device, velocity_device)
  velocity_covariance_ned = (
    ned_from_device @ velocity_covariance_device @ ned_from_device.swapaxes(1, 2)
  )

  origin_ecef = geodetic_to_ecef(data.gps_geodetic[:1])[0]
  origin_ned_from_ecef = ned_from_ecef_matrix(data.gps_geodetic[:1])[0]
  gps_ecef = geodetic_to_ecef(data.gps_geodetic)
  gps_position_ned = np.einsum("ij,kj->ki", origin_ned_from_ecef, gps_ecef - origin_ecef)
  position_ned, velocity_ned = _rts_trajectory(
    data.pose_t,
    velocity_ned,
    velocity_covariance_ned,
    gps_t=gps_t,
    gps_position=gps_position_ned,
    gps_position_std=data.gps_position_std,
    gps_velocity=data.gps_velocity_ned,
  )
  position_ecef = origin_ecef + np.einsum("ij,kj->ki", origin_ned_from_ecef.T, position_ned)
  velocity_ecef = np.einsum("ij,kj->ki", origin_ned_from_ecef.T, velocity_ned)

  geodetic_at_pose = ecef_to_geodetic(position_ecef)
  ecef_from_device = ned_from_ecef_matrix(geodetic_at_pose).swapaxes(1, 2) @ ned_from_device
  return position_ecef, velocity_ecef, rot_to_quat(ecef_from_device), data.acceleration_device


def localize(data: LocalizationInput, output_hz: float = 100.0) -> dict[str, np.ndarray]:
  mono_from_unix_ns = data.clock[0] - data.clock[1]
  gps_t = (data.gps_unix_ms * 1_000_000 + mono_from_unix_ns) * 1e-9
  for name, values in (("deviceMotion", data.pose_t), ("GPS", gps_t), ("road-camera", data.frame_t)):
    if not len(values) or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
      raise RuntimeError(f"{name} timestamps must be finite and strictly increasing")
  frame_t = np.asarray(data.frame_t, dtype=np.float64)
  position, velocity, quaternions, acceleration = _globalize_live_pose(data, gps_t)

  start, end = frame_t[0], max(frame_t[-1], data.pose_t[-1])
  trajectory_t = np.arange(start, end + 0.5 / output_hz, 1.0 / output_hz)
  states = np.zeros((len(trajectory_t), STATE_SIZE), dtype=np.float64)
  states[:, 0:3] = interpolate(trajectory_t, data.pose_t, position)
  states[:, 3:7] = nlerp_quaternions(trajectory_t, data.pose_t, quaternions)
  states[:, 7:10] = interpolate(trajectory_t, data.pose_t, velocity)
  states[:, 10:13] = interpolate(trajectory_t, data.pose_t, data.angular_velocity_device)
  states[:, 19:22] = interpolate(trajectory_t, data.pose_t, acceleration)
  states[:, [18, 22, 29]] = 1.0

  calibration, wide = data.calibration.copy(), data.wide_from_device_euler.copy()
  calibration[0] = wide[0] = 0.0
  states[:, 33:36] = wide
  states[:, 36:43] = states[:, :7]

  frame_states = interpolate(frame_t, trajectory_t, states)
  frame_states[:, 3:7] = nlerp_quaternions(frame_t, trajectory_t, states[:, 3:7])
  frame_states[:, 36:43] = frame_states[:, :7]

  return {
    "t": trajectory_t,
    "states": states,
    "frame_t": frame_t,
    "frame_states": frame_states,
    "rpy": calibration,
    "wide_from_device_euler": wide,
  }
