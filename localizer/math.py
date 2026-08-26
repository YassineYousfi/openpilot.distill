"""Small batched geometry and interpolation helpers."""

import numpy as np

WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(geodetic: np.ndarray) -> np.ndarray:
  geodetic = np.asarray(geodetic, dtype=np.float64)
  lat = np.radians(geodetic[:, 0])
  lon = np.radians(geodetic[:, 1])
  alt = geodetic[:, 2]
  radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
  return np.column_stack((
    (radius + alt) * np.cos(lat) * np.cos(lon),
    (radius + alt) * np.cos(lat) * np.sin(lon),
    (radius * (1.0 - WGS84_E2) + alt) * np.sin(lat),
  ))


def ecef_to_geodetic(ecef: np.ndarray) -> np.ndarray:
  ecef = np.asarray(ecef, dtype=np.float64)
  x, y, z = ecef.T
  longitude = np.arctan2(y, x)
  horizontal_radius = np.hypot(x, y)
  latitude = np.arctan2(z, horizontal_radius * (1.0 - WGS84_E2))
  for _ in range(5):
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(latitude) ** 2)
    altitude = horizontal_radius / np.cos(latitude) - prime_vertical
    latitude = np.arctan2(
      z,
      horizontal_radius * (1.0 - WGS84_E2 * prime_vertical / (prime_vertical + altitude)),
    )
  prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(latitude) ** 2)
  altitude = horizontal_radius / np.cos(latitude) - prime_vertical
  return np.column_stack((np.degrees(latitude), np.degrees(longitude), altitude))


def euler_to_rot(euler: np.ndarray) -> np.ndarray:
  euler = np.asarray(euler, dtype=np.float64)
  roll, pitch, yaw = euler.T
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rotations = np.empty((len(euler), 3, 3), dtype=np.float64)
  rotations[:, 0] = np.column_stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr))
  rotations[:, 1] = np.column_stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr))
  rotations[:, 2] = np.column_stack((-sp, cp * sr, cp * cr))
  return rotations


def rot_to_quat(rotations: np.ndarray) -> np.ndarray:
  rotations = np.asarray(rotations, dtype=np.float64)
  quaternions = np.empty((len(rotations), 4), dtype=np.float64)
  for i, rot in enumerate(rotations):
    trace = np.trace(rot)
    if trace > 0:
      scale = 0.5 / np.sqrt(trace + 1.0)
      quat = np.array([
        0.25 / scale,
        (rot[2, 1] - rot[1, 2]) * scale,
        (rot[0, 2] - rot[2, 0]) * scale,
        (rot[1, 0] - rot[0, 1]) * scale,
      ])
    else:
      axis = int(np.argmax(np.diag(rot)))
      axis_1, axis_2 = (axis + 1) % 3, (axis + 2) % 3
      scale = 2.0 * np.sqrt(1.0 + rot[axis, axis] - rot[axis_1, axis_1] - rot[axis_2, axis_2])
      quat = np.empty(4)
      quat[0] = (rot[axis_2, axis_1] - rot[axis_1, axis_2]) / scale
      quat[axis + 1] = 0.25 * scale
      quat[axis_1 + 1] = (rot[axis_1, axis] + rot[axis, axis_1]) / scale
      quat[axis_2 + 1] = (rot[axis_2, axis] + rot[axis, axis_2]) / scale
    quaternions[i] = quat if quat[0] >= 0 else -quat
  return quaternions


def ned_from_ecef_matrix(geodetic: np.ndarray) -> np.ndarray:
  geodetic = np.asarray(geodetic, dtype=np.float64)
  latitude, longitude = np.radians(geodetic[:, :2]).T
  slat, clat = np.sin(latitude), np.cos(latitude)
  slon, clon = np.sin(longitude), np.cos(longitude)
  return np.stack((
    np.column_stack((-slat * clon, -slat * slon, clat)),
    np.column_stack((-slon, clon, np.zeros_like(latitude))),
    np.column_stack((-clat * clon, -clat * slon, -slat)),
  ), axis=1)


def interpolate(t_new: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim == 1:
    return np.interp(t_new, t, values)
  return np.column_stack([np.interp(t_new, t, values[:, i]) for i in range(values.shape[1])])


def nlerp_quaternions(t_new: np.ndarray, t: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
  quaternions = np.asarray(quaternions, dtype=np.float64).copy()
  for i in range(1, len(quaternions)):
    if np.dot(quaternions[i - 1], quaternions[i]) < 0:
      quaternions[i] *= -1
  result = interpolate(t_new, t, quaternions)
  return result / np.linalg.norm(result, axis=1, keepdims=True)


def circular_offset(target: np.ndarray, source: np.ndarray) -> float:
  delta = target - source
  center = np.arctan2(np.mean(np.sin(delta)), np.mean(np.cos(delta)))
  residual = (delta - center + np.pi) % (2 * np.pi) - np.pi
  return float(center + np.median(residual))
