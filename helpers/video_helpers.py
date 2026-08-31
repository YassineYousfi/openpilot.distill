from pathlib import Path

import cv2
import numpy as np
import PyNvVideoCodec
from openpilot.common.transformations.camera import denormalize, view_frame_from_device_frame
from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler


_WARP_POINTS = np.asarray(
    [[-1.0, 1.22, 100.0], [1.0, 1.22, 100.0], [-1.0, 1.22, 200.0], [1.0, 1.22, 200.0]],
    dtype=np.float32,
)


def decode_frames(
    path: str | Path,
    index: np.ndarray,
    wanted: set[int],
    gpu_id: int = 0,
    fs=None,
) -> dict[int, np.ndarray]:
    """Decode selected RGB frames with NVDEC, reading only the required file prefix."""

    wanted = {int(frame) for frame in wanted}
    if not wanted:
        return {}
    frame_count = len(index) - 1
    if min(wanted) < 0 or max(wanted) >= frame_count:
        raise IndexError(f"wanted frames must be between 0 and {frame_count - 1}")

    iframe_indexes = np.flatnonzero(index[:-1, 0] == 2)
    next_iframe = np.searchsorted(iframe_indexes, max(wanted), side="right")
    read_end = int(iframe_indexes[next_iframe]) if next_iframe < len(iframe_indexes) else frame_count
    byte_count = int(index[read_end, 1])
    if fs is None:
        with Path(path).open("rb") as video:
            payload = video.read(byte_count)
    else:
        payload = fs.cat_file(str(path), end=byte_count)

    decoder = PyNvVideoCodec.CreateDecoder(
        gpuid=gpu_id,
        codec=PyNvVideoCodec.cudaVideoCodec.HEVC,
        usedevicememory=False,
        outputColorType=PyNvVideoCodec.OutputColorType.RGB,
    )
    decoded = {}
    frame_index = 0
    for packet in (payload, b""):
        packet_array = np.frombuffer(packet, dtype=np.uint8)
        packet_data = PyNvVideoCodec.PacketData()
        packet_data.bsl = len(packet)
        packet_data.bsl_data = packet_array.ctypes.data
        for frame in decoder.Decode(packet_data):
            if frame_index in wanted:
                decoded[frame_index] = np.from_dlpack(frame).copy()
            frame_index += 1

    missing = wanted - decoded.keys()
    if missing:
        raise RuntimeError(f"decoder did not return frames: {sorted(missing)}")
    return decoded


def calibration_view_eulers(
    device_from_calib_euler: np.ndarray,
    augment_from_calib: np.ndarray | None = None,
) -> np.ndarray:
    """Express calibration, optionally augmented, in camera-view Euler coordinates."""

    if augment_from_calib is None:
        augment_from_calib = np.eye(3)
    augment_from_device = np.einsum(
        "...ij,jk->...ik",
        augment_from_calib,
        rot_from_euler(device_from_calib_euler).T,
    )
    return np.einsum(
        "ij,...j->...i",
        view_frame_from_device_frame,
        euler_from_rot(augment_from_device.swapaxes(-1, -2)),
    )


def calibration_warp_matrix(
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    view_euler: np.ndarray,
) -> np.ndarray:
    """Return the perspective transform from a source camera to a calibrated model view."""

    after = _WARP_POINTS @ rot_from_euler(view_euler)
    before_pixels = denormalize(
        _WARP_POINTS[:, :2] / _WARP_POINTS[:, 2, None],
        intrinsics=source_intrinsics,
    )
    after_pixels = denormalize(after[:, :2] / after[:, 2, None], intrinsics=target_intrinsics)
    return cv2.getPerspectiveTransform(before_pixels.astype(np.float32), after_pixels.astype(np.float32))
