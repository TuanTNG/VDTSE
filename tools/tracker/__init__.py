import ctypes
import pathlib
import os
import numpy as np

dir_ = pathlib.Path(__file__).parent.absolute()
lib = ctypes.cdll.LoadLibrary(os.path.join(dir_, 'track'))

c_track = lib.track
c_track.restypes = ctypes.c_int

c_camerainfo = lib.camera_info


track_result = np.ndarray((256, 7, dtype=np.int32)
count_result = np.ndarray(256, dtype=np.int32)
def track(t, detection):
	detection = np.ascontiguousarray(detection, dtype=np.int32)
	n = c_track(
		ctypes.c_float(t),
		ctypes.c_void_p(detection.ctypes.data), len(detection),
		ctypes.c_void_p(track_result.ctypes.data),
		ctypes.c_void_p(count_result.ctypes.data)
	)

	return np.copy(track_result[:n]), np.copy(count_result)


def camera_info(cw, ch, a, fov, h):
	c_camerainfo(
		ctypes.c_int(cw),
		ctypes.c_int(ch),
		ctypes.c_float(fov * 0.017453292519943295),
		ctypes.c_float(a * 0.017453292519943295),
		ctypes.c_float(h)
	)