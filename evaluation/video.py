"""MP4 writer that produces browser-playable video (H.264, yuv420p).

OpenCV's default 'mp4v' codec does not play in most browsers, so all episode
videos go through this writer.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class Mp4Writer:
    """Context manager that appends HxWx3 uint8 RGB frames to an H.264 MP4 file."""

    def __init__(self, path: Path, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.num_frames = 0
        self._writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=16
        )

    def append(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 uint8 frame, got dtype={frame.dtype} shape={frame.shape}")
        self._writer.append_data(frame)
        self.num_frames += 1

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._writer.close()
        if exc_type is None and self.num_frames == 0:
            raise RuntimeError(f"No frames were written to {self.path}")
