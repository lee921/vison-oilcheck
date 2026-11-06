from __future__ import annotations

import numpy as np

from hik.decode import FrameBuffer


def test_frame_buffer_clear_removes_last_frame() -> None:
    buffer = FrameBuffer()
    buffer.update_bgr(np.zeros((4, 6, 3), dtype=np.uint8), 6, 4)

    frame = buffer.get()
    assert frame is not None
    assert frame.width == 6 and frame.height == 4
    assert buffer.last_ts() == frame.ts

    buffer.clear()
    assert buffer.get() is None
    assert buffer.last_ts() is None
