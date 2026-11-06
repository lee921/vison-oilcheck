"""监控画面控件：视频、ROI、检测框与状态横幅。

与旧预览控件不同，这里直接使用 QPainter 绘制，避免每帧做 OpenCV 叠加和
PIL 文本转换；ROI 使用归一化坐标，窗口缩放时同步缩放。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.oilcheck.roi import Roi
from app.oilcheck.types import Detection


@dataclass(frozen=True)
class RoiOverlay:
    name: str
    roi: Roi
    color: str
    dashed: bool = False


class OilCheckVideoView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._image: QImage | None = None
        self._source_size: tuple[int, int] = (0, 0)
        self._rois: tuple[RoiOverlay, ...] = ()
        self._detections: tuple[Detection, ...] = ()
        self._banner = ""
        self._hint = "等待视频帧"
        self._hand_label = "hand"
        self._rod_label = "rod"
        self._brush_label = "brush"
        self._tray_label = "tray"

    def set_frame(self, bgr: np.ndarray, ts: float) -> bool:
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return False
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return False
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        image = QImage(
            bgr.data, w, h, int(bgr.strides[0]), QImage.Format.Format_BGR888
        ).copy()
        self._image = image
        self._source_size = (int(w), int(h))
        self._hint = ""
        self.update()
        return True

    def set_overlays(
        self,
        *,
        rois: tuple[RoiOverlay, ...],
        detections: tuple[Detection, ...],
        banner: str = "",
        hand_label: str = "hand",
        rod_label: str = "rod",
        brush_label: str = "brush",
        tray_label: str = "tray",
    ) -> None:
        self._rois = rois
        self._detections = detections
        self._banner = banner
        self._hand_label = hand_label
        self._rod_label = rod_label
        self._brush_label = brush_label
        self._tray_label = tray_label
        self.update()

    def clear(self, hint: str = "等待视频帧") -> None:
        self._image = None
        self._hint = hint
        self._banner = ""
        self._rois = ()
        self._detections = ()
        self.update()

    @property
    def has_frame(self) -> bool:
        return self._image is not None

    def paintEvent(self, event) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08090a"))

        image_rect = self._image_rect()
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(image_rect, self._image)
            self._draw_rois(painter, image_rect)
            self._draw_detections(painter, image_rect)
        else:
            painter.setPen(QColor("#8a8f98"))
            painter.setFont(QFont("", 12))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._hint or "等待视频帧"
            )

        self._draw_banner(painter)
        painter.end()

    # ------------------------------------------------------------------
    def _image_rect(self) -> QRectF:
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        if self._image is None or self._source_size[0] <= 0:
            return QRectF(0, 0, widget_w, widget_h)

        src_w, src_h = self._source_size
        scale = min(widget_w / src_w, widget_h / src_h)
        draw_w = src_w * scale
        draw_h = src_h * scale
        return QRectF(
            (widget_w - draw_w) * 0.5,
            (widget_h - draw_h) * 0.5,
            draw_w,
            draw_h,
        )

    def _map_point(self, x: float, y: float, image_rect: QRectF) -> QPointF:
        src_w, src_h = self._source_size
        if src_w <= 0 or src_h <= 0:
            return QPointF(0, 0)
        return QPointF(
            image_rect.x() + x / src_w * image_rect.width(),
            image_rect.y() + y / src_h * image_rect.height(),
        )

    def _draw_rois(self, painter: QPainter, image_rect: QRectF) -> None:
        src_w, src_h = self._source_size
        for item in self._rois:
            color = QColor(item.color)
            pen = QPen(color)
            pen.setWidth(2)
            if item.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            points = item.roi.pixel_points(src_w, src_h)
            if item.roi.kind == "rect":
                (x1, y1), (x2, y2) = points
                painter.drawRect(
                    QRectF(
                        self._map_point(x1, y1, image_rect),
                        self._map_point(x2, y2, image_rect),
                    )
                )
                label_point = self._map_point(x1, y1, image_rect)
            else:
                polygon = QPolygonF(
                    [self._map_point(x, y, image_rect) for x, y in points]
                )
                painter.drawPolygon(polygon)
                label_point = polygon[0]

            if item.name:
                painter.setFont(QFont("", 9))
                label_rect = QRectF(
                    label_point.x() + 4,
                    label_point.y() - 18,
                    140,
                    16,
                )
                painter.fillRect(label_rect, QColor(0, 0, 0, 150))
                painter.setPen(color)
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    item.name,
                )

    def _draw_detections(self, painter: QPainter, image_rect: QRectF) -> None:
        painter.setFont(QFont("", 9))
        for det in self._detections:
            base_label = det.label.split("#", 1)[0]
            if base_label == self._hand_label:
                color = QColor("#ff9f0a")
            elif base_label == self._rod_label:
                color = QColor("#32d74b")
            elif base_label == self._brush_label:
                color = QColor("#ff375f")
            elif base_label == self._tray_label:
                color = QColor("#ffd60a")
            else:
                color = QColor("#64d2ff")

            top_left = self._map_point(det.box.x1, det.box.y1, image_rect)
            bottom_right = self._map_point(det.box.x2, det.box.y2, image_rect)
            box_rect = QRectF(top_left, bottom_right).normalized()

            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box_rect)

            label = f"{det.label} {det.confidence:.2f}"
            text_rect = QRectF(
                box_rect.x(),
                max(image_rect.y(), box_rect.y() - 18),
                max(80.0, box_rect.width()),
                16,
            )
            painter.fillRect(
                text_rect, QColor(color.red(), color.green(), color.blue(), 190)
            )
            painter.setPen(QColor("#000000"))
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

    def _draw_banner(self, painter: QPainter) -> None:
        if not self._banner:
            return
        painter.setFont(QFont("", 10, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(self._banner) + 20
        rect = QRectF(10, 10, text_w, 28)
        painter.fillRect(rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor("#f2f2f7"))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            self._banner,
        )
