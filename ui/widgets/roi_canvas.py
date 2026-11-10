"""ROI 标定画布：显示相机画面并支持拖拽绘制矩形 ROI。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.oilcheck.roi import Roi

ROI_COLORS = {
    "tray": QColor("#ffd60a"),
    "work": QColor("#64d2ff"),
    "oiling": QColor("#32d74b"),
}
ROI_NAMES = {
    "tray": "涂油盘区",
    "work": "工作追踪区",
    "oiling": "涂油区",
}
ROI_KEYS = tuple(ROI_NAMES)


class RoiCanvas(QWidget):
    roiChanged = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._image: QImage | None = None
        self._source_size: tuple[int, int] = (0, 0)
        self._rois: dict[str, Roi | None] = {key: None for key in ROI_KEYS}
        self._mode = "tray"
        self._editable = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self._hint = "点击“打开相机”获取画面"

    def set_frame(self, bgr: np.ndarray) -> bool:
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return False
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return False
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        self._image = QImage(
            bgr.data, w, h, int(bgr.strides[0]), QImage.Format.Format_BGR888
        ).copy()
        self._source_size = (int(w), int(h))
        self._hint = ""
        self.update()
        return True

    def clear(self, hint: str = "等待相机画面") -> None:
        self._image = None
        self._source_size = (0, 0)
        self._hint = hint
        self.update()

    def set_rois(self, rois: dict[str, Roi | None]) -> None:
        self._rois = {key: rois.get(key) for key in ROI_KEYS}
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode in self._rois:
            self._mode = mode
            self.update()

    def set_editable(self, editable: bool) -> None:
        self._editable = bool(editable)
        self.setCursor(
            Qt.CursorShape.CrossCursor if self._editable else Qt.CursorShape.ArrowCursor
        )
        if not self._editable:
            self._drag_start = None
            self._drag_end = None
        self.update()

    def clear_roi(self, target: str | None = None) -> bool:
        """清除指定区域的 ROI；未指定时清除当前绘制目标。"""
        key = target or self._mode
        if self._rois.get(key) is None:
            return False
        self._rois[key] = None
        self.roiChanged.emit(key, None)
        self.update()
        return True

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08090a"))

        image_rect = self._image_rect()
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(image_rect, self._image)
            self._draw_rois(painter, image_rect)
            self._draw_drag(painter, image_rect)
        else:
            painter.setPen(QColor("#8a8f98"))
            painter.setFont(QFont("", 12))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._hint or "等待相机画面"
            )

        if self._editable:
            banner = f"拖拽绘制{ROI_NAMES.get(self._mode, self._mode)}"
        else:
            banner = "截取画面后可拖拽绘制 ROI"
        painter.setFont(QFont("", 10, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(banner) + 20
        banner_rect = QRectF(10, 10, width, 28)
        painter.fillRect(banner_rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor("#f2f2f7"))
        painter.drawText(banner_rect, Qt.AlignmentFlag.AlignCenter, banner)
        painter.end()

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

    def _source_to_widget(self, x: float, y: float, image_rect: QRectF) -> QPointF:
        src_w, src_h = self._source_size
        if src_w <= 0 or src_h <= 0:
            return QPointF(0, 0)
        return QPointF(
            image_rect.x() + x / src_w * image_rect.width(),
            image_rect.y() + y / src_h * image_rect.height(),
        )

    def _widget_to_source(
        self, pos: QPointF, *, clamp: bool = False
    ) -> tuple[float, float] | None:
        image_rect = self._image_rect()
        if not clamp and not image_rect.contains(pos):
            return None
        src_w, src_h = self._source_size
        if src_w <= 0 or src_h <= 0:
            return None
        x = (pos.x() - image_rect.x()) / max(1.0, image_rect.width())
        y = (pos.y() - image_rect.y()) / max(1.0, image_rect.height())
        if not clamp and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _draw_rois(self, painter: QPainter, image_rect: QRectF) -> None:
        src_w, src_h = self._source_size
        for key, roi in self._rois.items():
            if roi is None:
                continue
            color = ROI_COLORS.get(key, QColor("#ffffff"))
            pen = QPen(color)
            pen.setWidth(3 if key == self._mode else 2)
            if key == "oiling":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            points = roi.pixel_points(src_w, src_h)
            if roi.kind == "rect":
                (x1, y1), (x2, y2) = points
                top_left = self._source_to_widget(x1, y1, image_rect)
                bottom_right = self._source_to_widget(x2, y2, image_rect)
                painter.drawRect(QRectF(top_left, bottom_right).normalized())
                label_pos = top_left
            else:
                from PySide6.QtGui import QPolygonF

                polygon = QPolygonF(
                    [self._source_to_widget(x, y, image_rect) for x, y in points]
                )
                painter.drawPolygon(polygon)
                label_pos = polygon[0]

            label = ROI_NAMES.get(key, key)
            if roi.kind == "rect":
                (x1, y1), (x2, y2) = points
                size = f" {abs(x2 - x1):.0f}×{abs(y2 - y1):.0f}px"
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                size = f" {max(xs) - min(xs):.0f}×{max(ys) - min(ys):.0f}px"
            painter.setFont(QFont("", 9))
            label_rect = QRectF(label_pos.x() + 4, label_pos.y() - 18, 190, 16)
            painter.fillRect(label_rect, QColor(0, 0, 0, 150))
            painter.setPen(color)
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{label}{size}",
            )

    def _draw_drag(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._drag_start is None or self._drag_end is None:
            return
        rect = QRectF(self._drag_start, self._drag_end).normalized()
        color = ROI_COLORS.get(self._mode, QColor("#ffffff"))
        pen = QPen(color)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        start = self._widget_to_source(self._drag_start, clamp=True)
        end = self._widget_to_source(self._drag_end, clamp=True)
        if start is None or end is None:
            return
        src_w, src_h = self._source_size
        width_px = abs(end[0] - start[0]) * src_w
        height_px = abs(end[1] - start[1]) * src_h
        painter.setFont(QFont("", 9))
        text = f"{width_px:.0f}×{height_px:.0f}px"
        text_rect = QRectF(rect.x(), max(0.0, rect.y() - 18), 150, 16)
        painter.fillRect(text_rect, QColor(0, 0, 0, 150))
        painter.setPen(color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f" {text}",
        )

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            self._editable
            and self._image is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            source = self._widget_to_source(event.position())
            if source is not None:
                self._drag_start = event.position()
                self._drag_end = event.position()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_end = event.position()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_end = event.position()
            start = self._widget_to_source(self._drag_start)
            # 拖到画面外时裁剪到画面边缘，避免整次拖拽被丢弃。
            end = self._widget_to_source(self._drag_end, clamp=True)
            self._drag_start = None
            self._drag_end = None
            if start is not None and end is not None:
                (x1, y1), (x2, y2) = start, end
                if abs(x2 - x1) > 0.005 and abs(y2 - y1) > 0.005:
                    roi = Roi.from_dict(
                        {
                            "kind": "rect",
                            "points": [[x1, y1], [x2, y2]],
                        }
                    )
                    if roi is not None:
                        self._rois[self._mode] = roi
                        self.roiChanged.emit(self._mode, roi)
            self.update()
            return
        super().mouseReleaseEvent(event)
