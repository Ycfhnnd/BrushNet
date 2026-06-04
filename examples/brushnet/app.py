from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from PyQt5.QtCore import QPointF, QRect, QRectF, QSize, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "当前环境没有安装 PyQt5。请先执行 `pip install PyQt5`，然后再启动桌面界面。"
    ) from exc

try:
    import brushnet_app_core as core
except Exception as exc:
    message = str(exc)
    if "huggingface-hub" in message and "transformers" in message:
        raise SystemExit(
            "BrushNet 环境依赖冲突：当前 `transformers` 和 `huggingface-hub` 版本不兼容。\n"
            "请在 `(brushnet)` 环境里执行下面这条命令后再启动：\n"
            'pip install "huggingface-hub>=0.34,<1.0" "transformers>=4.25.1,<5" -U\n'
            "如果你之前装过 1.x 版本的 huggingface-hub，建议先执行：\n"
            "pip uninstall -y huggingface-hub\n"
            '然后再执行上面的安装命令。'
        ) from exc
    raise


APP_STYLE = """
QMainWindow {
  background: #edf2f6;
}

QWidget {
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
  font-size: 14px;
  color: #1b2733;
}

QLabel#TitleLabel {
  font-size: 24px;
  font-weight: 700;
  color: #102a43;
}

QLabel#SubTitleLabel {
  font-size: 13px;
  color: #526170;
}

QLabel#HintLabel {
  font-size: 12px;
  color: #66788a;
}

QLabel#PillLabel {
  background: #e8f1fb;
  border: 1px solid #c9d8e8;
  border-radius: 6px;
  color: #244a73;
  font-size: 12px;
  padding: 4px 9px;
}

QLabel#BrandLogoLabel {
  background: transparent;
  border: none;
  padding: 0;
}

QFrame#HeroCard,
QGroupBox {
  background: #ffffff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
}

QFrame#HeroCard {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #ffffff, stop:0.58 #f8fbfd, stop:1 #eef6f3);
}

QGroupBox {
  font-weight: 700;
  margin-top: 18px;
  padding-top: 16px;
}

QGroupBox::title {
  subcontrol-origin: margin;
  left: 14px;
  padding: 0 7px;
  color: #243b53;
}

QPushButton {
  min-height: 36px;
  border-radius: 7px;
  background: #f7fafc;
  border: 1px solid #cbd5e1;
  padding: 0 14px;
  font-weight: 600;
  color: #22313f;
}

QPushButton:hover {
  background: #eef4f8;
  border-color: #9fb3c8;
}

QPushButton#PrimaryButton {
  background: #0f766e;
  color: white;
  border: none;
}

QPushButton#PrimaryButton:hover {
  background: #0b635d;
}

QPushButton#AccentButton {
  background: #255f9e;
  color: white;
  border: none;
}

QPushButton#AccentButton:hover {
  background: #1f4f84;
}

QLineEdit,
QTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QListWidget {
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 7px;
  padding: 6px 8px;
  selection-background-color: #d9ecff;
  selection-color: #102a43;
}

QListWidget {
  padding: 12px;
}

QScrollArea {
  border: none;
}

QSplitter::handle {
  background: #d9e2ec;
}
"""


def numpy_to_qimage(image: np.ndarray) -> QImage:
    # 将 numpy 图像转成 Qt 可显示的 QImage。
    image = core.ensure_uint8_rgb(image)
    if image is None:
        raise ValueError("Image is None.")
    height, width, channels = image.shape
    bytes_per_line = channels * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    # 生成结果一般是 PIL.Image，这里统一转成界面控件可直接显示的 QPixmap。
    rgb = np.array(image.convert("RGB"))
    return QPixmap.fromImage(numpy_to_qimage(rgb))


def load_image_from_path(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


PRIMARY_BRAND_IMAGE = Path("examples/brushnet/src/njut_logo.png")

BRAND_IMAGE_CANDIDATES = (
    Path("examples/brushnet/src/njut_logo.jpg"),
    Path("examples/brushnet/src/njut_logo.jpeg"),
    Path("examples/brushnet/src/nanjing_tech_logo.png"),
    Path("examples/brushnet/src/nanjing_tech_university.png"),
)


def _load_trimmed_brand_pixmap(image_path: Path, target_size: QSize) -> Optional[QPixmap]:
    # 自动裁掉 logo 周围的大块空白，让标题区展示更紧凑。
    with Image.open(image_path) as image:
        rgba_image = image.convert("RGBA")
        rgba_array = np.array(rgba_image)

    alpha_mask = rgba_array[:, :, 3] > 0
    content_mask = np.any(rgba_array[:, :, :3] < 245, axis=2)
    non_background_mask = alpha_mask & content_mask
    if not np.any(non_background_mask):
        non_background_mask = alpha_mask
    if not np.any(non_background_mask):
        return None

    rows, cols = np.where(non_background_mask)
    padding = 12
    top = max(0, int(rows.min()) - padding)
    bottom = min(rgba_array.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(cols.min()) - padding)
    right = min(rgba_array.shape[1], int(cols.max()) + padding + 1)

    cropped = Image.fromarray(rgba_array[top:bottom, left:right], mode="RGBA")
    pixmap = pil_to_qpixmap(cropped.convert("RGB"))
    if pixmap.isNull():
        return None
    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _build_shield_path(rect: QRectF) -> QPainterPath:
    path = QPainterPath()
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    width = rect.width()
    height = rect.height()

    path.moveTo(left + width * 0.12, top)
    path.lineTo(right - width * 0.12, top)
    path.lineTo(right, top + height * 0.11)
    path.lineTo(right, top + height * 0.67)
    path.quadTo(left + width * 0.5, bottom, left, top + height * 0.67)
    path.lineTo(left, top + height * 0.11)
    path.closeSubpath()
    return path

def build_brand_pixmap(target_size: QSize = QSize(112, 112)) -> QPixmap:
    # 优先使用本地图像资源，找不到再退回到代码绘制的备用图。
    primary_path = core.PROJECT_ROOT / PRIMARY_BRAND_IMAGE
    if primary_path.exists():
        pixmap = _load_trimmed_brand_pixmap(primary_path, target_size)
        if pixmap is not None:
            return pixmap

    for relative_path in BRAND_IMAGE_CANDIDATES:
        candidate = core.PROJECT_ROOT / relative_path
        if candidate.exists():
            pixmap = _load_trimmed_brand_pixmap(candidate, target_size)
            if pixmap is not None:
                return pixmap



class ImageCanvas(QLabel):
    imageClicked = pyqtSignal(int, int)

    def __init__(self, title: str, clickable: bool = False, minimum_height: int = 280, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # 这个控件既负责显示图像，也承担“点击取点”的交互能力。
        self._title = title
        self._clickable = clickable
        self._image: Optional[np.ndarray] = None
        self._qimage: Optional[QImage] = None
        self._display_rect = QRect()

        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(minimum_height)
        self.setContentsMargins(10, 10, 10, 10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            """
            QLabel {
              background: #fbfdff;
              border: 1px dashed #aab7c4;
              border-radius: 8px;
              color: #627386;
              padding: 10px;
            }
            """
        )
        self._set_placeholder()

    def _set_placeholder(self) -> None:
        suffix = "，单击后可在图像上添加分割点" if self._clickable else ""
        self.setText(f"{self._title}\n\n等待载入图像{suffix}")
        self.setPixmap(QPixmap())
        self._display_rect = QRect()

    def set_numpy_image(self, image: Optional[np.ndarray]) -> None:
        self._image = None if image is None else np.array(image, copy=True)
        self._qimage = None if image is None else numpy_to_qimage(image)
        self._update_pixmap()

    def clear_image(self) -> None:
        self._image = None
        self._qimage = None
        self._set_placeholder()

    def _update_pixmap(self) -> None:
        # 根据当前控件大小自适应缩放图片，同时记录真实显示区域，
        # 便于后续把鼠标点击位置映射回原图坐标。
        if self._qimage is None:
            self._set_placeholder()
            return

        available = self.contentsRect()
        if available.width() <= 0 or available.height() <= 0:
            return

        pixmap = QPixmap.fromImage(self._qimage)
        scaled = pixmap.scaled(available.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = available.x() + max(0, (available.width() - scaled.width()) // 2)
        y = available.y() + max(0, (available.height() - scaled.height()) // 2)
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setText("")
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        super().mousePressEvent(event)
        if not self._clickable or self._image is None or event.button() != Qt.LeftButton:
            return
        if not self._display_rect.contains(event.pos()):
            return

        display_width = max(1, self._display_rect.width())
        display_height = max(1, self._display_rect.height())
        local_x = event.pos().x() - self._display_rect.x()
        local_y = event.pos().y() - self._display_rect.y()
        image_height, image_width = self._image.shape[:2]
        # 将缩放后显示区域中的点击点，映射回原图坐标系。
        image_x = int(np.clip(local_x * image_width / display_width, 0, image_width - 1))
        image_y = int(np.clip(local_y * image_height / display_height, 0, image_height - 1))
        self.imageClicked.emit(image_x, image_y)


class InferenceWorker(QThread):
    succeeded = pyqtSignal(object, object, object, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        input_image: Optional[np.ndarray],
        original_image: Optional[np.ndarray],
        original_mask: Optional[np.ndarray],
        uploaded_mask: Optional[np.ndarray],
        selected_points: Sequence[Tuple[Tuple[int, int], int]],
        prompt: str,
        negative_prompt: str,
        blending: bool,
        invert_mask: bool,
        control_strength: float,
        seed: int,
        randomize_seed: bool,
        guidance_scale: float,
        num_inference_steps: int,
        num_outputs: int,
        base_model: str,
        brushnet_path: str,
    ):
        super().__init__()
        # 将当前界面状态完整打包，交给后台线程执行，避免推理时阻塞主界面。
        self.payload = {
            "input_image": None if input_image is None else np.array(input_image, copy=True),
            "original_image": None if original_image is None else np.array(original_image, copy=True),
            "original_mask": None if original_mask is None else np.array(original_mask, copy=True),
            "uploaded_mask": None if uploaded_mask is None else np.array(uploaded_mask, copy=True),
            "selected_points": list(selected_points),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "blending": blending,
            "invert_mask": invert_mask,
            "control_strength": control_strength,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_outputs": num_outputs,
            "base_model": base_model,
            "brushnet_path": brushnet_path,
        }

    def run(self) -> None:  # type: ignore[override]
        # 真正的推理逻辑在 core 层，这里只负责线程调用与结果转发。
        try:
            results, effective_mask, masked_image, status = core.run_brushnet_inference(**self.payload)
        except core.BrushNetAppError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"推理过程中发生异常：{exc}")
        else:
            self.succeeded.emit(results, effective_mask, masked_image, status)


class BrushNetQtWindow(QMainWindow):
    def __init__(self, base_model: str, brushnet_path: str):
        super().__init__()
        # 保存界面当前状态，所有交互都围绕这几个核心字段展开。
        self.default_base_model = base_model
        self.default_brushnet_path = brushnet_path

        self.original_image: Optional[np.ndarray] = None
        self.original_mask: Optional[np.ndarray] = None
        self.uploaded_mask: Optional[np.ndarray] = None
        self.selected_points: List[Tuple[Tuple[int, int], int]] = []
        self.current_results: List[Image.Image] = []
        self.worker: Optional[InferenceWorker] = None

        self.setWindowTitle("基于扩散模型的双分支结构的图像修复技术研究 - PyQt5")
        self.resize(1600, 980)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._set_initial_status()

    def _build_ui(self) -> None:
        # 主界面分为左右两栏：
        # 左侧负责输入图像、掩膜和参数；右侧负责预览和结果展示。
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        hero_card = QFrame()
        hero_card.setObjectName("HeroCard")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(18, 12, 18, 12)
        hero_layout.setSpacing(0)

        title_label = QLabel("基于扩散模型的双分支结构的图像修复技术研究")
        title_label.setObjectName("TitleLabel")
        title_label.setWordWrap(True)

        brand_label = QLabel()
        brand_label.setObjectName("BrandLogoLabel")
        brand_label.setAlignment(Qt.AlignCenter)
        brand_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        brand_pixmap = build_brand_pixmap(QSize(112, 112))
        brand_label.setPixmap(brand_pixmap)
        brand_label.setFixedSize(brand_pixmap.size())

        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)
        hero_row.addWidget(brand_label, 0, Qt.AlignVCenter)

        hero_text_widget = QWidget()
        hero_text_layout = QVBoxLayout(hero_text_widget)
        hero_text_layout.setContentsMargins(0, 0, 0, 0)
        hero_text_layout.setSpacing(2)
        hero_text_layout.addStretch(1)
        hero_text_layout.addWidget(title_label)
        hero_badges = QHBoxLayout()
        hero_badges.setSpacing(8)
        for text in ("SAM 交互分割", "结果展示"):
            badge = QLabel(text)
            badge.setObjectName("PillLabel")
            hero_badges.addWidget(badge)
        hero_badges.addStretch(1)
        hero_text_layout.addLayout(hero_badges)
        hero_text_layout.addStretch(1)
        hero_row.addWidget(hero_text_widget, 1)

        hero_layout.addLayout(hero_row)
        root_layout.addWidget(hero_card)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        image_group = QGroupBox("输入图像与掩码")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(12)

        self.input_canvas = ImageCanvas("输入图像", clickable=True, minimum_height=520)
        self.input_canvas.imageClicked.connect(self._handle_canvas_click)
        image_layout.addWidget(self.input_canvas)

        image_button_row = QHBoxLayout()
        self.open_image_button = QPushButton("上传图片")
        self.open_mask_button = QPushButton("上传黑白掩码")
        self.clear_mask_button = QPushButton("清空上传掩码")
        self.clear_mask_button.setObjectName("AccentButton")
        image_button_row.addWidget(self.open_image_button)
        image_button_row.addWidget(self.open_mask_button)
        image_button_row.addWidget(self.clear_mask_button)
        image_layout.addLayout(image_button_row)

        point_row = QHBoxLayout()
        self.point_mode_combo = QComboBox()
        self.point_mode_combo.addItems(["添加区域", "排除区域"])
        self.invert_mask_checkbox = QCheckBox("反转掩码语义")
        self.invert_mask_checkbox.setChecked(True)
        point_row.addWidget(QLabel("点击模式"))
        point_row.addWidget(self.point_mode_combo, 1)
        point_row.addWidget(self.invert_mask_checkbox)
        image_layout.addLayout(point_row)

        point_button_row = QHBoxLayout()
        self.undo_button = QPushButton("撤销一个点")
        self.clear_points_button = QPushButton("清空点击分割")
        point_button_row.addWidget(self.undo_button)
        point_button_row.addWidget(self.clear_points_button)
        image_layout.addLayout(point_button_row)

        left_layout.addWidget(image_group, 1)

        prompt_group = QGroupBox("提示词与生成操作")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(10)

        prompt_layout.addWidget(QLabel("Prompt"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(74)
        self.prompt_edit.setPlaceholderText("例如：A futuristic electric car parked in the city street")
        prompt_layout.addWidget(self.prompt_edit)

        prompt_layout.addWidget(QLabel("Negative Prompt"))
        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setFixedHeight(74)
        self.negative_prompt_edit.setPlainText("ugly, low quality, blurry, distorted")
        prompt_layout.addWidget(self.negative_prompt_edit)

        self.settings_summary_label = QLabel()
        self.settings_summary_label.setObjectName("HintLabel")
        self.settings_summary_label.setWordWrap(True)
        prompt_layout.addWidget(self.settings_summary_label)

        action_row = QHBoxLayout()
        self.settings_button = QPushButton("参数设置")
        self.settings_button.setObjectName("AccentButton")
        self.run_button = QPushButton("开始生成")
        self.run_button.setObjectName("PrimaryButton")
        action_row.addWidget(self.settings_button)
        action_row.addWidget(self.run_button, 1)
        prompt_layout.addLayout(action_row)

        left_layout.addWidget(prompt_group)

        self._build_settings_dialog()

        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)

        preview_group = QGroupBox("掩码预览")
        preview_layout = QHBoxLayout(preview_group)
        preview_layout.setSpacing(12)
        self.mask_preview = ImageCanvas("有效编辑掩码", clickable=False, minimum_height=260)
        self.masked_preview = ImageCanvas("送入模型的 Masked Image", clickable=False, minimum_height=260)
        preview_layout.addWidget(self.mask_preview, 1)
        preview_layout.addWidget(self.masked_preview, 1)
        right_layout.addWidget(preview_group)

        result_group = QGroupBox("生成结果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)

        # 原图 vs 生成结果 对比预览
        compare_row = QHBoxLayout()
        compare_row.setSpacing(10)
        self.compare_original = ImageCanvas("原图（含掩码）", clickable=False, minimum_height=300)
        self.compare_result   = ImageCanvas("生成结果",       clickable=False, minimum_height=300)
        compare_row.addWidget(self.compare_original, 1)
        compare_row.addWidget(self.compare_result,   1)
        result_layout.addLayout(compare_row, 1)

        # 缩略图列表（横排，点击切换对比视图）
        self.result_list = QListWidget()
        self.result_list.setViewMode(QListWidget.IconMode)
        self.result_list.setResizeMode(QListWidget.Adjust)
        self.result_list.setWrapping(False)
        self.result_list.setFlow(QListWidget.LeftToRight)
        self.result_list.setSpacing(8)
        self.result_list.setMovement(QListWidget.Static)
        self.result_list.setIconSize(QSize(120, 120))
        self.result_list.setFixedHeight(148)
        result_layout.addWidget(self.result_list)

        # 保存按钮
        save_row = QHBoxLayout()
        self.save_current_button = QPushButton("保存当前结果")
        self.save_all_button     = QPushButton("保存全部结果")
        self.save_all_button.setObjectName("AccentButton")
        save_row.addWidget(self.save_current_button)
        save_row.addWidget(self.save_all_button)
        result_layout.addLayout(save_row)

        right_layout.addWidget(result_group, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([860, 700])

        self.setCentralWidget(central)

        self.open_image_button.clicked.connect(self._choose_source_image)
        self.open_mask_button.clicked.connect(self._choose_mask_image)
        self.clear_mask_button.clicked.connect(self._clear_uploaded_mask)
        self.undo_button.clicked.connect(self._undo_last_point)
        self.clear_points_button.clicked.connect(self._clear_all_points)
        self.invert_mask_checkbox.stateChanged.connect(self._refresh_preview)
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self.result_list.currentRowChanged.connect(self._handle_result_selection)
        self.save_current_button.clicked.connect(self._save_current_result)
        self.save_all_button.clicked.connect(self._save_all_results)
        self.model_variant_combo.currentTextChanged.connect(self._apply_variant_selection)
        self.model_variant_combo.currentTextChanged.connect(self._update_settings_summary)
        self.base_model_edit.textChanged.connect(self._update_settings_summary)
        self.brushnet_path_edit.textChanged.connect(self._update_settings_summary)
        self.control_strength_spin.valueChanged.connect(self._update_settings_summary)
        self.guidance_scale_spin.valueChanged.connect(self._update_settings_summary)
        self.steps_spin.valueChanged.connect(self._update_settings_summary)
        self.outputs_spin.valueChanged.connect(self._update_settings_summary)
        self.seed_spin.valueChanged.connect(self._update_settings_summary)
        self.randomize_seed_checkbox.stateChanged.connect(self._update_settings_summary)
        self.blending_checkbox.stateChanged.connect(self._update_settings_summary)
        self.run_button.clicked.connect(self._run_inference)

    def _build_settings_dialog(self) -> None:
        # 参数对话框与主界面解耦，减少主窗口信息密度。
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("参数设置")
        self.settings_dialog.resize(680, 520)

        dialog_layout = QVBoxLayout(self.settings_dialog)
        dialog_layout.setContentsMargins(18, 18, 18, 18)
        dialog_layout.setSpacing(12)

        header = QLabel("推理参数与模型路径")
        header.setObjectName("TitleLabel")
        dialog_layout.addWidget(header)

        tabs = QTabWidget()
        dialog_layout.addWidget(tabs, 1)

        model_page = QWidget()
        model_form = QFormLayout(model_page)
        model_form.setContentsMargins(12, 14, 12, 12)
        model_form.setSpacing(12)
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(list(core.get_brushnet_variant_map(self.default_base_model).keys()))
        self.base_model_edit = QLineEdit(self.default_base_model)
        self.brushnet_path_edit = QLineEdit(self.default_brushnet_path)
        model_form.addRow("模型预设", self.model_variant_combo)
        model_form.addRow("基础模型路径 / Hugging Face ID", self.base_model_edit)
        model_form.addRow("BrushNet 权重路径", self.brushnet_path_edit)
        tabs.addTab(model_page, "模型")

        generation_page = QWidget()
        generation_grid = QGridLayout(generation_page)
        generation_grid.setContentsMargins(12, 14, 12, 12)
        generation_grid.setHorizontalSpacing(12)
        generation_grid.setVerticalSpacing(12)

        self.control_strength_spin = QDoubleSpinBox()
        self.control_strength_spin.setRange(0.0, 1.2)
        self.control_strength_spin.setSingleStep(0.01)
        self.control_strength_spin.setValue(1.0)

        self.guidance_scale_spin = QDoubleSpinBox()
        self.guidance_scale_spin.setRange(1.0, 15.0)
        self.guidance_scale_spin.setSingleStep(0.1)
        self.guidance_scale_spin.setValue(7.5)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 60)
        self.steps_spin.setValue(50)

        self.outputs_spin = QSpinBox()
        self.outputs_spin.setRange(1, 4)
        self.outputs_spin.setValue(1)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(1234)

        self.randomize_seed_checkbox = QCheckBox("每次生成随机种子")
        self.blending_checkbox = QCheckBox("启用边缘模糊融合")

        generation_grid.addWidget(QLabel("Control Strength"), 0, 0)
        generation_grid.addWidget(self.control_strength_spin, 0, 1)
        generation_grid.addWidget(QLabel("Guidance Scale"), 0, 2)
        generation_grid.addWidget(self.guidance_scale_spin, 0, 3)
        generation_grid.addWidget(QLabel("Inference Steps"), 1, 0)
        generation_grid.addWidget(self.steps_spin, 1, 1)
        generation_grid.addWidget(QLabel("生成结果数"), 1, 2)
        generation_grid.addWidget(self.outputs_spin, 1, 3)
        generation_grid.addWidget(QLabel("Seed"), 2, 0)
        generation_grid.addWidget(self.seed_spin, 2, 1)
        generation_grid.addWidget(self.randomize_seed_checkbox, 2, 2, 1, 2)
        generation_grid.addWidget(self.blending_checkbox, 3, 0, 1, 2)
        generation_grid.setColumnStretch(1, 1)
        generation_grid.setColumnStretch(3, 1)
        generation_grid.setRowStretch(4, 1)
        tabs.addTab(generation_page, "生成")

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.settings_dialog.close)
        dialog_layout.addWidget(button_box)

        self._update_settings_summary()

    def _open_settings_dialog(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _update_settings_summary(self, *_) -> None:
        if not hasattr(self, "settings_summary_label"):
            return
        seed_text = "随机" if self.randomize_seed_checkbox.isChecked() else str(self.seed_spin.value())
        blending_text = "开启" if self.blending_checkbox.isChecked() else "关闭"
        self.settings_summary_label.setText(
            "当前参数："
            f"{self.model_variant_combo.currentText()} | "
            f"结果 {self.outputs_spin.value()} 张 | "
            f"Steps {self.steps_spin.value()} | "
            f"CFG {self.guidance_scale_spin.value():.1f} | "
            f"Control {self.control_strength_spin.value():.2f} | "
            f"Seed {seed_text} | "
            f"融合 {blending_text}"
        )

    def _set_initial_status(self) -> None:
        self._apply_variant_selection(self.model_variant_combo.currentText())

    def _current_base_model(self) -> str:
        return self.base_model_edit.text().strip() or self.default_base_model

    def _current_brushnet_path(self) -> str:
        return self.brushnet_path_edit.text().strip() or self.default_brushnet_path

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "BrushNet", message)

    def _set_status(self, status: str) -> None:
        return

    def _set_gallery_results(self, images: Sequence[Image.Image]) -> None:
        self.result_list.clear()
        self.current_results = list(images)
        for index, image in enumerate(images, start=1):
            pixmap = pil_to_qpixmap(image)
            scaled = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(scaled), f"结果 {index}")
            item.setSizeHint(QSize(136, 140))
            self.result_list.addItem(item)
        if images:
            self.result_list.setCurrentRow(0)
            self.compare_result.set_numpy_image(np.array(images[0].convert("RGB")))

    def _clear_gallery_results(self) -> None:
        self.current_results = []
        self.result_list.clear()
        self.compare_result.clear_image()
        self.compare_original.clear_image()

    def _handle_result_selection(self, row: int) -> None:
        if 0 <= row < len(self.current_results):
            self.compare_result.set_numpy_image(
                np.array(self.current_results[row].convert("RGB"))
            )

    def _save_current_result(self) -> None:
        if not self.current_results:
            QMessageBox.information(self, "提示", "暂无生成结果可保存。")
            return
        row = self.result_list.currentRow()
        image = self.current_results[row if row >= 0 else 0]
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前结果", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if path:
            image.save(path)

    def _save_all_results(self) -> None:
        if not self.current_results:
            QMessageBox.information(self, "提示", "暂无生成结果可保存。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not folder:
            return
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, image in enumerate(self.current_results):
            image.save(f"{folder}/result_{ts}_{i + 1}.png")
        QMessageBox.information(
            self, "完成", f"已保存 {len(self.current_results)} 张结果到：\n{folder}"
        )

    def _choose_source_image(self) -> None:
        # 载入原图后，重置所有与掩膜和结果相关的状态。
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            str(core.PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        try:
            image = load_image_from_path(path)
            payload = core.prepare_source_image(image, self._current_base_model(), self._current_brushnet_path())
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:
            self._show_error(f"读取图片失败：{exc}")
            return

        self.original_image = payload["image"]  # type: ignore[assignment]
        self.original_mask = None
        self.uploaded_mask = None
        self.selected_points = []
        self.input_canvas.set_numpy_image(payload["overlay"])  # type: ignore[arg-type]
        self.mask_preview.clear_image()
        self.masked_preview.set_numpy_image(self.original_image)
        self._clear_gallery_results()
        self._set_status(payload["status"])  # type: ignore[arg-type]
        self.outputs_spin.setValue(1)
        self.steps_spin.setValue(50)
        self.seed_spin.setValue(1234)
        self.control_strength_spin.setValue(1.0)
        self.guidance_scale_spin.setValue(7.5)
        self.randomize_seed_checkbox.setChecked(False)
        self.model_variant_combo.setCurrentText("分割掩码模型")
        self._update_settings_summary()

    def _choose_mask_image(self) -> None:
        # 上传外部掩膜后，会清空点选分割状态，避免两套掩膜来源混用。
        if self.original_image is None:
            self._show_error("请先上传原始图片，再上传黑白掩码。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择黑白掩码图",
            str(core.PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        try:
            self.uploaded_mask = load_image_from_path(path)
            self.original_mask = None
            self.selected_points = []
            preview = core.build_preview_from_uploaded_mask(
                self.original_image,
                self.uploaded_mask,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:
            self._show_error(f"读取掩码失败：{exc}")
            return

        self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]
        self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]
        self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]
        self._clear_gallery_results()
        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _clear_uploaded_mask(self) -> None:
        if self.uploaded_mask is None:
            return
        self.uploaded_mask = None
        self._refresh_preview()

    def _handle_canvas_click(self, x: int, y: int) -> None:
        # 用户在图像上点前景/背景点后，实时调用 core 层生成新的 SAM 预览。
        if self.original_image is None:
            return

        label = 1 if self.point_mode_combo.currentText() == "添加区域" else 0
        self.selected_points.append(((x, y), label))
        self.uploaded_mask = None

        try:
            preview = core.build_preview_from_points(
                self.original_image,
                self.selected_points,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            if self.selected_points:
                self.selected_points.pop()
            return

        self.original_mask = preview["raw_mask"]  # type: ignore[assignment]
        self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]
        self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]
        self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]
        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _undo_last_point(self) -> None:
        if not self.selected_points:
            return
        self.selected_points.pop()
        if not self.selected_points:
            self.original_mask = None
        self._refresh_preview()

    def _clear_all_points(self) -> None:
        if not self.selected_points and self.original_mask is None:
            return
        self.selected_points = []
        self.original_mask = None
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        # 所有“状态变化但尚未推理”的场景，统一走这个刷新逻辑。
        try:
            preview = core.refresh_mask_preview(
                self.original_image,
                self.original_mask,
                self.uploaded_mask,
                self.selected_points,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return

        if preview["overlay"] is None:
            self.input_canvas.clear_image()
        else:
            self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]

        if preview["effective_mask"] is None:
            self.mask_preview.clear_image()
        else:
            self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]

        if preview["masked_image"] is None:
            self.masked_preview.clear_image()
        else:
            self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]

        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _apply_variant_selection(self, variant_name: str) -> None:
        # 根据下拉框中的预设方案，自动填充模型路径。
        variant_map = core.get_brushnet_variant_map(self._current_base_model())
        selected = variant_map.get(variant_name)
        if not selected:
            return
        self.base_model_edit.setText(selected["base_model"])
        self.brushnet_path_edit.setText(selected["brushnet_path"])

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(enabled)
        self.settings_button.setEnabled(enabled)
        self.open_image_button.setEnabled(enabled)
        self.open_mask_button.setEnabled(enabled)

    def _run_inference(self) -> None:
        # 推理走后台线程，主线程只负责禁用控件和显示执行状态。
        self._clear_gallery_results()
        self._set_controls_enabled(False)
        self.run_button.setText("生成中...")
        self._set_status(
            core.summarize_status(
                "模型推理已启动，请稍候。",
                self._current_base_model(),
                self._current_brushnet_path(),
                mask_source="处理中",
            )
        )

        self.worker = InferenceWorker(
            input_image=self.original_image,
            original_image=self.original_image,
            original_mask=self.original_mask,
            uploaded_mask=self.uploaded_mask,
            selected_points=self.selected_points,
            prompt=self.prompt_edit.toPlainText(),
            negative_prompt=self.negative_prompt_edit.toPlainText(),
            blending=self.blending_checkbox.isChecked(),
            invert_mask=self.invert_mask_checkbox.isChecked(),
            control_strength=self.control_strength_spin.value(),
            seed=self.seed_spin.value(),
            randomize_seed=self.randomize_seed_checkbox.isChecked(),
            guidance_scale=self.guidance_scale_spin.value(),
            num_inference_steps=self.steps_spin.value(),
            num_outputs=self.outputs_spin.value(),
            base_model=self._current_base_model(),
            brushnet_path=self._current_brushnet_path(),
        )
        self.worker.succeeded.connect(self._handle_inference_success)
        self.worker.failed.connect(self._handle_inference_failure)
        self.worker.finished.connect(self._handle_worker_finished)
        self.worker.start()

    def _handle_inference_success(self, results, effective_mask, masked_image, status: str) -> None:
        if effective_mask is not None and self.original_image is not None:
            overlay_points = self.selected_points if self.uploaded_mask is None else None
            overlay = core.build_overlay_image(self.original_image, effective_mask, overlay_points)
            self.input_canvas.set_numpy_image(overlay)
            self.mask_preview.set_numpy_image(effective_mask)
            self.masked_preview.set_numpy_image(masked_image)
            # 左侧对比区显示原图（含掩码叠加）
            self.compare_original.set_numpy_image(overlay)
        self._set_gallery_results(results)
        self._set_status(status)

    def _handle_inference_failure(self, message: str) -> None:
        self._show_error(message)
        self._set_status(
            core.summarize_status(
                message,
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        )

    def _handle_worker_finished(self) -> None:
        self._set_controls_enabled(True)
        self.run_button.setText("开始生成")
        self.worker = None


def parse_args() -> argparse.Namespace:
    # 允许从命令行覆盖默认模型路径，便于调试和切换权重。
    parser = argparse.ArgumentParser(description="BrushNet PyQt5 desktop UI")
    parser.add_argument("--base-model", default=core.resolve_default_base_model(), help="Base model path or model id")
    parser.add_argument("--brushnet-path", default=core.resolve_default_brushnet_path(), help="BrushNet checkpoint path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("BrushNet PyQt5")
    app.setFont(QFont("Microsoft YaHei", 10))

    window = BrushNetQtWindow(args.base_model, args.brushnet_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())