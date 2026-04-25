#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler
from segment_anything import SamPredictor, sam_model_registry

os.chdir(PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAM_CHECKPOINT = PROJECT_ROOT / "data" / "ckpt" / "sam_vit_h_4b8939.pth"
DEFAULT_REMOTE_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DISABLE_SAFETY_CHECKER = True

POINT_COLORS = [(28, 163, 131), (215, 82, 54)]
POINT_MARKERS = [1, 5]

SEGMENTATION_BRUSHNET_DIR = PROJECT_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
RANDOM_MASK_BRUSHNET_DIR = PROJECT_ROOT / "data" / "ckpt" / "random_mask_brushnet_ckpt"

PIPELINE_CACHE: Dict[str, Optional[object]] = {
    "pipe": None,
    "base_model": None,
    "brushnet_path": None,
}
PIPELINE_LOCK = threading.Lock()
SAM_STATE: Dict[str, Optional[object]] = {"model": None, "predictor": None}

PointRecord = Tuple[Tuple[int, int], int]


class BrushNetAppError(RuntimeError):
    pass


EXAMPLE_CASES = [
    {
        "label": "蛋糕桌面编辑",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "test_image.jpg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "test_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "test_result.png",
        "prompt": "A beautiful cake on the table",
        "negative_prompt": "",
        "scene": "适合观察局部替换与背景一致性",
    },
    {
        "label": "人物服饰编辑",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_1.jpg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_1_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_1_result.png",
        "prompt": "A man in Chinese traditional clothes",
        "negative_prompt": "",
        "scene": "适合展示人物局部换装",
    },
    {
        "label": "海边人物增强",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_2.jpg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_2_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_2_result.png",
        "prompt": "a charming woman with dress standing by the sea",
        "negative_prompt": "",
        "scene": "适合展示语义驱动的人像编辑",
    },
    {
        "label": "玩具目标替换",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_3.jpg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_3_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_3_result.png",
        "prompt": "a cute toy on the table",
        "negative_prompt": "",
        "scene": "适合展示小目标修复与纹理生成",
    },
    {
        "label": "车辆场景编辑",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_4.jpeg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_4_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_4_result.png",
        "prompt": "a car driving in the wild",
        "negative_prompt": "",
        "scene": "适合展示复杂背景中的目标重绘",
    },
    {
        "label": "森林人像重绘",
        "image": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_5.jpg",
        "mask": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_5_mask.jpg",
        "result": PROJECT_ROOT / "examples" / "brushnet" / "src" / "example_5_result.png",
        "prompt": "a charming woman wearing dress standing in the dark forest",
        "negative_prompt": "",
        "scene": "适合展示光照与氛围迁移",
    },
]


def read_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise BrushNetAppError(f"找不到图片文件：{path}")
    return np.array(Image.open(path).convert("RGB"))


def open_rgb_pil(path: Path) -> Image.Image:
    if not path.exists():
        raise BrushNetAppError(f"找不到图片文件：{path}")
    return Image.open(path).convert("RGB")


def normalize_model_reference(value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        return stripped

    candidate = Path(stripped)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if candidate.exists():
        return str(candidate)
    return stripped


def resolve_default_brushnet_path() -> str:
    preferred = PROJECT_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
    fallback = PROJECT_ROOT / "data" / "ckpt"
    fallback_weight_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )

    if preferred.exists():
        return str(preferred)

    if (fallback / "config.json").exists() and any((fallback / name).exists() for name in fallback_weight_names):
        return str(fallback)

    return str(preferred)


def resolve_default_base_model() -> str:
    env_value = os.environ.get("BRUSHNET_BASE_MODEL")
    if env_value:
        return env_value

    ckpt_dir = PROJECT_ROOT / "data" / "ckpt"
    candidate_names = [
        "realisticVisionV60B1_v51VAE",
        "stable-diffusion-v1-5",
        "runwayml--stable-diffusion-v1-5",
    ]

    for name in candidate_names:
        candidate = ckpt_dir / name
        if (candidate / "model_index.json").exists():
            return str(candidate)

    for candidate in ckpt_dir.glob("*"):
        if candidate.is_dir() and (candidate / "model_index.json").exists():
            return str(candidate)

    return DEFAULT_REMOTE_BASE_MODEL


def get_brushnet_variant_map(base_model: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    resolved_base_model = normalize_model_reference(base_model or "") or resolve_default_base_model()
    default_brushnet = normalize_model_reference(resolve_default_brushnet_path())
    segmentation_path = str(SEGMENTATION_BRUSHNET_DIR) if SEGMENTATION_BRUSHNET_DIR.exists() else default_brushnet
    random_mask_path = str(RANDOM_MASK_BRUSHNET_DIR) if RANDOM_MASK_BRUSHNET_DIR.exists() else default_brushnet

    return {
        "分割掩码模型": {
            "base_model": resolved_base_model,
            "brushnet_path": segmentation_path,
        },
        "随机掩码模型": {
            "base_model": resolved_base_model,
            "brushnet_path": random_mask_path,
        },
    }


def summarize_status(
    message: str,
    base_model: str,
    brushnet_path: str,
    mask_source: str = "未生成",
    actual_seed: Optional[int] = None,
    elapsed: Optional[float] = None,
) -> str:
    resolved_base = normalize_model_reference(base_model) or resolve_default_base_model()
    resolved_brushnet = normalize_model_reference(brushnet_path) or resolve_default_brushnet_path()

    lines = [
        "系统状态",
        f"设备: {DEVICE}",
        f"基础模型: {resolved_base}",
        f"BrushNet 权重: {resolved_brushnet}",
        f"SAM 权重: {'已检测到' if SAM_CHECKPOINT.exists() else '未检测到'}",
        f"安全审查: {'已禁用' if DISABLE_SAFETY_CHECKER else '已启用'}",
        f"掩码来源: {mask_source}",
    ]
    if actual_seed is not None:
        lines.append(f"随机种子: {actual_seed}")
    if elapsed is not None:
        lines.append(f"推理耗时: {elapsed:.2f}s")
    lines.append("")
    lines.append(f"说明: {message}")
    return "\n".join(lines)


def ensure_uint8_rgb(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None:
        return None
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_for_workflow(image: np.ndarray, max_short_side: int = 768) -> np.ndarray:
    image = ensure_uint8_rgb(image)
    assert image is not None

    height, width = image.shape[:2]
    short_side = min(height, width)
    if short_side <= max_short_side:
        return image

    scale = max_short_side / float(short_side)
    new_height = int(np.round(height * scale / 64.0)) * 64
    new_width = int(np.round(width * scale / 64.0)) * 64
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def validate_source_image(image: np.ndarray) -> np.ndarray:
    image = resize_for_workflow(image)
    height, width = image.shape[:2]
    ratio = max(height, width) / max(1, min(height, width))
    if ratio > 2.0:
        raise BrushNetAppError("图片长宽比不能超过 2:1，建议先裁剪后再演示。")
    return image


def build_binary_mask(mask_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    mask_image = ensure_uint8_rgb(mask_image)
    assert mask_image is not None

    target_width, target_height = target_size
    if mask_image.shape[0] != target_height or mask_image.shape[1] != target_width:
        mask_image = cv2.resize(mask_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Keep mask semantics aligned with the original Gradio demo and test_brushnet.py:
    # any sufficiently bright RGB pixel is treated as white mask.
    binary = np.where(mask_image.sum(axis=-1) > 255, 255, 0).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def maybe_invert_mask(mask_image: np.ndarray, invert_mask: bool) -> np.ndarray:
    return 255 - mask_image if invert_mask else mask_image


def build_overlay_image(
    image: np.ndarray,
    edit_mask: np.ndarray,
    points: Optional[Sequence[PointRecord]] = None,
) -> np.ndarray:
    image = ensure_uint8_rgb(image)
    assert image is not None

    overlay = image.astype(np.float32).copy()
    binary = edit_mask[:, :, 0] > 127
    tint = np.array([245, 134, 52], dtype=np.float32)
    overlay[binary] = overlay[binary] * 0.35 + tint * 0.65

    if points:
        for point, label in points:
            draw_point = tuple(int(v) for v in point)
            cv2.drawMarker(
                overlay,
                draw_point,
                POINT_COLORS[label],
                markerType=POINT_MARKERS[label],
                markerSize=18,
                thickness=4,
            )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_masked_image(image: np.ndarray, edit_mask: np.ndarray) -> np.ndarray:
    image = ensure_uint8_rgb(image)
    assert image is not None

    binary = (edit_mask[:, :, 0] > 127).astype(np.float32)[:, :, None]
    return np.clip(image * (1.0 - binary), 0, 255).astype(np.uint8)


def clear_cached_pipeline() -> None:
    pipe = PIPELINE_CACHE.get("pipe")
    if pipe is not None:
        PIPELINE_CACHE["pipe"] = None
        PIPELINE_CACHE["base_model"] = None
        PIPELINE_CACHE["brushnet_path"] = None
        del pipe
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _brushnet_checkpoint_looks_valid(path_value: str) -> bool:
    candidate = Path(path_value)
    if not candidate.exists() or not candidate.is_dir():
        return False

    weight_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )
    return (candidate / "config.json").exists() and any((candidate / name).exists() for name in weight_names)


def _base_model_looks_valid(path_value: str) -> bool:
    candidate = Path(path_value)
    return candidate.exists() and candidate.is_dir() and (candidate / "model_index.json").exists()


def ensure_pipeline(base_model: str, brushnet_path: str) -> Tuple[StableDiffusionBrushNetPipeline, bool, str, str]:
    resolved_base = normalize_model_reference(base_model) or resolve_default_base_model()
    resolved_brushnet = normalize_model_reference(brushnet_path) or resolve_default_brushnet_path()

    with PIPELINE_LOCK:
        cached_pipe = PIPELINE_CACHE.get("pipe")
        if (
            cached_pipe is not None
            and PIPELINE_CACHE.get("base_model") == resolved_base
            and PIPELINE_CACHE.get("brushnet_path") == resolved_brushnet
        ):
            return cached_pipe, False, resolved_base, resolved_brushnet

        clear_cached_pipeline()
        if not _brushnet_checkpoint_looks_valid(resolved_brushnet):
            raise BrushNetAppError(
                "当前 BrushNet 权重路径无效。\n"
                f"检测到的路径: {resolved_brushnet}\n"
                "这个目录里应至少包含 `config.json` 和 `diffusion_pytorch_model.safetensors`。"
            )

        base_candidate = Path(resolved_base)
        if base_candidate.exists() and not _base_model_looks_valid(resolved_base):
            raise BrushNetAppError(
                "当前基础模型路径无效。\n"
                f"检测到的路径: {resolved_base}\n"
                "这个目录里应包含 `model_index.json`，也就是一个完整的 diffusers 基础模型目录。"
            )

        try:
            brushnet = BrushNetModel.from_pretrained(resolved_brushnet, torch_dtype=WEIGHT_DTYPE)
            pipe_kwargs = {
                "brushnet": brushnet,
                "torch_dtype": WEIGHT_DTYPE,
                "low_cpu_mem_usage": False,
            }
            if DISABLE_SAFETY_CHECKER:
                pipe_kwargs["safety_checker"] = None
                pipe_kwargs["requires_safety_checker"] = False
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                resolved_base,
                **pipe_kwargs,
            )
        except Exception as exc:
            raise BrushNetAppError(
                "BrushNet 推理管线加载失败。\n"
                f"基础模型: {resolved_base}\n"
                f"BrushNet 权重: {resolved_brushnet}\n"
                f"底层报错: {exc}"
            ) from exc

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        PIPELINE_CACHE["pipe"] = pipe
        PIPELINE_CACHE["base_model"] = resolved_base
        PIPELINE_CACHE["brushnet_path"] = resolved_brushnet
        return pipe, True, resolved_base, resolved_brushnet


def ensure_sam_predictor() -> SamPredictor:
    predictor = SAM_STATE.get("predictor")
    if predictor is not None:
        return predictor

    if not SAM_CHECKPOINT.exists():
        raise BrushNetAppError(
            "没有找到 SAM 权重 data/ckpt/sam_vit_h_4b8939.pth。没有它时可以改为上传黑白掩码。"
        )

    try:
        sam_model = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).to(DEVICE)
        sam_model.eval()
        predictor = SamPredictor(sam_model)
    except Exception as exc:
        raise BrushNetAppError(
            "SAM 权重加载失败，通常是文件损坏或下载不完整。请重新下载 sam_vit_h_4b8939.pth。"
        ) from exc

    SAM_STATE["model"] = sam_model
    SAM_STATE["predictor"] = predictor
    return predictor


def infer_mask_from_points(image: np.ndarray, selected_points: Sequence[PointRecord]) -> np.ndarray:
    if not selected_points:
        raise BrushNetAppError("当前还没有点击点，无法生成分割掩码。")

    predictor = ensure_sam_predictor()
    point_coords = np.array([point for point, _ in selected_points], dtype=np.float32)
    point_labels = np.array([label for _, label in selected_points], dtype=np.int32)
    predictor.set_image(image)
    with torch.no_grad():
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    binary = np.where(masks[0], 255, 0).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def derive_effective_mask(
    original_image: np.ndarray,
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
) -> Tuple[np.ndarray, str]:
    if uploaded_mask is not None:
        raw_mask = build_binary_mask(uploaded_mask, (original_image.shape[1], original_image.shape[0]))
        return maybe_invert_mask(raw_mask, invert_mask), "上传掩码"

    if original_mask is not None:
        raw_mask = build_binary_mask(original_mask, (original_image.shape[1], original_image.shape[0]))
        mask_source = "SAM 点击分割" if selected_points else "示例掩码"
        return maybe_invert_mask(raw_mask, invert_mask), mask_source

    if selected_points:
        raw_mask = infer_mask_from_points(original_image, selected_points)
        return maybe_invert_mask(raw_mask, invert_mask), "SAM 点击分割"

    raise BrushNetAppError("请先上传黑白掩码，或在图片上点击选择要编辑的区域。")


def load_example_case(case_label: str, base_model: str, brushnet_path: str) -> Dict[str, object]:
    case = next((item for item in EXAMPLE_CASES if item["label"] == case_label), None)
    if case is None:
        raise BrushNetAppError("没有找到对应的示例。")

    image = validate_source_image(read_rgb_image(case["image"]))
    raw_mask = build_binary_mask(read_rgb_image(case["mask"]), (image.shape[1], image.shape[0]))
    overlay = build_overlay_image(image, raw_mask)
    masked_image = build_masked_image(image, raw_mask)
    status = summarize_status(
        f"已加载示例：{case['label']}。这类场景 {case['scene']}。",
        base_model,
        brushnet_path,
        mask_source="示例掩码",
    )
    return {
        "image": image,
        "raw_mask": raw_mask,
        "overlay": overlay,
        "masked_image": masked_image,
        "prompt": case["prompt"],
        "negative_prompt": case["negative_prompt"],
        "results": [open_rgb_pil(case["result"])],
        "status": status,
    }


def prepare_source_image(image: np.ndarray, base_model: str, brushnet_path: str) -> Dict[str, object]:
    prepared_image = validate_source_image(image)
    status = summarize_status(
        "图片已载入。下一步可以点击图像做 SAM 分割，或者直接上传一张白色区域表示待编辑区域的掩码图。",
        base_model,
        brushnet_path,
        mask_source="未生成",
    )
    return {
        "image": prepared_image,
        "overlay": prepared_image,
        "status": status,
    }


def build_preview_from_uploaded_mask(
    original_image: np.ndarray,
    uploaded_mask: np.ndarray,
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    if original_image is None:
        raise BrushNetAppError("请先上传原始图片，再上传掩码。")

    original_image = ensure_uint8_rgb(original_image)
    assert original_image is not None

    effective_mask = maybe_invert_mask(
        build_binary_mask(uploaded_mask, (original_image.shape[1], original_image.shape[0])),
        invert_mask,
    )
    overlay = build_overlay_image(original_image, effective_mask)
    masked_image = build_masked_image(original_image, effective_mask)
    status = summarize_status(
        "已根据上传掩码生成编辑区域预览。白色区域表示会被重新生成的区域。",
        base_model,
        brushnet_path,
        mask_source="上传掩码",
    )
    return {
        "overlay": overlay,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def build_preview_from_points(
    original_image: np.ndarray,
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    if original_image is None:
        raise BrushNetAppError("请先上传图片或加载示例。")
    if not selected_points:
        raise BrushNetAppError("当前没有点击点，无法生成分割结果。")

    original_image = ensure_uint8_rgb(original_image)
    assert original_image is not None

    raw_mask = infer_mask_from_points(original_image, selected_points)
    effective_mask = maybe_invert_mask(raw_mask, invert_mask)
    overlay = build_overlay_image(original_image, effective_mask, selected_points)
    masked_image = build_masked_image(original_image, effective_mask)
    status = summarize_status(
        "已更新点击分割结果。绿色点表示添加区域，红色点表示排除区域。",
        base_model,
        brushnet_path,
        mask_source="SAM 点击分割",
    )
    return {
        "overlay": overlay,
        "raw_mask": raw_mask,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def refresh_mask_preview(
    original_image: Optional[np.ndarray],
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    if original_image is None:
        status = summarize_status("等待上传图片。", base_model, brushnet_path)
        return {
            "overlay": None,
            "effective_mask": None,
            "masked_image": None,
            "status": status,
        }

    image = ensure_uint8_rgb(original_image)
    assert image is not None

    try:
        effective_mask, mask_source = derive_effective_mask(
            image,
            original_mask,
            uploaded_mask,
            selected_points,
            invert_mask,
        )
    except BrushNetAppError:
        status = summarize_status(
            "当前没有可更新的掩码，等待新的点击或上传掩码。",
            base_model,
            brushnet_path,
        )
        return {
            "overlay": image,
            "effective_mask": None,
            "masked_image": image,
            "status": status,
        }

    overlay_points = selected_points if mask_source == "SAM 点击分割" and uploaded_mask is None else None
    overlay = build_overlay_image(image, effective_mask, overlay_points)
    masked_image = build_masked_image(image, effective_mask)
    status = summarize_status(
        "已根据当前设置刷新掩码语义。白色区域表示将被重新生成。",
        base_model,
        brushnet_path,
        mask_source=mask_source,
    )
    return {
        "overlay": overlay,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def run_brushnet_inference(
    input_image: Optional[np.ndarray],
    original_image: Optional[np.ndarray],
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
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
) -> Tuple[List[Image.Image], np.ndarray, np.ndarray, str]:
    source_image = ensure_uint8_rgb(original_image if original_image is not None else input_image)
    if source_image is None:
        raise BrushNetAppError("请先上传图片。")
    if not (prompt or "").strip():
        raise BrushNetAppError("请输入提示词后再开始生成。")

    effective_mask, mask_source = derive_effective_mask(
        source_image,
        original_mask,
        uploaded_mask,
        selected_points,
        invert_mask,
    )
    masked_image = build_masked_image(source_image, effective_mask)

    if blending and control_strength < 1.0:
        raise BrushNetAppError("启用模糊融合时，建议将 Control Strength 设为 1.0 或更高。")

    pipe, _, resolved_base, resolved_brushnet = ensure_pipeline(base_model, brushnet_path)
    actual_seed = random.randint(0, 2_147_483_647) if randomize_seed else int(seed)
    generator = torch.Generator(DEVICE).manual_seed(actual_seed)

    init_image = Image.fromarray(masked_image).convert("RGB")
    mask_image = Image.fromarray(effective_mask).convert("RGB")

    prompt_batch = [prompt.strip()] * int(num_outputs)
    negative_text = (negative_prompt or "").strip()
    negative_batch = [negative_text] * int(num_outputs) if negative_text else None

    start = time.perf_counter()
    results = pipe(
        prompt_batch,
        init_image,
        mask_image,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        brushnet_conditioning_scale=float(control_strength),
        negative_prompt=negative_batch,
    ).images
    elapsed = time.perf_counter() - start

    if blending:
        blended_results = []
        mask_np = (effective_mask[:, :, 0] > 127).astype(np.float32)[:, :, None]
        mask_blurred = cv2.GaussianBlur(mask_np[:, :, 0] * 255, (21, 21), 0) / 255.0
        soft_mask = 1 - (1 - mask_np) * (1 - mask_blurred[:, :, None])
        for image in results:
            image_np = np.array(image)
            pasted = source_image * (1 - soft_mask) + image_np * soft_mask
            blended_results.append(Image.fromarray(np.clip(pasted, 0, 255).astype(np.uint8)))
        results = blended_results

    status = summarize_status(
        "推理完成。右侧结果区域展示了 BrushNet 生成结果，你可以继续修改提示词或掩码后再次生成。",
        resolved_base,
        resolved_brushnet,
        mask_source=mask_source,
        actual_seed=actual_seed,
        elapsed=elapsed,
    )
    return results, effective_mask, masked_image, status
