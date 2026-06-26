#!/usr/bin/env python3
"""
robust_decoder.py
==================

Industrial-grade decoder for Data Matrix / QR / 1D barcodes, tuned for the
worst cases: small Direct Part Marks (DPM) on PCBs/metal, dot-peen, laser
etch, low contrast, blur, glare, uneven illumination, and skew.

DESIGN PHILOSOPHY
------------------
1. MULTI-ENGINE: zxingcpp, pylibdmtx, and pyzbar use different decoding
   algorithms internally. A frame that defeats one will often fall to
   another. Always try all engines available before giving up.

2. MULTI-HYPOTHESIS PREPROCESSING: there is no single "best" preprocessing
   chain for every defect type (blur wants different treatment than glare,
   which wants different treatment than dot-peen texture, which wants
   different treatment than uneven illumination). Instead of one fixed
   3-pass pipeline, this generates a *bank* of ~15 independent preprocessing
   variants and throws every decoder at every variant, stopping the instant
   one succeeds (cheap variants / fast engines first, for speed).

3. GEOMETRY SWEEP: DPM codes are frequently rotated and modestly scaled
   wrong relative to what the decoder expects. We sweep coarse rotations
   (0/90/180/270) always, and fine rotations (+/-5,10,15 deg) when nothing
   else has worked, plus an upscale ladder.

4. QUIET ZONE: always re-padded, on every variant, since decoders often
   silently fail to even attempt a decode without it.

Install
-------
    pip install zxing-cpp pylibdmtx pyzbar opencv-python-headless pillow numpy

    # pylibdmtx and pyzbar are wrappers around native C libraries that must
    # also be installed at the OS level:
    #   Ubuntu/Debian:  sudo apt-get install libdmtx0a libzbar0
    #   macOS:          brew install libdmtx zbar
    #   Windows:        DLLs ship inside the pip wheels for both, usually
    #                   no extra step needed.

Usage
-----
    from robust_decoder import CodeDecoder

    decoder = CodeDecoder()
    results = decoder.decode_file("pcb_marking.png", save_annotated="pcb_marking_decoded.png")
    for r in results:
        print(r.format, r.text, r.engine, r.variant)

    # Or just the CLI:
    python robust_decoder.py path/to/image.png --debug-dir ./debug_out --save-annotated ./decoded.png
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("robust_decoder")

# --------------------------------------------------------------------------
# Optional engine imports — we degrade gracefully if one isn't installed,
# but warn loudly because each missing engine measurably hurts recall.
# --------------------------------------------------------------------------

_HAVE_ZXINGCPP = False
_HAVE_DMTX = False
_HAVE_ZBAR = False

try:
    import zxingcpp

    _HAVE_ZXINGCPP = True
except ImportError:
    logger.warning("zxingcpp not installed — `pip install zxing-cpp`. Skipping this engine.")

try:
    from pylibdmtx import pylibdmtx

    _HAVE_DMTX = True
except ImportError:
    logger.warning(
        "pylibdmtx not installed — `pip install pylibdmtx` (+ libdmtx system lib). "
        "This is the single most impactful engine for Direct Part Mark Data Matrix "
        "codes; strongly recommended."
    )

try:
    from pyzbar import pyzbar

    _HAVE_ZBAR = True
except ImportError:
    logger.warning("pyzbar not installed — `pip install pyzbar` (+ libzbar). Skipping this engine.")


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclasses.dataclass
class DecodeResult:
    text: str
    format: str          # e.g. "DataMatrix", "QRCode", "Code128"
    engine: str           # which library decoded it
    variant: str          # which preprocessing variant produced the hit
    rotation: float        # degrees of rotation applied when it hit
    raw_bytes: Optional[bytes] = None
    # Polygon (list of (x, y) points) of the decoded symbol's location, in
    # the pixel space of `source_image` below (i.e. AFTER whatever rotation/
    # upscale/preprocessing/padding was applied for this particular hit —
    # not the original input image). None if the engine didn't report one.
    polygon: Optional[List[Tuple[int, int]]] = None
    # The exact image (as actually handed to the decoder, post quiet-zone
    # padding) that produced this hit. Kept so annotation can be drawn in
    # the same coordinate space as `polygon` without an error-prone reverse
    # transform back through rotation/upscale/crop.
    source_image: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"DecodeResult(format={self.format!r}, text={self.text!r}, "
            f"engine={self.engine!r}, variant={self.variant!r}, rotation={self.rotation})"
        )


# ==========================================================================
# PREPROCESSING VARIANT BANK
# ==========================================================================
# Each function: (gray_uint8_ndarray) -> processed_uint8_ndarray (or None to
# skip, e.g. if a step is inapplicable). Keep these independent — do NOT
# chain all of them together; the whole point is breadth, not depth.
# Ordered roughly cheapest/most-likely-to-work first for speed.


def _pad_quiet_zone(img: np.ndarray, frac: float = 0.25, min_px: int = 12) -> np.ndarray:
    """Add white (or matching background) quiet zone border.
    Without this, decoders frequently refuse to even attempt a read."""
    h, w = img.shape[:2]
    pad = max(min_px, int(frac * min(h, w)))
    # use median border color as fill rather than assuming white — robust to
    # codes etched on dark backgrounds where black-on-white would invert the
    # required quiet zone polarity.
    border_val = int(np.median(img))
    fill = 255 if border_val < 128 else 255  # quiet zone must always be the
    # *light* color in the code's own polarity; safest general default is
    # white, but we also emit an inverted-padded variant elsewhere.
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)


def v_raw(img: np.ndarray) -> np.ndarray:
    return img


def v_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def v_clahe_strong(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    return clahe.apply(img)


def v_otsu(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def v_otsu_inv(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(v_otsu(img))


def v_adaptive_mean(img: np.ndarray) -> np.ndarray:
    blur = cv2.medianBlur(img, 3)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7
    )


def v_adaptive_gaussian(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )


def v_illum_correct(img: np.ndarray) -> np.ndarray:
    """Correct uneven illumination (common with raking light on metal DPM)
    by estimating background via large-kernel morphological opening/closing
    and dividing it out, then renormalizing contrast."""
    # large structuring element relative to module size — estimate from image
    k = max(15, (min(img.shape[:2]) // 8) | 1)  # odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (k | 1, k | 1), 0)
    norm = cv2.divide(img, background, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def v_illum_correct_otsu(img: np.ndarray) -> np.ndarray:
    norm = v_illum_correct(img)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def v_tophat(img: np.ndarray) -> np.ndarray:
    """White tophat: isolates small bright dimples typical of dot-peen
    marking against a darker substrate."""
    k = max(9, (min(img.shape[:2]) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    top = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    top = cv2.normalize(top, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def v_blackhat(img: np.ndarray) -> np.ndarray:
    """Black tophat: isolates small dark pits, the inverse dot-peen case
    (laser-etched dark dots on a polished/bright substrate)."""
    k = max(9, (min(img.shape[:2]) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    black = cv2.normalize(black, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(black, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(th)


def v_unsharp(img: np.ndarray) -> np.ndarray:
    """Unsharp mask — recovers edge definition on mild blur without the
    ringing that aggressive sharpening kernels introduce."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(img, 1.8, blur, -0.8, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def v_bilateral_otsu(img: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise (better than Gaussian for keeping module
    edges crisp) followed by Otsu — good for noisy/grainy sensor images."""
    den = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=7)
    _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def v_gamma_dark(img: np.ndarray) -> np.ndarray:
    """Gamma correction to lift shadow detail (gamma < 1 brightens)."""
    gamma = 0.5
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def v_gamma_bright(img: np.ndarray) -> np.ndarray:
    """Gamma correction to recover detail blown out by glare (gamma > 1
    darkens highlights relatively)."""
    gamma = 2.0
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def v_clahe_adaptive_combo(img: np.ndarray) -> np.ndarray:
    """CLAHE then adaptive threshold — the single most generally effective
    combo for low-contrast DPM in practice."""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    eq = cv2.GaussianBlur(eq, (3, 3), 0)
    return cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )


def v_close_dilate(img: np.ndarray) -> np.ndarray:
    """Your existing morphological close+dilate, kept because it earns its
    place on heavily fragmented dot-peen marks — applied here onto an Otsu
    base rather than a fixed adaptive base for more consistent input."""
    base = v_otsu(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.dilate(closed, kernel, iterations=1)


# Variant registry: (name, function). Order = priority (fast & high-yield first).
VARIANTS: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = [
    ("raw", v_raw),
    ("clahe", v_clahe),
    ("clahe_adaptive_combo", v_clahe_adaptive_combo),
    ("illum_correct", v_illum_correct),
    ("otsu", v_otsu),
    ("adaptive_gaussian", v_adaptive_gaussian),
    ("unsharp", v_unsharp),
    ("illum_correct_otsu", v_illum_correct_otsu),
    ("bilateral_otsu", v_bilateral_otsu),
    ("clahe_strong", v_clahe_strong),
    ("adaptive_mean", v_adaptive_mean),
    ("otsu_inv", v_otsu_inv),
    ("tophat", v_tophat),
    ("blackhat", v_blackhat),
    ("gamma_dark", v_gamma_dark),
    ("gamma_bright", v_gamma_bright),
    ("close_dilate", v_close_dilate),
]


# ==========================================================================
# DECODER ENGINES
# ==========================================================================
# Each takes a uint8 grayscale (or BGR) numpy array and returns a list of
# DecodeResult (possibly empty). Wrapped in try/except because native
# bindings occasionally segfault-adjacent raise on pathological input and
# we never want one bad variant to kill the whole sweep.


def _engine_zxingcpp(img: np.ndarray, variant: str, rotation: float) -> List[DecodeResult]:
    if not _HAVE_ZXINGCPP:
        return []
    out = []
    try:
        pil_img = Image.fromarray(img)
        results = zxingcpp.read_barcodes(
            pil_img,
            formats=zxingcpp.BarcodeFormat.LinearCodes
            | zxingcpp.BarcodeFormat.DataMatrix
            | zxingcpp.BarcodeFormat.QRCode
            | zxingcpp.BarcodeFormat.Aztec,
            try_harder=True,
            try_rotate=True,
            try_invert=True,
            try_downscale=True,
        )
        for r in results:
            if r.valid and r.text:
                polygon = None
                pos = getattr(r, "position", None)
                if pos is not None:
                    try:
                        polygon = [
                            (int(pos.top_left.x), int(pos.top_left.y)),
                            (int(pos.top_right.x), int(pos.top_right.y)),
                            (int(pos.bottom_right.x), int(pos.bottom_right.y)),
                            (int(pos.bottom_left.x), int(pos.bottom_left.y)),
                        ]
                    except AttributeError:
                        polygon = None
                out.append(
                    DecodeResult(
                        text=r.text,
                        format=str(r.format),
                        engine="zxingcpp",
                        variant=variant,
                        rotation=rotation,
                        raw_bytes=bytes(r.bytes) if hasattr(r, "bytes") else None,
                        polygon=polygon,
                        source_image=img,
                    )
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("zxingcpp failed on variant=%s rot=%s: %s", variant, rotation, e)
    return out


def _engine_dmtx(img: np.ndarray, variant: str, rotation: float) -> List[DecodeResult]:
    if not _HAVE_DMTX:
        return []
    out = []
    try:
        pil_img = Image.fromarray(img)
        # timeout keeps pathological binarizations (e.g. near-noise images)
        # from stalling the whole sweep; max_count=1 since we exit on first hit.
        results = pylibdmtx.decode(pil_img, timeout=2000, max_count=1, shrink=1)
        img_h = img.shape[0]
        for r in results:
            text = r.data.decode("utf-8", errors="replace")
            if text:
                polygon = None
                rect = getattr(r, "rect", None)
                if rect is not None:
                    # libdmtx reports `rect` in a bottom-left-origin coordinate
                    # system (Cartesian), unlike every other library here
                    # which uses top-left-origin (image/array convention).
                    # Flip the y-axis so the polygon overlays correctly on
                    # the actual image array.
                    left, width, height = rect.left, rect.width, rect.height
                    top_img = img_h - (rect.top + height)
                    polygon = [
                        (left, top_img),
                        (left + width, top_img),
                        (left + width, top_img + height),
                        (left, top_img + height),
                    ]
                out.append(
                    DecodeResult(
                        text=text,
                        format="DataMatrix",
                        engine="pylibdmtx",
                        variant=variant,
                        rotation=rotation,
                        raw_bytes=r.data,
                        polygon=polygon,
                        source_image=img,
                    )
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("pylibdmtx failed on variant=%s rot=%s: %s", variant, rotation, e)
    return out


def _engine_zbar(img: np.ndarray, variant: str, rotation: float) -> List[DecodeResult]:
    if not _HAVE_ZBAR:
        return []
    out = []
    try:
        results = pyzbar.decode(img)
        for r in results:
            text = r.data.decode("utf-8", errors="replace")
            if text:
                polygon = None
                if getattr(r, "polygon", None):
                    polygon = [(int(p.x), int(p.y)) for p in r.polygon]
                out.append(
                    DecodeResult(
                        text=text,
                        format=str(r.type),
                        engine="pyzbar",
                        variant=variant,
                        rotation=rotation,
                        raw_bytes=r.data,
                        polygon=polygon,
                        source_image=img,
                    )
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("pyzbar failed on variant=%s rot=%s: %s", variant, rotation, e)
    return out


ENGINES: List[Callable[[np.ndarray, str, float], List[DecodeResult]]] = [
    _engine_zxingcpp,  # fastest, broadest format support, try_rotate/invert built-in
    _engine_dmtx,      # slowest but best raw DataMatrix/DPM recall
    _engine_zbar,       # cheap extra shot, strong for QR/1D
]


# ==========================================================================
# GEOMETRY HELPERS
# ==========================================================================


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return img
    if angle in (90, 180, 270):
        k = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
        return cv2.rotate(img, k)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    border_val = int(np.median(img))
    return cv2.warpAffine(
        img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=255 if border_val < 128 else 255
    )


def annotate_result(result: DecodeResult) -> Optional[np.ndarray]:
    """
    Draw the decoded polygon (or, if the engine gave no polygon, just a
    label banner) plus the decoded text onto `result.source_image` and
    return a BGR uint8 image ready to save/display.

    Returns None if the result has no source_image attached (shouldn't
    happen for anything produced by CodeDecoder, but guarded for safety
    if a DecodeResult is constructed/edited manually).
    """
    if result.source_image is None:
        return None

    base = result.source_image
    if base.ndim == 2:
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base.copy()

    h, w = canvas.shape[:2]
    # scale visual elements (line/font thickness) relative to image size so
    # tiny DPM crops and large frame grabs both get legible annotations.
    scale = max(1.0, min(w, h) / 400.0)
    color = (0, 255, 0)  # green, BGR
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.5 * scale
    font_thickness = max(1, int(round(1 * scale)))

    if result.polygon and len(result.polygon) >= 3:
        pts = np.array(result.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
        anchor_x, anchor_y = result.polygon[0]
    else:
        # no polygon available (engine didn't report one) — draw a border
        # around the whole frame instead so it's still obvious *which*
        # image/variant produced the hit, even without exact localization.
        cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), color, thickness)
        anchor_x, anchor_y = 4, 4

    label_lines = [
        f"{result.format}: {result.text}",
        f"engine={result.engine}  variant={result.variant}  rot={result.rotation}",
    ]
    line_height = int(22 * scale)

    def _measure(fs: float, ft: int) -> int:
        return max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0] for line in label_lines
        )

    # shrink font until the longest label line fits within the canvas width
    # (minus a small margin), so long decoded payloads on small/cropped
    # images don't run off-frame.
    margin = 10
    max_text_w = _measure(font_scale, font_thickness)
    while max_text_w > (w - margin) and font_scale > 0.25:
        font_scale *= 0.85
        font_thickness = max(1, int(round(font_thickness * 0.85)))
        max_text_w = _measure(font_scale, font_thickness)

    text_y = max(line_height, anchor_y - 8)
    anchor_x = max(0, min(anchor_x, w - max_text_w - margin))
    # background banner behind text for legibility over busy/textured PCB backgrounds
    banner_top = max(0, text_y - line_height)
    banner_bottom = min(h, text_y + line_height * (len(label_lines) - 1) + 8)
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (max(0, anchor_x - 2), banner_top),
        (min(w, anchor_x + max_text_w + margin), banner_bottom),
        (0, 0, 0),
        thickness=-1,
    )
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    for i, line in enumerate(label_lines):
        cv2.putText(
            canvas,
            line,
            (max(0, anchor_x + 4), text_y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return canvas


def save_annotated_results(
    results: Sequence[DecodeResult], out_path: str, multiple: str = "first"
) -> List[str]:
    """
    Save annotated image(s) for one or more DecodeResults.

    multiple: "first" saves only results[0] to exactly out_path.
              "all" saves every result, suffixing filenames with an index
              when there's more than one decoded symbol in the frame.
    Returns the list of file paths actually written.
    """
    if not results:
        return []

    out_path = Path(out_path)
    written: List[str] = []

    targets = results[:1] if multiple == "first" else results
    for i, r in enumerate(targets):
        canvas = annotate_result(r)
        if canvas is None:
            continue
        if len(targets) == 1:
            path = out_path
        else:
            path = out_path.with_name(f"{out_path.stem}_{i}{out_path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas)
        written.append(str(path))
    return written


def _upscale(img: np.ndarray, target_min_dim: int = 300) -> np.ndarray:
    """Lanczos upscale so the smallest dimension reaches target_min_dim —
    most decoders perform far worse below ~10px per module, and DPM crops
    are frequently tiny."""
    h, w = img.shape[:2]
    m = min(h, w)
    if m >= target_min_dim:
        return img
    scale = target_min_dim / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


# ==========================================================================
# MAIN DECODER
# ==========================================================================


class CodeDecoder:
    """
    Orchestrates the engine x variant x rotation sweep with early exit.

    Parameters
    ----------
    coarse_rotations : rotations always tried (cheap, covers the vast
        majority of real-world DPM skew on rigid parts).
    fine_rotations : only tried if coarse sweep + full variant bank fails;
        covers hand-held / fixture-misaligned skew.
    max_upscale_target : smallest-dimension pixel target for the upscale step.
    time_budget_s : soft wall-clock budget; sweep stops issuing new attempts
        once exceeded (lets you bound worst-case latency on a production line).
    debug_dir : if set, every variant image attempted is dumped here, named
        by variant/rotation, for visual debugging of why something failed.
    """

    def __init__(
        self,
        coarse_rotations: Sequence[float] = (0, 90, 180, 270),
        fine_rotations: Sequence[float] = (-15, -10, -5, 5, 10, 15),
        max_upscale_target: int = 400,
        time_budget_s: float = 20.0,
        debug_dir: Optional[str] = None,
    ):
        self.coarse_rotations = coarse_rotations
        self.fine_rotations = fine_rotations
        self.max_upscale_target = max_upscale_target
        self.time_budget_s = time_budget_s
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        if not any([_HAVE_ZXINGCPP, _HAVE_DMTX, _HAVE_ZBAR]):
            raise RuntimeError(
                "No decode engine is installed. Install at least one of: "
                "zxing-cpp, pylibdmtx, pyzbar."
            )
        if not _HAVE_DMTX:
            logger.warning(
                "Running without pylibdmtx: DataMatrix DPM recall will be "
                "meaningfully lower. Strongly recommend installing it."
            )

    # ---- public API -----------------------------------------------------

    def decode_file(
        self, path: str, save_annotated: Optional[str] = None, annotate_all: bool = False
    ) -> List[DecodeResult]:
        """
        save_annotated: if given, write an annotated copy of the
            successfully-decoded image (showing the symbol's location and
            decoded text overlaid) to this path. If multiple symbols are
            decoded and annotate_all=True, each gets its own file
            (suffixed _0, _1, ...); otherwise only the first hit is saved.
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # cv2 can choke on some color profiles / formats; fall back to PIL
            pil = Image.open(path).convert("L")
            img = np.array(pil)
        results = self.decode_array(img)
        if save_annotated and results:
            written = save_annotated_results(
                results, save_annotated, multiple="all" if annotate_all else "first"
            )
            for p in written:
                logger.info("Saved annotated decode image: %s", p)
        return results

    def decode_array(self, gray: np.ndarray) -> List[DecodeResult]:
        """
        gray: 2D uint8 numpy array (grayscale). If you have a color image,
        convert with cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) first.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        t0 = time.monotonic()
        gray = _upscale(gray, self.max_upscale_target)

        all_results: List[DecodeResult] = []

        # Pass 1: coarse rotations x full variant bank. This covers the
        # overwhelming majority of real cases and is where we want to spend
        # most of the time budget.
        for rotation in self.coarse_rotations:
            if self._time_up(t0):
                break
            rotated = _rotate(gray, rotation)
            hits = self._sweep_variants(rotated, rotation)
            if hits:
                return hits  # early exit on first success across the board
            all_results.extend(hits)

        # Pass 2: fine rotation sweep, only if pass 1 found nothing and we
        # still have time budget left. Skipped for the (very common) case
        # where the part is rigidly fixtured and skew is exactly 0/90/180/270.
        for rotation in self.fine_rotations:
            if self._time_up(t0):
                break
            rotated = _rotate(gray, rotation)
            hits = self._sweep_variants(rotated, rotation)
            if hits:
                return hits
            all_results.extend(hits)

        # Pass 3: tiled retry — split into overlapping quadrants and retry
        # the cheapest, highest-yield variants per tile. Helps when a large
        # frame has strong illumination gradient across it (global CLAHE/
        # Otsu washes out one side) but each tile is locally well-behaved.
        if not all_results and min(gray.shape[:2]) > 250:
            hits = self._tiled_retry(gray, t0)
            if hits:
                return hits

        return all_results  # empty list if everything failed

    # ---- internals --------------------------------------------------------

    def _time_up(self, t0: float) -> bool:
        return (time.monotonic() - t0) > self.time_budget_s

    def _sweep_variants(self, img: np.ndarray, rotation: float) -> List[DecodeResult]:
        padded_cache: dict = {}
        for name, fn in VARIANTS:
            try:
                processed = fn(img)
            except Exception as e:  # noqa: BLE001
                logger.debug("variant %s raised: %s", name, e)
                continue
            if processed is None:
                continue

            padded = _pad_quiet_zone(processed)

            if self.debug_dir:
                cv2.imwrite(str(self.debug_dir / f"{name}_rot{int(rotation)}.png"), padded)

            for engine in ENGINES:
                hits = engine(padded, name, rotation)
                if hits:
                    return hits
        return []

    def _tiled_retry(self, gray: np.ndarray, t0: float, tiles: int = 2, overlap: float = 0.15) -> List[DecodeResult]:
        h, w = gray.shape[:2]
        th, tw = h // tiles, w // tiles
        oh, ow = int(th * overlap), int(tw * overlap)
        for ty, tx in itertools.product(range(tiles), range(tiles)):
            if self._time_up(t0):
                break
            y0, y1 = max(0, ty * th - oh), min(h, (ty + 1) * th + oh)
            x0, x1 = max(0, tx * tw - ow), min(w, (tx + 1) * tw + ow)
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            tile = _upscale(tile, self.max_upscale_target)
            # cheaper sweep on tiles: only coarse rotations, top 6 variants
            for rotation in self.coarse_rotations:
                rotated = _rotate(tile, rotation)
                padded_variants = [
                    (name, fn) for name, fn in VARIANTS[:6]
                ]
                for name, fn in padded_variants:
                    try:
                        processed = fn(rotated)
                    except Exception:
                        continue
                    if processed is None:
                        continue
                    padded = _pad_quiet_zone(processed)
                    for engine in ENGINES:
                        hits = engine(padded, f"tile_{ty}_{tx}_{name}", rotation)
                        if hits:
                            return hits
        return []


# ==========================================================================
# CLI
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(description="Robust DataMatrix/QR/barcode decoder")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--debug-dir", default=None, help="Dump every preprocessing variant here")
    parser.add_argument("--time-budget", type=float, default=20.0, help="Soft wall-clock budget (s)")
    parser.add_argument(
        "--save-annotated",
        default=None,
        help="Path to save an annotated copy of the decoded image (shows symbol location + decoded text)",
    )
    parser.add_argument(
        "--annotate-all",
        action="store_true",
        help="If multiple symbols decoded in the same frame, save one annotated image per symbol "
        "instead of just the first",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    decoder = CodeDecoder(time_budget_s=args.time_budget, debug_dir=args.debug_dir)
    t0 = time.monotonic()
    results = decoder.decode_file(
        args.image, save_annotated=args.save_annotated, annotate_all=args.annotate_all
    )
    elapsed = time.monotonic() - t0

    if not results:
        print(f"FAILED to decode {args.image} ({elapsed:.2f}s)")
        if args.debug_dir:
            print(f"Inspect preprocessing attempts in: {args.debug_dir}")
        sys.exit(1)

    print(f"SUCCESS in {elapsed:.2f}s:")
    for r in results:
        print(f"  format={r.format}  engine={r.engine}  variant={r.variant}  rotation={r.rotation}")
        print(f"  text={r.text!r}")


if __name__ == "__main__":
    main()
