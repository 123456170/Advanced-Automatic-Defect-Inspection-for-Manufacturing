from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Advanced Production Line Defect Inspection",
    page_icon="🏭",
    layout="wide",
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FRAME_W, FRAME_H = 1280, 720
CROP_W, CROP_H = 220, 160

BELT_TOP, BELT_BOTTOM = 250, 520
INSPECT_LEFT, INSPECT_RIGHT = 470, 850

PRODUCT_PITCH_PX = 300

PRODUCT_TYPES = ("plate", "gear", "bracket")

KNOWN_DEFECTS = (
    "scratch",
    "crack",
    "missing",
    "dent",
    "contamination",
    "discoloration",
    "foreign_object",
)

UNKNOWN_DEFECTS = ("unknown",)
ALL_DEFECTS = KNOWN_DEFECTS + UNKNOWN_DEFECTS

DEFECT_LABELS = {
    "none": "Good",
    "scratch": "Scratch",
    "crack": "Crack",
    "missing": "Missing part",
    "dent": "Dent",
    "contamination": "Contamination",
    "discoloration": "Discoloration",
    "foreign_object": "Foreign object",
    "unknown": "Unknown defect",
    "unknown_anomaly": "Unknown anomaly",
    "review": "Review",
    "no_data": "No data",
}

PRODUCT_LABELS = {
    "plate": "Plate",
    "gear": "Gear",
    "bracket": "Bracket",
}


# ---------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------

def default_params() -> Dict[str, Any]:
    return {
        # Simulation
        "seed": 11,
        "duration": 8.0,
        "fps": 15,
        "line_speed": 180,

        # Product mix
        "mix_plate": 0.50,
        "mix_gear": 0.30,
        "mix_bracket": 0.20,

        # Defect rates
        "scratch_rate": 0.06,
        "crack_rate": 0.04,
        "missing_rate": 0.03,
        "dent_rate": 0.03,
        "contamination_rate": 0.04,
        "discoloration_rate": 0.03,
        "foreign_object_rate": 0.02,
        "unknown_rate": 0.03,
        "severity_bias": 0.55,

        # Environment
        "lighting_drift": 0.25,
        "dust": 0.15,
        "sensor_noise": 0.18,
        "motion_blur": 0.12,
        "compression": 0.10,
        "orientation_jitter_deg": 1.5,

        # Detector
        "sensitivity": 0.58,
        "reject_threshold": 0.22,
        "known_threshold": 0.28,
        "review_band": 0.08,
        "min_align_conf": 0.35,
        "w_known": 0.55,
        "w_anomaly": 0.45,
        "anomaly_top_k": 0.15,
        "temporal_ema": 0.45,
        "vote_ratio": 0.45,
        "bank_samples": 18,

        # Cost matrix
        "escape_cost": 250.0,
        "false_reject_cost": 8.0,
        "review_cost": 1.0,

        # Operations
        "latency_budget_ms": 80.0,
        "plc_mode": "Dry-run",
        "gallery_limit": 18,
    }


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def safe_mean(arr: List[float]) -> float:
    if not arr:
        return float("nan")
    return float(np.mean(np.asarray(arr, dtype=np.float32)))


def safe_percentile(arr: List[float], q: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float32), q))


def histogram_counts(values: List[float], bins: int = 14) -> List[int]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return [0] * bins

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if hi - lo < 1e-6:
        out = [0] * bins
        out[bins // 2] = int(arr.size)
        return out

    counts, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    return [int(x) for x in counts]


def population_stability_index(expected: List[float], actual: List[float], bins: int = 12) -> float:
    exp = np.asarray(expected, dtype=np.float32).ravel()
    act = np.asarray(actual, dtype=np.float32).ravel()

    if exp.size < 5 or act.size < 5:
        return float("nan")

    lo = float(min(exp.min(), act.min()))
    hi = float(max(exp.max(), act.max())) + 1e-3
    edges = np.linspace(lo, hi, bins + 1)

    eh, _ = np.histogram(exp, bins=edges)
    ah, _ = np.histogram(act, bins=edges)

    ep = (eh.astype(np.float32) + 1.0) / (float(eh.sum()) + float(bins))
    ap = (ah.astype(np.float32) + 1.0) / (float(ah.sum()) + float(bins))

    return float(np.sum((ap - ep) * np.log(ap / ep)))


def normalize_weights(weights: Dict[str, float], cap: float = 0.95) -> Tuple[Dict[str, float], float]:
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(w.values())

    if total <= 0.0:
        return {}, 1.0

    if total > cap:
        factor = cap / total
        w = {k: v * factor for k, v in w.items()}
        total = cap

    return w, max(0.0, 1.0 - total)


def choose_from_weights(rng: np.random.Generator, weights: Dict[str, float]) -> Optional[str]:
    if not weights:
        return None

    items = list(weights.items())
    total = sum(v for _, v in items)
    if total <= 0:
        return None

    r = float(rng.random()) * total
    cum = 0.0

    for k, v in items:
        cum += v
        if r <= cum:
            return k

    return items[-1][0]


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return int(x), int(y), int(w), int(h)


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def make_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------
# Golden product templates
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def make_golden_products() -> Dict[str, np.ndarray]:
    """
    Procedurally generated golden reference parts.

    In a real system these would be captured from known-good production parts
    under fixed optics and aligned to a canonical pose.
    """
    out: Dict[str, np.ndarray] = {}

    # Plate
    plate = np.full((CROP_H, CROP_W, 3), 142, dtype=np.uint8)
    for x in range(CROP_W):
        v = int(128 + 36 * (x / max(1, CROP_W - 1)))
        plate[:, x] = (v, max(0, v - 3), max(0, v - 7))

    cv2.rectangle(plate, (8, 8), (CROP_W - 9, CROP_H - 9), (92, 92, 92), 2, cv2.LINE_AA)
    cv2.circle(plate, (54, 72), 23, (54, 54, 54), -1, cv2.LINE_AA)
    cv2.circle(plate, (54, 72), 23, (122, 122, 122), 2, cv2.LINE_AA)
    cv2.rectangle(plate, (112, 38), (188, 70), (68, 68, 68), -1, cv2.LINE_AA)
    cv2.rectangle(plate, (112, 38), (188, 70), (126, 126, 126), 2, cv2.LINE_AA)

    for y in range(96, 142, 9):
        cv2.line(plate, (24, y), (196, y), (128, 128, 128), 1, cv2.LINE_AA)

    for pt in [(18, 18), (CROP_W - 19, 18), (18, CROP_H - 19), (CROP_W - 19, CROP_H - 19)]:
        cv2.circle(plate, pt, 5, (78, 78, 78), -1, cv2.LINE_AA)

    out["plate"] = plate

    # Gear
    gear = np.full((CROP_H, CROP_W, 3), 48, dtype=np.uint8)
    cx, cy = CROP_W // 2, CROP_H // 2
    outer_r = 62
    root_r = 48
    teeth = 14

    pts: List[Tuple[int, int]] = []
    for i in range(teeth * 2):
        ang = math.pi * i / teeth
        r = outer_r if i % 2 == 0 else root_r
        pts.append((int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))))

    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(gear, [pts_arr], (138, 138, 138))
    cv2.polylines(gear, [pts_arr], True, (190, 190, 190), 2, cv2.LINE_AA)

    cv2.circle(gear, (cx, cy), 22, (62, 62, 62), -1, cv2.LINE_AA)
    cv2.circle(gear, (cx, cy), 22, (170, 170, 170), 2, cv2.LINE_AA)

    for ang in range(0, 360, 60):
        a = math.radians(ang)
        x1 = int(cx + 24 * math.cos(a))
        y1 = int(cy + 24 * math.sin(a))
        x2 = int(cx + 42 * math.cos(a))
        y2 = int(cy + 42 * math.sin(a))
        cv2.line(gear, (x1, y1), (x2, y2), (105, 105, 105), 6, cv2.LINE_AA)

    out["gear"] = gear

    # Bracket
    bracket = np.full((CROP_H, CROP_W, 3), 58, dtype=np.uint8)

    base = np.array(
        [
            (18, 92),
            (198, 92),
            (198, 142),
            (18, 142),
        ],
        dtype=np.int32,
    )
    flange = np.array(
        [
            (22, 22),
            (72, 22),
            (72, 138),
            (22, 138),
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(bracket, [base], (142, 142, 142))
    cv2.fillPoly(bracket, [flange], (132, 132, 132))
    cv2.polylines(bracket, [base], True, (190, 190, 190), 2, cv2.LINE_AA)
    cv2.polylines(bracket, [flange], True, (190, 190, 190), 2, cv2.LINE_AA)

    cv2.circle(bracket, (46, 52), 10, (66, 66, 66), -1, cv2.LINE_AA)
    cv2.circle(bracket, (46, 108), 10, (66, 66, 66), -1, cv2.LINE_AA)
    cv2.circle(bracket, (150, 116), 12, (66, 66, 66), -1, cv2.LINE_AA)

    cv2.rectangle(bracket, (92, 104), (138, 128), (82, 82, 82), -1, cv2.LINE_AA)
    cv2.rectangle(bracket, (92, 104), (138, 128), (150, 150, 150), 2, cv2.LINE_AA)

    out["bracket"] = bracket

    return out


GOLDENS = make_golden_products()
GOLDEN_GRAYS = {
    k: cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    for k, v in GOLDENS.items()
}


# ---------------------------------------------------------------------
# Defect synthesis
# ---------------------------------------------------------------------

def synthesize_defect(
    base_img: np.ndarray,
    defect: str,
    severity: float,
    rng: np.random.Generator,
    product_type: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Returns:
        defective image, ground-truth binary mask, bbox.
    """
    img = base_img.copy()
    mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

    if defect == "none":
        return img, mask, None

    h, w = img.shape[:2]
    sev = clamp01(severity)

    if defect == "scratch":
        length = int((0.35 + 0.60 * sev) * max(h, w))
        angle = float(rng.uniform(-0.85, 0.85))
        cx, cy = w // 2, h // 2
        dx = 0.5 * length * math.cos(angle)
        dy = 0.5 * length * math.sin(angle)

        x1 = int(np.clip(cx - dx + rng.normal(0, 5), 3, w - 4))
        y1 = int(np.clip(cy - dy + rng.normal(0, 5), 3, h - 4))
        x2 = int(np.clip(cx + dx + rng.normal(0, 5), 3, w - 4))
        y2 = int(np.clip(cy + dy + rng.normal(0, 5), 3, h - 4))

        th = max(1, int(1 + 2.5 * sev))
        cv2.line(img, (x1, y1), (x2, y2), (246, 246, 246), th, cv2.LINE_AA)
        cv2.line(img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (78, 78, 78), max(1, th - 1), cv2.LINE_AA)
        cv2.line(mask, (x1, y1), (x2, y2), 255, th + 3, cv2.LINE_AA)

    elif defect == "crack":
        x = int(rng.integers(22, max(23, w - 44)))
        y = int(rng.integers(22, max(23, h - 44)))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        pts = [(x, y)]
        steps = int(4 + 8 * sev)

        for _ in range(steps):
            step = float(rng.uniform(9.0, 20.0)) * (0.65 + sev)
            angle += float(rng.normal(0.0, 0.60))
            x = int(np.clip(x + step * math.cos(angle), 4, w - 5))
            y = int(np.clip(y + step * math.sin(angle), 4, h - 5))
            pts.append((x, y))

        th = max(1, int(1 + 2.2 * sev))
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], False, (24, 24, 24), th, cv2.LINE_AA)
        cv2.polylines(mask, [arr], False, 255, th + 3, cv2.LINE_AA)

        if len(pts) > 3 and float(rng.random()) < 0.75:
            bx, by = pts[len(pts) // 2]
            ba = float(rng.uniform(0.0, 2.0 * math.pi))
            ex = int(np.clip(bx + 26.0 * math.cos(ba), 4, w - 5))
            ey = int(np.clip(by + 26.0 * math.sin(ba), 4, h - 5))
            cv2.line(img, (bx, by), (ex, ey), (34, 34, 34), max(1, th - 1), cv2.LINE_AA)
            cv2.line(mask, (bx, by), (ex, ey), 255, th + 2, cv2.LINE_AA)

    elif defect == "missing":
        dark = (30, 30, 30)
        edge = (96, 96, 96)
        choice = int(rng.integers(0, 3))

        if product_type == "gear" and choice == 0:
            # Missing tooth sector.
            cx, cy = w // 2, h // 2
            start = int(rng.integers(0, 360))
            extent = int(18 + 34 * sev)
            cv2.ellipse(img, (cx, cy), (64, 64), 0, start, start + extent, dark, -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (64, 64), 0, start, start + extent, edge, 2, cv2.LINE_AA)
            cv2.ellipse(mask, (cx, cy), (64, 64), 0, start, start + extent, 255, -1, cv2.LINE_AA)

        elif choice == 0:
            cx, cy = 54 if product_type == "plate" else w // 2, 72 if product_type == "plate" else h // 2
            rad = int(12 + 24 * sev)
            cv2.circle(img, (cx, cy), rad, dark, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), rad, edge, 2, cv2.LINE_AA)
            cv2.circle(mask, (cx, cy), rad, 255, -1, cv2.LINE_AA)

        elif choice == 1:
            x1, y1 = 112, 38
            x2 = int(112 + (188 - 112) * (0.35 + 0.65 * sev))
            y2 = int(38 + (70 - 38) * (0.35 + 0.65 * sev))
            cv2.rectangle(img, (x1, y1), (x2, y2), dark, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), edge, 2, cv2.LINE_AA)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1, cv2.LINE_AA)

        else:
            size = int(20 + 42 * sev)
            corner = int(rng.integers(0, 4))
            if corner == 0:
                pts = np.array([[0, 0], [size, 0], [0, size]], dtype=np.int32)
            elif corner == 1:
                pts = np.array([[w, 0], [w - size, 0], [w, size]], dtype=np.int32)
            elif corner == 2:
                pts = np.array([[0, h], [size, h], [0, h - size]], dtype=np.int32)
            else:
                pts = np.array([[w, h], [w - size, h], [w, h - size]], dtype=np.int32)

            cv2.fillPoly(img, [pts], dark)
            cv2.polylines(img, [pts], True, edge, 2, cv2.LINE_AA)
            cv2.fillPoly(mask, [pts], 255)

    elif defect == "dent":
        cx = int(rng.integers(45, max(46, w - 45)))
        cy = int(rng.integers(35, max(36, h - 35)))
        ax = int(14 + 28 * sev)
        ay = int(10 + 22 * sev)
        ang = float(rng.uniform(0.0, 180.0))

        cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, (72, 72, 72), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, (125, 125, 125), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx - ax // 4, cy - ay // 4), (max(3, ax // 3), max(2, ay // 3)), ang, 0, 360, (185, 185, 185), -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1, cv2.LINE_AA)

    elif defect == "contamination":
        cx = int(rng.integers(35, max(36, w - 35)))
        cy = int(rng.integers(30, max(31, h - 30)))
        npts = 10
        angles = np.linspace(0.0, 2.0 * math.pi, npts, endpoint=False) + rng.normal(0.0, 0.25, size=npts)
        radii = (9.0 + 24.0 * sev) * (0.55 + 0.90 * rng.random(size=npts))
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        pts = np.column_stack((xs, ys)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 3, w - 4)
        pts[:, 1] = np.clip(pts[:, 1], 3, h - 4)

        palette = [
            (54, 92, 42),
            (38, 56, 94),
            (28, 28, 28),
            (92, 64, 34),
        ]
        color = palette[int(rng.integers(0, len(palette)))]
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255)

    elif defect == "discoloration":
        cx = int(rng.integers(45, max(46, w - 45)))
        cy = int(rng.integers(35, max(36, h - 35)))
        ax = int(22 + 42 * sev)
        ay = int(16 + 32 * sev)
        ang = float(rng.uniform(0.0, 180.0))

        x1 = max(0, cx - ax)
        x2 = min(w, cx + ax)
        y1 = max(0, cy - ay)
        y2 = min(h, cy + ay)

        roi = img[y1:y2, x1:x2].astype(np.float32)
        shift = np.array(
            [
                float(rng.integers(-55, 55)),
                float(rng.integers(-45, 45)),
                float(rng.integers(-35, 35)),
            ],
            dtype=np.float32,
        )
        roi = np.clip(roi + shift, 0.0, 255.0).astype(np.uint8)
        img[y1:y2, x1:x2] = roi

        cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1, cv2.LINE_AA)

    elif defect == "foreign_object":
        cx = int(rng.integers(40, max(41, w - 40)))
        cy = int(rng.integers(35, max(36, h - 35)))
        size = int(8 + 18 * sev)
        kind = int(rng.integers(0, 3))

        if kind == 0:
            pts = np.array(
                [
                    (cx - size, cy),
                    (cx - size // 2, cy - size),
                    (cx + size // 2, cy - size),
                    (cx + size, cy),
                    (cx + size // 2, cy + size),
                    (cx - size // 2, cy + size),
                ],
                dtype=np.int32,
            )
            color = (210, 210, 210)
        elif kind == 1:
            pts = np.array(
                [
                    (cx - size, cy - size // 2),
                    (cx + size, cy - size // 2),
                    (cx + size, cy + size // 2),
                    (cx - size, cy + size // 2),
                ],
                dtype=np.int32,
            )
            color = (42, 42, 42)
        else:
            pts = np.array(
                [
                    (cx - size, cy + size),
                    (cx, cy - size),
                    (cx + size, cy + size),
                ],
                dtype=np.int32,
            )
            color = (64, 180, 220)

        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255)

    else:
        # Unknown / never-seen defect.
        cx = int(rng.integers(35, max(36, w - 35)))
        cy = int(rng.integers(30, max(31, h - 30)))
        npts = 11
        angles = np.linspace(0.0, 2.0 * math.pi, npts, endpoint=False) + rng.normal(0.0, 0.30, size=npts)
        radii = (11.0 + 30.0 * sev) * (0.50 + 1.00 * rng.random(size=npts))
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        pts = np.column_stack((xs, ys)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 3, w - 4)
        pts[:, 1] = np.clip(pts[:, 1], 3, h - 4)

        if float(rng.random()) < 0.5:
            shade = int(rng.integers(18, 68))
        else:
            shade = int(rng.integers(208, 252))

        color = (shade, shade, shade)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (118, 118, 118), 1, cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255)

    bbox = bbox_from_mask(mask)
    return img, mask, bbox


# ---------------------------------------------------------------------
# Product schedule
# ---------------------------------------------------------------------

def build_products(
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], float, int]:
    fps = max(1, int(params["fps"]))
    duration = max(1.0, float(params["duration"]))
    line_speed = max(1, int(params["line_speed"]))

    total_frames = int(duration * fps)
    spawn_interval = max(1, int(round(60.0 * fps / line_speed)))
    px_per_frame = float(PRODUCT_PITCH_PX) * line_speed / (60.0 * fps)
    travel_frames = int((FRAME_W + CROP_W) / max(0.1, px_per_frame)) + 3

    mix, _ = normalize_weights(
        {
            "plate": float(params["mix_plate"]),
            "gear": float(params["mix_gear"]),
            "bracket": float(params["mix_bracket"]),
        },
        cap=1.0,
    )

    defect_rates, good_p = normalize_weights(
        {
            "scratch": float(params["scratch_rate"]),
            "crack": float(params["crack_rate"]),
            "missing": float(params["missing_rate"]),
            "dent": float(params["dent_rate"]),
            "contamination": float(params["contamination_rate"]),
            "discoloration": float(params["discoloration_rate"]),
            "foreign_object": float(params["foreign_object_rate"]),
            "unknown": float(params["unknown_rate"]),
        },
        cap=0.95,
    )

    products: List[Dict[str, Any]] = []
    pid = 0

    for spawn_frame in range(0, total_frames + travel_frames, spawn_interval):
        product_type = choose_from_weights(rng, mix) or "plate"

        if float(rng.random()) < good_p:
            defect = "none"
        else:
            defect = choose_from_weights(rng, defect_rates) or "none"

        severity = float(np.clip(rng.normal(float(params["severity_bias"]), 0.18), 0.25, 1.0))
        if defect == "none":
            severity = 0.0

        angle = float(
            np.clip(
                rng.normal(0.0, max(0.0, float(params["orientation_jitter_deg"]))),
                -22.0,
                22.0,
            )
        )
        y_jitter = float(rng.normal(0.0, 2.2))

        base = GOLDENS[product_type]
        img, mask, bbox = synthesize_defect(base, defect, severity, rng, product_type)

        products.append(
            {
                "id": pid,
                "spawn_frame": spawn_frame,
                "product_type": product_type,
                "defect": defect,
                "severity": severity,
                "angle": angle,
                "y_jitter": y_jitter,
                "image": img,
                "mask": mask,
                "gt_bbox": bbox,
            }
        )
        pid += 1

    return products, px_per_frame, total_frames


# ---------------------------------------------------------------------
# Image transforms and camera effects
# ---------------------------------------------------------------------

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) < 0.05:
        return img.copy()

    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_camera_effects(
    img: np.ndarray,
    frame_idx: int,
    params: Dict[str, Any],
    rng: np.random.Generator,
    is_crop: bool = False,
) -> np.ndarray:
    out = img.astype(np.float32)

    # Lighting drift and exposure ripple.
    drift = float(params["lighting_drift"]) * (
        0.10 * np.sin(frame_idx / 23.0)
        + 0.045 * np.sin(frame_idx / 6.0 + 0.8)
    )
    out *= 1.0 + drift
    out += float(params["lighting_drift"]) * (8.0 if is_crop else 12.0) * np.sin(frame_idx / 37.0 + 1.1)

    # Mild white balance drift.
    wb = float(params["lighting_drift"]) * 5.0
    out[:, :, 0] += wb * np.sin(frame_idx / 41.0)
    out[:, :, 2] -= wb * np.sin(frame_idx / 29.0 + 0.5)

    # Sensor noise.
    if float(params["sensor_noise"]) > 0:
        out += rng.normal(0.0, float(params["sensor_noise"]) * (6.0 if is_crop else 5.0), out.shape)

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    # Dust / lens contamination.
    if float(params["dust"]) > 0:
        n = int(float(params["dust"]) * (6 if is_crop else 22))
        for _ in range(n):
            x = int(rng.integers(0, out.shape[1]))
            y = int(rng.integers(0, out.shape[0]))
            r = int(rng.integers(1, 4 if not is_crop else 3))
            val = 238 if float(rng.random()) < 0.55 else 24
            cv2.circle(out, (x, y), r, (val, val, val), -1, cv2.LINE_AA)

    # Motion blur.
    if float(params["motion_blur"]) > 0:
        k = int(1 + float(params["motion_blur"]) * (6 if is_crop else 9))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)

    # Compression artifacts.
    if float(params["compression"]) > 0.02:
        q = int(np.clip(94.0 - 58.0 * float(params["compression"]), 25.0, 96.0))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    return out


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_background(frame_idx: int, px_per_frame: float, params: Dict[str, Any]) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), 34, dtype=np.uint8)

    # Factory wall.
    cv2.rectangle(frame, (0, 0), (FRAME_W, 145), (24, 28, 34), -1)

    # Safety stripe.
    for x in range(0, FRAME_W, 42):
        cv2.line(frame, (x, 142), (x + 22, 154), (0, 210, 255), 8, cv2.LINE_AA)

    # Camera rig.
    cam_x = (INSPECT_LEFT + INSPECT_RIGHT) // 2
    cv2.rectangle(frame, (cam_x - 95, 35), (cam_x + 95, 112), (88, 88, 88), -1)
    cv2.rectangle(frame, (cam_x - 95, 35), (cam_x + 95, 112), (150, 150, 150), 2, cv2.LINE_AA)
    cv2.circle(frame, (cam_x, 88), 22, (28, 28, 28), -1, cv2.LINE_AA)
    cv2.circle(frame, (cam_x, 88), 22, (190, 190, 190), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "TELECENTRIC CAMERA",
        (cam_x - 88, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Light bars.
    for lx in [INSPECT_LEFT + 25, INSPECT_RIGHT - 85]:
        cv2.rectangle(frame, (lx, 120), (lx + 60, 134), (245, 245, 210), -1)
        glow = 160 + int(50 * math.sin(frame_idx / 5.0))
        cv2.rectangle(frame, (lx, 134), (lx + 60, 138), (glow, glow, 120), -1)

    # Conveyor belt.
    cv2.rectangle(frame, (0, BELT_TOP), (FRAME_W, BELT_BOTTOM), (72, 72, 72), -1)

    offset = int((frame_idx * max(0.1, px_per_frame) * 0.38) % 90.0)
    for x in range(-110, FRAME_W + 110, 90):
        cv2.line(
            frame,
            (x + offset, BELT_TOP),
            (x + offset - 34, BELT_BOTTOM),
            (94, 94, 94),
            2,
            cv2.LINE_AA,
        )

    # Rollers.
    for x in range(35, FRAME_W, 100):
        cv2.circle(frame, (x, BELT_BOTTOM + 20), 11, (54, 54, 54), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, BELT_BOTTOM + 20), 11, (116, 116, 116), 2, cv2.LINE_AA)

    # Rails.
    cv2.rectangle(frame, (0, BELT_TOP - 9), (FRAME_W, BELT_TOP - 2), (126, 126, 126), -1)
    cv2.rectangle(frame, (0, BELT_BOTTOM + 2), (FRAME_W, BELT_BOTTOM + 9), (126, 126, 126), -1)

    # Inspection gantry.
    cv2.rectangle(frame, (INSPECT_LEFT, 145), (INSPECT_LEFT + 10, BELT_TOP - 9), (108, 108, 108), -1)
    cv2.rectangle(frame, (INSPECT_RIGHT - 10, 145), (INSPECT_RIGHT, BELT_TOP - 9), (108, 108, 108), -1)
    cv2.rectangle(frame, (INSPECT_LEFT, BELT_BOTTOM + 9), (INSPECT_LEFT + 10, 595), (108, 108, 108), -1)
    cv2.rectangle(frame, (INSPECT_RIGHT - 10, BELT_BOTTOM + 9), (INSPECT_RIGHT, 595), (108, 108, 108), -1)

    cv2.rectangle(
        frame,
        (INSPECT_LEFT, BELT_TOP - 28),
        (INSPECT_RIGHT, BELT_BOTTOM + 28),
        (255, 190, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "INSPECTION ZONE",
        (INSPECT_LEFT + 6, BELT_TOP - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 190, 0),
        1,
        cv2.LINE_AA,
    )

    # Scan line.
    scan_y = BELT_TOP + int((frame_idx * 11) % max(1, BELT_BOTTOM - BELT_TOP))
    cv2.line(
        frame,
        (INSPECT_LEFT + 3, scan_y),
        (INSPECT_RIGHT - 3, scan_y),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # PLC cabinet.
    cv2.rectangle(frame, (35, 560), (215, 690), (48, 52, 58), -1)
    cv2.rectangle(frame, (35, 560), (215, 690), (130, 130, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, "PLC / OPC-UA", (52, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    led_colors = [
        (0, 220, 0) if frame_idx % 30 < 20 else (0, 90, 0),
        (0, 180, 255) if frame_idx % 18 < 9 else (0, 70, 110),
        (255, 60, 60) if frame_idx % 50 < 8 else (80, 20, 20),
    ]
    for i, col in enumerate(led_colors):
        cv2.circle(frame, (70 + i * 38, 625), 8, col, -1, cv2.LINE_AA)

    # Bins.
    cv2.rectangle(frame, (260, 590), (470, 690), (35, 82, 35), -1)
    cv2.rectangle(frame, (260, 590), (470, 690), (80, 190, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "OK BIN", (325, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 255, 220), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (FRAME_W - 470, 590), (FRAME_W - 260, 690), (92, 35, 35), -1)
    cv2.rectangle(frame, (FRAME_W - 470, 590), (FRAME_W - 260, 690), (190, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "NG / REVIEW", (FRAME_W - 445, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 220), 2, cv2.LINE_AA)

    # HMI mini screen.
    cv2.rectangle(frame, (FRAME_W - 220, 35), (FRAME_W - 35, 135), (18, 18, 18), -1)
    cv2.rectangle(frame, (FRAME_W - 220, 35), (FRAME_W - 35, 135), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, "LINE HMI", (FRAME_W - 205, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(params['line_speed'])} PPM", (FRAME_W - 205, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, params["plc_mode"].upper(), (FRAME_W - 205, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    return frame


def paste_product(
    frame: np.ndarray,
    img: np.ndarray,
    center_x: int,
    center_y: int,
    angle_deg: float,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = img.shape[:2]

    # Simple contact shadow.
    cv2.ellipse(
        frame,
        (center_x, center_y + int(h * 0.44)),
        (int(w * 0.42), 12),
        0,
        0,
        360,
        (18, 18, 18),
        -1,
        cv2.LINE_AA,
    )

    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])

    n_w = int(h * sin + w * cos)
    n_h = int(h * cos + w * sin)

    m[0, 2] += n_w / 2.0 - w / 2.0
    m[1, 2] += n_h / 2.0 - h / 2.0

    rot = cv2.warpAffine(
        img,
        m,
        (n_w, n_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    x = int(center_x - n_w / 2.0)
    y = int(center_y - n_h / 2.0)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(FRAME_W, x + n_w)
    y2 = min(FRAME_H, y + n_h)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    rot_crop = rot[y1 - y:y2 - y, x1 - x:x2 - x]

    mask = cv2.cvtColor(rot_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    mask3 = cv2.merge((mask, mask, mask))

    np.copyto(roi, rot_crop, where=(mask3 > 0))

    return x1, y1, x2, y2


def draw_pip(
    frame: np.ndarray,
    view: Dict[str, Any],
    insp: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    px, py = 900, 155
    pw, ph = 355, 350

    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (18, 20, 24), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"CLOSE-UP #{view['id']} {PRODUCT_LABELS.get(view['product_type'], view['product_type'])}",
        (px + 8, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    crop_small = cv2.resize(insp["crop"], (160, 116), interpolation=cv2.INTER_AREA)
    frame[py + 35:py + 151, px + 12:px + 172] = crop_small

    heat_small = cv2.resize(insp["anomaly_heat_bgr"], (160, 116), interpolation=cv2.INTER_AREA)
    frame[py + 35:py + 151, px + 182:px + 342] = heat_small

    cv2.putText(frame, "Crop", (px + 12, py + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "Anomaly heat", (px + 182, py + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

    # Score bars.
    bars = [
        ("Fused", float(insp["fused"]), (0, 190, 255)),
        ("Known", float(insp["known_max"]), (0, 255, 160)),
        ("Anomaly", float(insp["anomaly"]), (255, 120, 80)),
        ("Align", float(insp["align_conf"]), (180, 180, 180)),
    ]

    y0 = py + 180
    for i, (label, val, col) in enumerate(bars):
        yy = y0 + i * 28
        cv2.putText(frame, label, (px + 12, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (px + 78, yy), (px + 330, yy + 16), (45, 45, 45), -1)
        bw = int((px + 330 - px - 78) * clamp01(val))
        cv2.rectangle(frame, (px + 78, yy), (px + 78 + bw, yy + 16), col, -1)
        cv2.putText(frame, f"{val:.2f}", (px + 335, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)

    pred = str(insp["frame_pred"])
    pred_label = DEFECT_LABELS.get(pred, pred)
    color = (0, 0, 255) if pred in ALL_DEFECTS or pred == "unknown_anomaly" else (0, 210, 0)
    if pred == "review":
        color = (0, 140, 255)

    cv2.putText(
        frame,
        f"Pred: {pred_label}",
        (px + 12, py + 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Uncertainty: {float(insp['uncertainty']):.2f}",
        (px + 12, py + 338),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame: np.ndarray,
    views: List[Dict[str, Any]],
    inspections: List[Optional[Dict[str, Any]]],
    params: Dict[str, Any],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    out = frame.copy()

    current_ok = 0
    current_review = 0
    current_reject = 0

    for v, insp in zip(views, inspections):
        bbox = v.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        if insp is None:
            cv2.rectangle(out, (x1, y1), (x2, y2), (145, 145, 145), 1, cv2.LINE_AA)
            cv2.putText(
                out,
                f"#{v['id']} {PRODUCT_LABELS.get(v['product_type'], v['product_type'])}",
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            continue

        fused = float(insp["fused"])
        known = float(insp["known_max"])
        pred = str(insp["frame_pred"])

        if fused >= float(params["reject_threshold"]) or known >= float(params["known_threshold"]):
            color = (0, 0, 255)
            label = "REJECT"
            current_reject += 1
        elif fused >= float(params["reject_threshold"]) - float(params["review_band"]):
            color = (0, 140, 255)
            label = "REVIEW"
            current_review += 1
        else:
            color = (0, 215, 0)
            label = "OK"
            current_ok += 1

        thickness = 3 if bool(v["in_zone"]) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        text = f"#{v['id']} {label} f={fused:.2f} k={known:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        th = 1
        (tw, th_px), baseline = cv2.getTextSize(text, font, scale, th)
        ty = max(th_px + 5, y1 - 4)

        cv2.rectangle(out, (x1, ty - th_px - 4), (x1 + tw + 4, ty + baseline + 1), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - 2), font, scale, (255, 255, 255), th, cv2.LINE_AA)

    # Top HUD.
    cv2.rectangle(out, (0, 0), (FRAME_W, 48), (16, 16, 16), -1)
    hud = (
        f"ADVANCED HYBRID INSPECTION | frame {frame_idx + 1}/{total_frames} | "
        f"{int(params['line_speed'])} PPM | sens {float(params['sensitivity']):.2f} | "
        f"reject>{float(params['reject_threshold']):.2f} | known>{float(params['known_threshold']):.2f}"
    )
    cv2.putText(out, hud, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)

    # PLC banner.
    any_reject = current_reject > 0
    plc_color = (0, 0, 255) if any_reject else (0, 210, 0)
    plc_text = f"{params['plc_mode'].upper()}: REJECT SIGNAL" if any_reject else f"{params['plc_mode'].upper()}: PASS"
    cv2.rectangle(out, (FRAME_W - 285, 54), (FRAME_W - 10, 84), (20, 20, 20), -1)
    cv2.putText(out, plc_text, (FRAME_W - 278, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, plc_color, 1, cv2.LINE_AA)

    # Bottom ticker.
    cv2.rectangle(out, (0, FRAME_H - 34), (FRAME_W, FRAME_H), (16, 16, 16), -1)
    ticker = (
        f"Current frame: OK={current_ok}  REVIEW={current_review}  REJECT={current_reject}  |  "
        f"Dust={float(params['dust']):.2f}  Noise={float(params['sensor_noise']):.2f}  "
        f"Blur={float(params['motion_blur']):.2f}  Compression={float(params['compression']):.2f}"
    )
    cv2.putText(out, ticker, (8, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def render_frame(
    frame_idx: int,
    products: List[Dict[str, Any]],
    px_per_frame: float,
    params: Dict[str, Any],
    banks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
    frame = render_background(frame_idx, px_per_frame, params)
    views: List[Dict[str, Any]] = []
    inspections: List[Optional[Dict[str, Any]]] = []

    frame_seed = (int(params["seed"]) + frame_idx * 1009) % (2 ** 32)
    frame_rng = np.random.default_rng(frame_seed)

    center_y_base = int((BELT_TOP + BELT_BOTTOM) / 2)
    pip_target: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None

    for p in products:
        x = -float(CROP_W) + (frame_idx - int(p["spawn_frame"])) * px_per_frame
        center_x = int(x + CROP_W / 2.0)

        if center_x < -CROP_W - 30 or center_x > FRAME_W + CROP_W + 30:
            continue

        center_y = int(center_y_base + float(p.get("y_jitter", 0.0)))
        bbox = paste_product(frame, p["image"], center_x, center_y, float(p["angle"]))

        residual_angle = float(p["angle"]) * 0.32
        crop_base = rotate_image(p["image"], residual_angle)

        crop_seed = (int(params["seed"]) + frame_idx * 1009 + int(p["id"]) * 131) % (2 ** 32)
        crop_rng = np.random.default_rng(crop_seed)
        degraded = apply_camera_effects(crop_base, frame_idx, params, crop_rng, is_crop=True)

        in_zone = INSPECT_LEFT <= center_x <= INSPECT_RIGHT

        insp: Optional[Dict[str, Any]] = None
        if in_zone:
            insp = inspect_product(
                degraded,
                str(p["product_type"]),
                params,
                banks.get(str(p["product_type"])),
            )
            if pip_target is None:
                pip_target = (
                    {
                        "id": int(p["id"]),
                        "product_type": str(p["product_type"]),
                    },
                    insp,
                )

        view = {
            "id": int(p["id"]),
            "product_type": str(p["product_type"]),
            "bbox": bbox,
            "crop_degraded": degraded,
            "gt_mask": p["mask"],
            "defect": str(p["defect"]),
            "severity": float(p["severity"]),
            "center_x": center_x,
            "center_y": center_y,
            "in_zone": bool(in_zone),
        }

        views.append(view)
        inspections.append(insp)

    # Apply global camera effects to the physical scene before crisp UI overlays.
    frame = apply_camera_effects(frame, frame_idx, params, frame_rng, is_crop=False)

    if pip_target is not None:
        draw_pip(frame, pip_target[0], pip_target[1], params)

    frame = annotate_frame(frame, views, inspections, params, frame_idx, max(1, int(params["duration"] * params["fps"])))
    return frame, views, inspections


# ---------------------------------------------------------------------
# Feature bank and inspection model
# ---------------------------------------------------------------------

TILE_H, TILE_W = 32, 32


def make_tile_descriptors(img: np.ndarray) -> np.ndarray:
    crop = cv2.resize(img, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)

    descs: List[np.ndarray] = []

    for y in range(0, CROP_H, TILE_H):
        for x in range(0, CROP_W, TILE_W):
            y2 = min(CROP_H, y + TILE_H)
            x2 = min(CROP_W, x + TILE_W)

            patch = gray[y:y2, x:x2]
            pmag = mag[y:y2, x:x2]

            p = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).ravel()
            m = cv2.resize(pmag, (8, 8), interpolation=cv2.INTER_AREA).ravel()

            extra = np.array([float(p.mean()), float(p.std()), float(m.mean())], dtype=np.float32)
            d = np.concatenate((p, m, extra)).astype(np.float32)
            n = float(np.linalg.norm(d)) + 1e-6
            descs.append(d / n)

    return np.vstack(descs).astype(np.float32)


def build_good_bank(
    product_type: str,
    params: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    golden = GOLDENS[product_type]
    feats: List[np.ndarray] = []

    for i in range(max(1, int(count))):
        seed = (int(params["seed"]) * 100003 + i * 9176 + hash(product_type)) % (2 ** 32)
        rng_i = np.random.default_rng(seed)

        angle = float(
            rng_i.normal(
                0.0,
                max(0.05, float(params["orientation_jitter_deg"]) * 0.18),
            )
        )
        base = rotate_image(golden, angle)
        deg = apply_camera_effects(base, i * 7, params, rng_i, is_crop=True)
        feats.append(make_tile_descriptors(deg))

    return np.vstack(feats).astype(np.float32)


def inspect_product(
    crop_bgr: np.ndarray,
    product_type: str,
    params: Dict[str, Any],
    bank: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    crop = cv2.resize(crop_bgr, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    golden_gray = GOLDEN_GRAYS[product_type]

    # Alignment confidence.
    a = gray.astype(np.float32).ravel()
    b = golden_gray.astype(np.float32).ravel()
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    ncc = float(np.dot(a, b) / denom)
    align_conf = clamp01((ncc + 1.0) / 2.0)

    # Difference map.
    diff = cv2.absdiff(gray, golden_gray).astype(np.float32)
    floor = float(np.percentile(diff, 5))
    diff = np.clip(diff - floor, 0.0, 255.0)
    blur = cv2.GaussianBlur(diff, (3, 3), 0)

    thresh = float(
        np.clip(
            42.0 - 26.0 * float(params["sensitivity"]) + 14.0 * (1.0 - align_conf),
            8.0,
            95.0,
        )
    )

    mask = (blur > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signed = gray.astype(np.float32) - golden_gray.astype(np.float32)
    edge = cv2.Canny(gray, 70, 160)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    class_scores = {k: 0.0 for k in KNOWN_DEFECTS}
    bboxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 14.0:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(ww), int(hh)))

        elong = float(max(ww, hh) / max(1, min(ww, hh)))
        extent = area / float(max(1, ww * hh))

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(1.0, hull_area)

        peri = float(cv2.arcLength(cnt, True))
        circularity = float(4.0 * math.pi * area / max(1e-6, peri * peri))

        cnt_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        pixels = cnt_mask > 0

        if not bool(np.any(pixels)):
            continue

        mean_signed = float(signed[pixels].mean())
        mean_sat = float(sat[pixels].mean())
        edge_density = float(edge[pixels].mean() / 255.0)

        bright = clamp01(mean_signed / 34.0)
        dark = clamp01(-mean_signed / 34.0)
        colorfulness = clamp01((mean_sat - 55.0) / 125.0)

        thin = clamp01((elong - 2.0) / 5.0)
        linear_area = clamp01(area / 240.0)
        medium_area = clamp01(area / 260.0) * (1.0 - clamp01(area / 950.0))
        large_area = clamp01(area / 950.0)
        compact = clamp01((1.18 - extent) * 2.8)
        jagged = clamp01((1.08 - solidity) * 4.0)
        smooth = clamp01(1.0 - edge_density)

        class_scores["scratch"] = max(
            class_scores["scratch"],
            clamp01(thin * bright * linear_area * (0.55 + 0.45 * edge_density)),
        )
        class_scores["crack"] = max(
            class_scores["crack"],
            clamp01(thin * dark * jagged * linear_area * (0.50 + 0.50 * edge_density)),
        )
        class_scores["missing"] = max(
            class_scores["missing"],
            clamp01(large_area * dark * compact * (0.45 + 0.55 * smooth)),
        )
        class_scores["dent"] = max(
            class_scores["dent"],
            clamp01(medium_area * dark * solidity * circularity * smooth),
        )
        class_scores["contamination"] = max(
            class_scores["contamination"],
            clamp01(medium_area * colorfulness * jagged * (0.45 + 0.55 * edge_density)),
        )
        class_scores["discoloration"] = max(
            class_scores["discoloration"],
            clamp01(large_area * colorfulness * smooth * (1.0 - dark * 0.65)),
        )
        class_scores["foreign_object"] = max(
            class_scores["foreign_object"],
            clamp01(medium_area * (bright + dark) * edge_density * compact),
        )

    # Tile anomaly.
    anomaly_map = np.zeros((CROP_H, CROP_W), dtype=np.float32)
    anomaly_bank = 0.0

    desc = make_tile_descriptors(crop)
    n_tiles = desc.shape[0]

    if bank is not None and len(bank) > 0 and n_tiles > 0:
        dots = bank.dot(desc.T)  # good_tiles x query_tiles
        max_sim = np.max(dots, axis=0)
        dist = 1.0 - max_sim

        top_k = max(1, int(math.ceil(float(params["anomaly_top_k"]) * float(dist.size))))
        top_vals = np.sort(dist)[::-1][:top_k]
        anomaly_bank = clamp01(float(top_vals.mean()) / 0.34)

        # Build coarse anomaly map.
        tile_vals = dist.reshape((-1,))
        idx = 0
        for y in range(0, CROP_H, TILE_H):
            for x in range(0, CROP_W, TILE_W):
                y2 = min(CROP_H, y + TILE_H)
                x2 = min(CROP_W, x + TILE_W)
                if idx < tile_vals.size:
                    anomaly_map[y:y2, x:x2] = float(tile_vals[idx])
                idx += 1

    anomaly_diff = clamp01(float(np.percentile(blur, 99.5)) / 120.0)
    anomaly = clamp01(max(anomaly_bank, anomaly_diff))

    anomaly_u8 = (np.clip(anomaly_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    anomaly_heat = cv2.applyColorMap(anomaly_u8, cv2.COLORMAP_HOT)

    diff_norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    known_max = float(max(class_scores.values())) if class_scores else 0.0

    fused = float(
        np.clip(
            max(
                known_max,
                float(params["w_known"]) * known_max + float(params["w_anomaly"]) * anomaly,
            ),
            0.0,
            1.0,
        )
    )

    if align_conf < float(params["min_align_conf"]):
        penalty = 0.62 + 0.38 * (align_conf / max(1e-6, float(params["min_align_conf"])))
        fused = float(np.clip(fused * penalty, 0.0, 1.0))

    near = 1.0 - min(1.0, abs(fused - float(params["reject_threshold"])) / max(0.05, float(params["review_band"]) + 0.05))
    align_unc = 1.0 - align_conf
    uncertainty = clamp01(0.65 * near + 0.35 * align_unc)

    frame_pred = "none"
    if known_max >= float(params["known_threshold"]):
        frame_pred = max(class_scores, key=lambda k: class_scores[k])
    elif fused >= float(params["reject_threshold"]):
        frame_pred = "unknown_anomaly"
    elif fused >= float(params["reject_threshold"]) - float(params["review_band"]):
        frame_pred = "review"

    return {
        "crop": crop,
        "gray": gray,
        "diff_heat_bgr": diff_heat,
        "anomaly_heat_bgr": anomaly_heat,
        "mask": mask,
        "class_scores": class_scores,
        "known_max": known_max,
        "anomaly": anomaly,
        "anomaly_bank": anomaly_bank,
        "anomaly_diff": anomaly_diff,
        "fused": fused,
        "align_conf": align_conf,
        "uncertainty": uncertainty,
        "frame_pred": frame_pred,
        "bboxes": bboxes,
        "n_contours": len(contours),
    }


# ---------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------

def open_video_writer(fps: int) -> Tuple[Any, str, str]:
    timestamp = int(time.time() * 1000)
    candidates = [
        ("mp4v", ".mp4", "video/mp4"),
        ("avc1", ".mp4", "video/mp4"),
        ("MJPG", ".avi", "video/avi"),
    ]

    last_error = ""

    for cc, ext, mime in candidates:
        path = os.path.join(tempfile.gettempdir(), f"advanced_defect_demo_{timestamp}_{cc}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(path, fourcc, float(max(1, fps)), (FRAME_W, FRAME_H))

        if writer.isOpened():
            return writer, path, mime

        writer.release()
        last_error = f"Failed to open writer with {cc}{ext}."

    raise RuntimeError(f"Could not create demo video. {last_error}")


# ---------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------

def ema_series(arr: np.ndarray, alpha: float) -> np.ndarray:
    if arr.size == 0:
        return arr

    alpha = float(np.clip(alpha, 0.05, 1.0))
    out = np.empty_like(arr, dtype=np.float32)
    prev = float(arr[0])
    out[0] = prev

    for i in range(1, arr.size):
        prev = alpha * float(arr[i]) + (1.0 - alpha) * prev
        out[i] = prev

    return out


def severity_tier(sev: float) -> str:
    s = float(sev)
    if s < 0.45:
        return "low"
    if s < 0.75:
        return "medium"
    return "high"


def finalize_products(
    products: List[Dict[str, Any]],
    trackers: Dict[int, Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []

    reject_th = float(params["reject_threshold"])
    known_th = float(params["known_threshold"])
    review_th = reject_th - float(params["review_band"])
    vote_ratio = float(params["vote_ratio"])

    for p in products:
        pid = int(p["id"])
        tr = trackers.get(pid, {"scores": []})
        scores: List[Dict[str, Any]] = tr.get("scores", [])
        frames = len(scores)

        is_defect = str(p["defect"]) != "none"

        if frames == 0:
            logs.append(
                {
                    "id": pid,
                    "spawn_frame": int(p["spawn_frame"]),
                    "product_type": str(p["product_type"]),
                    "gt": str(p["defect"]),
                    "severity": float(p["severity"]),
                    "severity_tier": severity_tier(p["severity"]),
                    "included": False,
                    "frames": 0,
                    "is_defect": is_defect,
                    "decision": "no_data",
                    "pred_class": "none",
                    "fused_mean": 0.0,
                    "peak_fused": 0.0,
                    "peak_known": 0.0,
                    "anomaly_mean": 0.0,
                    "align_mean": 0.0,
                    "uncertainty_mean": 0.0,
                    "temporal_std": 0.0,
                    "escaped": False,
                    "false_reject": False,
                    "reviewed": False,
                }
            )
            continue

        fused_arr = np.array([float(s["fused"]) for s in scores], dtype=np.float32)
        known_arr = np.array([float(s["known_max"]) for s in scores], dtype=np.float32)
        anomaly_arr = np.array([float(s["anomaly"]) for s in scores], dtype=np.float32)
        align_arr = np.array([float(s["align_conf"]) for s in scores], dtype=np.float32)
        unc_arr = np.array([float(s["uncertainty"]) for s in scores], dtype=np.float32)

        fused_sm = ema_series(fused_arr, float(params["temporal_ema"]))
        known_sm = ema_series(known_arr, float(params["temporal_ema"]))

        votes = int(np.sum((fused_sm >= reject_th) | (known_sm >= known_th)))
        required = max(1, int(math.ceil(vote_ratio * float(frames))))

        if votes >= required:
            decision = "reject"
        elif float(np.max(fused_sm)) >= review_th:
            decision = "review"
        else:
            decision = "accept"

        class_means = {
            k: float(np.mean([float(s["class_scores"][k]) for s in scores]))
            for k in KNOWN_DEFECTS
        }
        best_class = max(class_means, key=lambda k: class_means[k])

        if decision == "reject":
            pred_class = best_class if class_means[best_class] >= 0.16 else "unknown_anomaly"
        elif decision == "review":
            pred_class = best_class if class_means[best_class] >= 0.16 else "review"
        else:
            pred_class = "none"

        escaped = bool(is_defect and decision == "accept")
        false_reject = bool((not is_defect) and decision == "reject")
        reviewed = bool(decision == "review")

        logs.append(
            {
                "id": pid,
                "spawn_frame": int(p["spawn_frame"]),
                "product_type": str(p["product_type"]),
                "gt": str(p["defect"]),
                "severity": float(p["severity"]),
                "severity_tier": severity_tier(p["severity"]),
                "included": True,
                "frames": frames,
                "is_defect": is_defect,
                "decision": decision,
                "pred_class": pred_class,
                "fused_mean": float(fused_arr.mean()),
                "peak_fused": float(fused_sm.max()),
                "peak_known": float(known_sm.max()),
                "anomaly_mean": float(anomaly_arr.mean()),
                "align_mean": float(align_arr.mean()),
                "uncertainty_mean": float(unc_arr.mean()),
                "temporal_std": float(fused_arr.std()),
                "escaped": escaped,
                "false_reject": false_reject,
                "reviewed": reviewed,
            }
        )

    return logs


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def bootstrap_binary_ci(values: np.ndarray, seed: int = 99, n_boot: int = 800) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)

    if arr.size == 0:
        return float("nan"), float("nan")

    if arr.size == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float32)

    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(sample.mean())

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def roc_pr_curves(labels: List[bool], scores: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float32)

    if y.size == 0 or not np.any(y) or np.all(y):
        empty = pd.DataFrame(columns=["threshold", "fpr", "tpr", "precision", "recall"])
        return empty, empty.copy(), float("nan"), float("nan")

    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    thresholds = np.unique(s)[::-1]
    rows = []

    tp = 0
    fp = 0
    total_pos = int(np.sum(y))
    total_neg = int(np.sum(~y))

    # Walk thresholds descending.
    idx = 0
    for thr in thresholds:
        while idx < s.size and s[idx] >= thr:
            if y[idx]:
                tp += 1
            else:
                fp += 1
            idx += 1

        fn = total_pos - tp
        tpr = tp / max(1, total_pos)
        fpr = fp / max(1, total_neg)
        precision = tp / max(1, tp + fp)
        recall = tpr

        rows.append(
            {
                "threshold": float(thr),
                "fpr": float(fpr),
                "tpr": float(tpr),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    rows.insert(0, {"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0, "precision": 1.0, "recall": 0.0})
    rows.append({"threshold": -float("inf"), "fpr": 1.0, "tpr": 1.0, "precision": total_pos / max(1, y.size), "recall": 1.0})

    df = pd.DataFrame(rows)
    roc_df = df[["threshold", "fpr", "tpr"]].copy()
    pr_df = df[["threshold", "recall", "precision"]].copy()

    auc = auc_trapz(df["fpr"].to_numpy(dtype=np.float32), df["tpr"].to_numpy(dtype=np.float32))

    # Average precision.
    rec = df["recall"].to_numpy(dtype=np.float32)
    prec = df["precision"].to_numpy(dtype=np.float32)
    ap = float(np.sum(np.diff(rec, prepend=0.0) * prec))

    return roc_df, pr_df, auc, ap


def compute_metrics(
    logs: List[Dict[str, Any]],
    frame_ms: List[float],
    brightness: List[float],
    contrast: List[float],
    blur: List[float],
    fused_all: List[float],
    anomaly_all: List[float],
    align_all: List[float],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    inc = [l for l in logs if bool(l.get("included", False))]
    df = pd.DataFrame(inc)

    base = {
        "n_products": len(inc),
        "n_good": 0,
        "n_defect": 0,
        "n_no_data": len(logs) - len(inc),
        "escape_rate": float("nan"),
        "escape_ci": (float("nan"), float("nan")),
        "false_reject_rate": float("nan"),
        "false_reject_ci": (float("nan"), float("nan")),
        "review_rate": float("nan"),
        "overkill_rate": float("nan"),
        "underkill_rate": float("nan"),
        "repeatability_std": float("nan"),
        "brightness_psi": float("nan"),
        "contrast_psi": float("nan"),
        "blur_psi": float("nan"),
        "prediction_psi": float("nan"),
        "latency_mean_ms": safe_mean(frame_ms),
        "latency_p50_ms": safe_percentile(frame_ms, 50),
        "latency_p95_ms": safe_percentile(frame_ms, 95),
        "fused_hist": histogram_counts(fused_all),
        "anomaly_hist": histogram_counts(anomaly_all),
        "align_hist": histogram_counts(align_all),
        "brightness_hist": histogram_counts(brightness),
        "latency_hist": histogram_counts(frame_ms),
        "roc_df": pd.DataFrame(),
        "pr_df": pd.DataFrame(),
        "auc": float("nan"),
        "ap": float("nan"),
        "confusion_df": pd.DataFrame(),
        "slices_df": pd.DataFrame(),
        "class_recall_df": pd.DataFrame(),
        "reliability_df": pd.DataFrame(),
        "gates_df": pd.DataFrame(),
        "plc_df": pd.DataFrame(),
    }

    if df.empty:
        return base

    base["n_good"] = int((~df["is_defect"]).sum())
    base["n_defect"] = int(df["is_defect"].sum())

    escape_vals = df.loc[df["is_defect"], "escaped"].to_numpy(dtype=np.float32)
    fr_vals = df.loc[~df["is_defect"], "false_reject"].to_numpy(dtype=np.float32)

    base["escape_rate"] = float(escape_vals.mean()) if escape_vals.size else float("nan")
    base["false_reject_rate"] = float(fr_vals.mean()) if fr_vals.size else float("nan")
    base["review_rate"] = float(df["reviewed"].mean())
    base["overkill_rate"] = base["false_reject_rate"]
    base["underkill_rate"] = base["escape_rate"]

    base["escape_ci"] = bootstrap_binary_ci(escape_vals, seed=101)
    base["false_reject_ci"] = bootstrap_binary_ci(fr_vals, seed=103)

    base["repeatability_std"] = float(df.loc[df["frames"] >= 3, "temporal_std"].mean()) if (df["frames"] >= 3).any() else float("nan")

    base["brightness_psi"] = population_stability_index(brightness[: len(brightness) // 2], brightness[len(brightness) // 2:])
    base["contrast_psi"] = population_stability_index(contrast[: len(contrast) // 2], contrast[len(contrast) // 2:])
    base["blur_psi"] = population_stability_index(blur[: len(blur) // 2], blur[len(blur) // 2:])
    base["prediction_psi"] = population_stability_index(fused_all[: len(fused_all) // 2], fused_all[len(fused_all) // 2:])

    roc_df, pr_df, auc, ap = roc_pr_curves(df["is_defect"].tolist(), df["peak_fused"].tolist())
    base["roc_df"] = roc_df
    base["pr_df"] = pr_df
    base["auc"] = auc
    base["ap"] = ap

    # Confusion matrix.
    try:
        conf = pd.crosstab(df["gt"], df["pred_class"])
        conf = conf.reindex(index=sorted(set(list(ALL_DEFECTS) + ["none"])), fill_value=0)
        conf = conf.reindex(columns=sorted(set(list(ALL_DEFECTS) + ["none", "unknown_anomaly", "review"])), fill_value=0)
        base["confusion_df"] = conf.reset_index()
    except Exception:
        base["confusion_df"] = pd.DataFrame()

    # Per-class recall.
    class_rows = []
    for cls in ALL_DEFECTS:
        sub = df[df["gt"] == cls]
        if sub.empty:
            continue
        class_rows.append(
            {
                "Defect class": DEFECT_LABELS.get(cls, cls),
                "Count": int(len(sub)),
                "Escape rate": float(sub["escaped"].mean()),
                "Review rate": float(sub["reviewed"].mean()),
                "Mean severity": float(sub["severity"].mean()),
            }
        )
    base["class_recall_df"] = pd.DataFrame(class_rows)

    # Slices.
    slice_rows = []
    for col, label in [("product_type", "Product"), ("severity_tier", "Severity")]:
        for val, grp in df.groupby(col):
            defects = grp[grp["is_defect"]]
            goods = grp[~grp["is_defect"]]
            slice_rows.append(
                {
                    "Dimension": label,
                    "Slice": str(val),
                    "Products": int(len(grp)),
                    "Escape rate": float(defects["escaped"].mean()) if not defects.empty else float("nan"),
                    "False-reject rate": float(goods["false_reject"].mean()) if not goods.empty else float("nan"),
                    "Review rate": float(grp["reviewed"].mean()),
                }
            )
    base["slices_df"] = pd.DataFrame(slice_rows)

    # Reliability curve.
    try:
        scores = df["peak_fused"].to_numpy(dtype=np.float32)
        labels = df["is_defect"].to_numpy(dtype=bool)
        qs = np.quantile(scores, np.linspace(0, 1, 11))
        qs = np.unique(qs)

        rel_rows = []
        for i in range(len(qs) - 1):
            m = (scores >= qs[i]) & (scores <= qs[i + 1] if i == len(qs) - 2 else scores < qs[i + 1])
            if not np.any(m):
                continue
            rel_rows.append(
                {
                    "bin": i + 1,
                    "mean_score": float(scores[m].mean()),
                    "defect_rate": float(labels[m].mean()),
                    "count": int(np.sum(m)),
                }
            )
        base["reliability_df"] = pd.DataFrame(rel_rows)
    except Exception:
        base["reliability_df"] = pd.DataFrame()

    # Gates.
    def gate_status(rate: float, ci: Tuple[float, float], target: float, n: int, min_n: int = 1000) -> str:
        if rate is None or np.isnan(rate) or n == 0:
            return "NO DATA"
        if rate < target:
            if n >= min_n:
                return "PASS"
            if (not np.isnan(ci[1])) and ci[1] < target:
                return "PASS"
            return "PROVISIONAL"
        return "FAIL"

    escape_status = gate_status(base["escape_rate"], base["escape_ci"], 0.005, base["n_defect"])
    fr_status = gate_status(base["false_reject_rate"], base["false_reject_ci"], 0.02, base["n_good"])
    lat_status = "PASS" if base["latency_p95_ms"] <= float(params["latency_budget_ms"]) else "FAIL"
    drift_status = "PASS" if (not np.isnan(base["brightness_psi"]) and base["brightness_psi"] <= 0.20) else "CHECK"

    base["gates_df"] = pd.DataFrame(
        [
            {
                "Gate": "Escape rate < 0.50%",
                "Measured": base["escape_rate"],
                "95% CI low": base["escape_ci"][0],
                "95% CI high": base["escape_ci"][1],
                "Sample size": base["n_defect"],
                "Status": escape_status,
            },
            {
                "Gate": "False-reject rate < 2.00%",
                "Measured": base["false_reject_rate"],
                "95% CI low": base["false_reject_ci"][0],
                "95% CI high": base["false_reject_ci"][1],
                "Sample size": base["n_good"],
                "Status": fr_status,
            },
            {
                "Gate": f"p95 latency <= {float(params['latency_budget_ms']):.0f} ms",
                "Measured": base["latency_p95_ms"],
                "95% CI low": float("nan"),
                "95% CI high": float("nan"),
                "Sample size": len(frame_ms),
                "Status": lat_status,
            },
            {
                "Gate": "Brightness drift PSI <= 0.20",
                "Measured": base["brightness_psi"],
                "95% CI low": float("nan"),
                "95% CI high": float("nan"),
                "Sample size": len(brightness),
                "Status": drift_status,
            },
        ]
    )

    # PLC events.
    plc_rows = []
    for _, r in df.sort_values("spawn_frame").iterrows():
        sig = r["decision"]
        sig_num = 0 if sig == "accept" else (1 if sig == "review" else 2)
        plc_rows.append(
            {
                "product_id": int(r["id"]),
                "time_s": round(float(r["spawn_frame"]) / max(1.0, float(params["fps"])), 2),
                "signal": str(sig).upper(),
                "signal_num": sig_num,
                "reason": DEFECT_LABELS.get(str(r["pred_class"]), str(r["pred_class"])),
                "peak_fused": round(float(r["peak_fused"]), 3),
                "peak_known": round(float(r["peak_known"]), 3),
            }
        )
    base["plc_df"] = pd.DataFrame(plc_rows)

    return base


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def calibrate_thresholds(
    logs: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    inc = [l for l in logs if bool(l.get("included", False))]
    if not inc:
        return None, pd.DataFrame()

    fused = np.array([float(l["peak_fused"]) for l in inc], dtype=np.float32)
    known = np.array([float(l["peak_known"]) for l in inc], dtype=np.float32)
    isdef = np.array([bool(l["is_defect"]) for l in inc], dtype=bool)

    escape_cost = float(params["escape_cost"])
    fr_cost = float(params["false_reject_cost"])
    review_cost = float(params["review_cost"])
    review_band = float(params["review_band"])

    best = None
    rows = []

    for rt in np.arange(0.05, 0.81, 0.02):
        for kt in np.arange(0.05, 0.91, 0.02):
            reject = (fused >= rt) | (known >= kt)
            review = (~reject) & (fused >= rt - review_band)

            esc = int(np.sum(isdef & (~reject)))
            fr = int(np.sum((~isdef) & reject))
            rv = int(np.sum(review))

            cost = esc * escape_cost + fr * fr_cost + rv * review_cost

            row = {
                "reject_threshold": round(float(rt), 3),
                "known_threshold": round(float(kt), 3),
                "escapes": esc,
                "false_rejects": fr,
                "reviews": rv,
                "expected_cost": round(float(cost), 2),
            }
            rows.append(row)

            if best is None or cost < float(best["expected_cost"]):
                best = row

    df = pd.DataFrame(rows).sort_values("expected_cost").head(12)
    return best, df


# ---------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------

def apply_feedback(logs: List[Dict[str, Any]], feedback: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for l in logs:
        nl = dict(l)
        fb = feedback.get(str(int(l["id"])))

        if fb:
            action = fb.get("action", "keep")
            if action == "good":
                nl["is_defect"] = False
                nl["gt"] = "none"
            elif action == "defect":
                nl["is_defect"] = True
                nl["gt"] = fb.get("cls", "unknown")

            nl["escaped"] = bool(nl["is_defect"] and nl["decision"] == "accept")
            nl["false_reject"] = bool((not nl["is_defect"]) and nl["decision"] == "reject")

        out.append(nl)

    return out


# ---------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------

def classify_failure(log: Dict[str, Any], params: Dict[str, Any]) -> str:
    if bool(log["escaped"]):
        if str(log["gt"]) == "unknown":
            return "unknown_defect_missed"
        if float(log["align_mean"]) < 0.55:
            return "alignment_loss_escape"
        if float(log["peak_known"]) < 0.15 and float(log["anomaly_mean"]) < 0.20:
            return "low_signal_escape"
        return "known_defect_missed"

    if bool(log["false_reject"]):
        if float(log["align_mean"]) < 0.60:
            return "alignment_false_reject"
        if float(log["anomaly_mean"]) > 0.28:
            return "environment_anomaly_false_reject"
        return "threshold_false_reject"

    return "none"


FAILURE_FIXES = {
    "unknown_defect_missed": "Lower anomaly threshold, expand good-sample bank, or add environmental variants.",
    "alignment_loss_escape": "Improve golden alignment, add rotation search, or route low-alignment parts to review.",
    "low_signal_escape": "Increase lighting contrast, use polarized light, or lower minimum detectable defect area.",
    "known_defect_missed": "Collect more examples of this defect type and recalibrate known-defect scorer.",
    "alignment_false_reject": "Increase alignment tolerance or send low-alignment parts to manual review.",
    "environment_anomaly_false_reject": "Suppress dust/glare/noise, clean lens, or recalibrate anomaly threshold.",
    "threshold_false_reject": "Raise reject threshold or require stronger temporal confirmation.",
    "none": "No action.",
}


# ---------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------

def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    start_wall = time.perf_counter()

    rng = np.random.default_rng(int(params["seed"]))
    products, px_per_frame, total_frames = build_products(params, rng)

    banks: Dict[str, np.ndarray] = {}
    for pt in PRODUCT_TYPES:
        bank_rng = np.random.default_rng(int(params["seed"]) * 7717 + hash(pt) % 100000)
        banks[pt] = build_good_bank(pt, params, bank_rng, count=int(params["bank_samples"]))

    writer, video_path, video_mime = open_video_writer(int(params["fps"]))

    trackers: Dict[int, Dict[str, Any]] = {}
    frame_ms: List[float] = []
    brightness: List[float] = []
    contrast: List[float] = []
    blur: List[float] = []

    fused_all: List[float] = []
    anomaly_all: List[float] = []
    align_all: List[float] = []

    last_frame_rgb: Optional[np.ndarray] = None

    try:
        for f in range(total_frames):
            t0 = time.perf_counter()

            frame, views, inspections = render_frame(f, products, px_per_frame, params, banks)

            for v, insp in zip(views, inspections):
                if insp is None or not bool(v["in_zone"]):
                    continue

                pid = int(v["id"])
                if pid not in trackers:
                    trackers[pid] = {
                        "scores": [],
                        "crop": v["crop_degraded"].copy(),
                        "gt_mask": v["gt_mask"].copy(),
                        "anomaly_heat": insp["anomaly_heat_bgr"].copy(),
                        "diff_heat": insp["diff_heat_bgr"].copy(),
                        "bbox": v["bbox"],
                        "product_type": v["product_type"],
                    }

                trackers[pid]["scores"].append(insp)
                trackers[pid]["bbox"] = v["bbox"]

                fused_all.append(float(insp["fused"]))
                anomaly_all.append(float(insp["anomaly"]))
                align_all.append(float(insp["align_conf"]))

            belt_roi = frame[BELT_TOP:BELT_BOTTOM, :]
            belt_gray = cv2.cvtColor(belt_roi, cv2.COLOR_BGR2GRAY)
            brightness.append(float(belt_gray.mean()))
            contrast.append(float(belt_gray.std()))
            lap = cv2.Laplacian(belt_gray, cv2.CV_64F)
            blur.append(float(lap.var()))

            writer.write(frame)
            last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)

    finally:
        writer.release()

    logs = finalize_products(products, trackers, params)
    metrics = compute_metrics(
        logs=logs,
        frame_ms=frame_ms,
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        fused_all=fused_all,
        anomaly_all=anomaly_all,
        align_all=align_all,
        params=params,
    )

    with open(video_path, "rb") as fh:
        video_bytes = fh.read()

    try:
        os.remove(video_path)
    except OSError:
        pass

    wall_time = time.perf_counter() - start_wall

    return {
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "last_frame_rgb": last_frame_rgb,
        "logs": logs,
        "details": trackers,
        "metrics": metrics,
        "frame_ms": frame_ms,
        "total_frames": total_frames,
        "wall_time": wall_time,
        "params": dict(params),
    }


# ---------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------

def build_gallery(
    logs: List[Dict[str, Any]],
    details: Dict[int, Dict[str, Any]],
    params: Dict[str, Any],
    limit: int = 18,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[int, float, Dict[str, Any], Dict[str, Any]]] = []

    rt = float(params["reject_threshold"])
    kt = float(params["known_threshold"])

    for l in logs:
        if not bool(l.get("included", False)):
            continue

        det = details.get(int(l["id"]))
        if not det or det.get("crop") is None:
            continue

        margin = min(
            abs(float(l["peak_fused"]) - rt),
            abs(float(l["peak_known"]) - kt),
        )

        priority = 0
        if bool(l["escaped"]) or bool(l["false_reject"]):
            priority = 3
        elif bool(l["reviewed"]) or str(l["pred_class"]) == "unknown_anomaly" or float(l["align_mean"]) < 0.60:
            priority = 2
        elif margin < 0.10:
            priority = 1

        if priority > 0:
            candidates.append((priority, margin, l, det))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    out: List[Dict[str, Any]] = []
    for _, _, l, det in candidates[:max(1, int(limit))]:
        out.append(
            {
                "id": int(l["id"]),
                "image_rgb": cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB),
                "gt": str(l["gt"]),
                "pred": str(l["pred_class"]),
                "decision": str(l["decision"]),
                "fused": float(l["peak_fused"]),
                "known": float(l["peak_known"]),
                "anomaly": float(l["anomaly_mean"]),
                "align": float(l["align_mean"]),
                "product_type": str(l["product_type"]),
            }
        )

    return out


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def run_builtin_tests() -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    def add(name: str, ok: bool, value: str = "") -> None:
        results.append({"Test": name, "Status": "PASS" if ok else "FAIL", "Value": value})

    p = default_params()

    add(
        "golden_shapes",
        all(GOLDENS[k].shape == (CROP_H, CROP_W, 3) for k in PRODUCT_TYPES),
        str({k: GOLDENS[k].shape for k in PRODUCT_TYPES}),
    )

    rng = np.random.default_rng(123)
    bank = build_good_bank("plate", p, rng, count=8)

    good_insp = inspect_product(GOLDENS["plate"].copy(), "plate", p, bank)
    add(
        "good_low_score",
        good_insp["fused"] < p["reject_threshold"],
        f"fused={good_insp['fused']:.3f}, anomaly={good_insp['anomaly']:.3f}",
    )

    rng2 = np.random.default_rng(1)
    for defect in ALL_DEFECTS:
        img, _, _ = synthesize_defect(GOLDENS["plate"].copy(), defect, 0.90, rng2, "plate")
        insp = inspect_product(img, "plate", p, bank)

        if defect in KNOWN_DEFECTS:
            score = float(insp["class_scores"][defect])
            ok = score >= float(p["known_threshold"]) or float(insp["fused"]) >= float(p["reject_threshold"])
            value = f"class={score:.3f}, fused={insp['fused']:.3f}"
        else:
            ok = float(insp["fused"]) >= float(p["reject_threshold"]) or float(insp["anomaly"]) > 0.22
            value = f"anomaly={insp['anomaly']:.3f}, fused={insp['fused']:.3f}"

        add(f"detects_{defect}", bool(ok), value)

    # Deterministic inspection.
    insp_a = inspect_product(GOLDENS["plate"].copy(), "plate", p, bank)
    insp_b = inspect_product(GOLDENS["plate"].copy(), "plate", p, bank)
    add(
        "deterministic_inspect",
        bool(
            np.isclose(insp_a["fused"], insp_b["fused"], atol=1e-6)
            and np.isclose(insp_a["anomaly"], insp_b["anomaly"], atol=1e-6)
        ),
        f"fused={insp_a['fused']:.6f}",
    )

    try:
        writer, path, mime = open_video_writer(5)
        black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(black)
        writer.release()

        ok = os.path.exists(path) and os.path.getsize(path) > 0
        if ok:
            try:
                os.remove(path)
            except OSError:
                pass

        add("video_writer", ok, mime)
    except Exception as exc:
        add("video_writer", False, str(exc))

    return results


def run_edge_case_tests(params: Dict[str, Any]) -> List[Dict[str, str]]:
    p = dict(params)
    p["duration"] = 1.0
    p["fps"] = 5

    rng = np.random.default_rng(int(p["seed"]) + 555)
    bank = build_good_bank("plate", p, rng, count=8)

    rows: List[Dict[str, str]] = []

    def finite(insp: Dict[str, Any]) -> bool:
        vals = [
            float(insp["fused"]),
            float(insp["anomaly"]),
            float(insp["align_conf"]),
            float(insp["known_max"]),
        ]
        return bool(np.all(np.isfinite(vals)))

    allowed = ("none", "review", "unknown_anomaly") + KNOWN_DEFECTS

    for ang in [-12.0, 0.0, 12.0]:
        img = rotate_image(GOLDENS["plate"], ang)
        deg = apply_camera_effects(img, 1, p, np.random.default_rng(int(p["seed"]) + int(abs(ang) * 100)), is_crop=True)
        insp = inspect_product(deg, "plate", p, bank)
        ok = finite(insp) and str(insp["frame_pred"]) in allowed
        rows.append(
            {
                "Scenario": f"Part orientation change {ang:+.0f} deg",
                "Expected": "No crash; valid output",
                "Status": "PASS" if ok else "FAIL",
                "Evidence": f"pred={insp['frame_pred']}, fused={insp['fused']:.3f}, align={insp['align_conf']:.3f}",
            }
        )

    p_dust = dict(p)
    p_dust["dust"] = 0.95
    deg = apply_camera_effects(GOLDENS["plate"], 2, p_dust, rng, is_crop=True)
    insp = inspect_product(deg, "plate", p_dust, bank)
    rows.append(
        {
            "Scenario": "Heavy dust / lens contamination",
            "Expected": "No crash; safe output or review",
            "Status": "PASS" if finite(insp) else "FAIL",
            "Evidence": f"pred={insp['frame_pred']}, fused={insp['fused']:.3f}, anomaly={insp['anomaly']:.3f}",
        }
    )

    for gain in [0.55, 1.00, 1.55]:
        img = (GOLDENS["plate"].astype(np.float32) * float(gain)).clip(0, 255).astype(np.uint8)
        insp = inspect_product(img, "plate", p, bank)
        rows.append(
            {
                "Scenario": f"Lighting drift gain={gain:.2f}",
                "Expected": "Finite metric; no crash",
                "Status": "PASS" if finite(insp) else "FAIL",
                "Evidence": f"pred={insp['frame_pred']}, fused={insp['fused']:.3f}, align={insp['align_conf']:.3f}",
            }
        )

    p_blur = dict(p)
    p_blur["motion_blur"] = 0.95
    deg = apply_camera_effects(GOLDENS["plate"], 3, p_blur, rng, is_crop=True)
    insp = inspect_product(deg, "plate", p_blur, bank)
    rows.append(
        {
            "Scenario": "Heavy motion blur",
            "Expected": "Finite metric; review or safe fallback",
            "Status": "PASS" if finite(insp) else "FAIL",
            "Evidence": f"pred={insp['frame_pred']}, fused={insp['fused']:.3f}, align={insp['align_conf']:.3f}",
        }
    )

    img, _, _ = synthesize_defect(GOLDENS["plate"].copy(), "unknown", 0.90, rng, "plate")
    insp = inspect_product(img, "plate", p, bank)
    ok = (
        finite(insp)
        and (
            float(insp["anomaly"]) > 0.10
            or float(insp["fused"]) >= float(p["reject_threshold"])
            or str(insp["frame_pred"]) == "unknown_anomaly"
        )
    )
    rows.append(
        {
            "Scenario": "New defect type never seen in training",
            "Expected": "Anomaly flag, review, or reject",
            "Status": "PASS" if ok else "FAIL",
            "Evidence": f"pred={insp['frame_pred']}, fused={insp['fused']:.3f}, anomaly={insp['anomaly']:.3f}",
        }
    )

    return rows


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def build_report_markdown(result: Dict[str, Any], feedback: Dict[str, Dict[str, Any]]) -> str:
    m = result["metrics"]
    p = result["params"]

    adjusted_logs = apply_feedback(result["logs"], feedback)
    am = compute_metrics(
        logs=adjusted_logs,
        frame_ms=result["frame_ms"],
        brightness=[0.0],
        contrast=[0.0],
        blur=[0.0],
        fused_all=[],
        anomaly_all=[],
        align_all=[],
        params=p,
    )

    lines = [
        "# Advanced Production Line Defect Inspection Report",
        "",
        "## 1. Scope",
        "This report is generated from a synthetic local demo. It is not a certified production validation.",
        "",
        "## 2. Assumptions",
        f"- Seed: {p['seed']}",
        f"- Duration: {p['duration']} s",
        f"- FPS: {p['fps']}",
        f"- Line speed: {p['line_speed']} products/min",
        f"- PLC mode: {p['plc_mode']}",
        f"- Latency budget: {p['latency_budget_ms']} ms",
        "",
        "## 3. Headline metrics",
        f"- Products with data: {m['n_products']}",
        f"- Good parts: {m['n_good']}",
        f"- Defective parts: {m['n_defect']}",
        f"- Escape rate: {m['escape_rate'] * 100.0:.2f}%" if not np.isnan(m['escape_rate']) else "- Escape rate: n/a",
        f"- False-reject rate: {m['false_reject_rate'] * 100.0:.2f}%" if not np.isnan(m['false_reject_rate']) else "- False-reject rate: n/a",
        f"- Review rate: {m['review_rate'] * 100.0:.2f}%" if not np.isnan(m['review_rate']) else "- Review rate: n/a",
        f"- p95 latency: {m['latency_p95_ms']:.1f} ms" if not np.isnan(m['latency_p95_ms']) else "- p95 latency: n/a",
        f"- Brightness drift PSI: {m['brightness_psi']:.3f}" if not np.isnan(m['brightness_psi']) else "- Brightness drift PSI: n/a",
        "",
        "## 4. Adjusted metrics after operator feedback",
        f"- Products with data: {am['n_products']}",
        f"- Escape rate: {am['escape_rate'] * 100.0:.2f}%" if not np.isnan(am['escape_rate']) else "- Escape rate: n/a",
        f"- False-reject rate: {am['false_reject_rate'] * 100.0:.2f}%" if not np.isnan(am['false_reject_rate']) else "- False-reject rate: n/a",
        "",
        "## 5. Release gates",
    ]

    if not m["gates_df"].empty:
        for _, r in m["gates_df"].iterrows():
            lines.append(
                f"- {r['Gate']}: measured={float(r['Measured']):.4f}, status={r['Status']}"
                if not np.isnan(float(r["Measured"]))
                else f"- {r['Gate']}: measured=n/a, status={r['Status']}"
            )
    else:
        lines.append("- No gate data.")

    lines += [
        "",
        "## 6. Failure analysis",
    ]

    failures = [l for l in adjusted_logs if bool(l.get("included", False)) and (bool(l["escaped"]) or bool(l["false_reject"]))]
    if not failures:
        lines.append("- No escapes or false rejects in this sample.")
    else:
        cats = Counter(classify_failure(l, p) for l in failures)
        for k, v in cats.most_common():
            lines.append(f"- {k}: {v} | fix: {FAILURE_FIXES.get(k, 'Investigate manually.')}")

    lines += [
        "",
        "## 7. Risk register",
        "- Data leakage via near-identical frames: split by scene/site/camera and deduplicate.",
        "- Overfitting to one sensor/location: hold out entire sites and run cross-domain tests.",
        "- Metric looks good but misses safety-critical cases: report per-slice recall and fixed-FPR recall.",
        "- Latency regression after export: automate parity and latency CI checks.",
        "- Label noise caps accuracy: estimate noise ceiling from double-annotated subset.",
        "",
        "## 8. Staged rollout",
        "1. Shadow mode: no PLC reject, log only.",
        "2. Limited pilot: automatic reject on one station.",
        "3. Production: full automatic reject with review queue.",
        "",
        "## 9. Security, privacy, ethics",
        "- Store only images required for inspection and traceability.",
        "- Apply least-privilege access to image archives and PLC interfaces.",
        "- Provide operator contest/correction path for automated decisions.",
        "- This system is not a safety interlock and must not be used for human presence detection.",
    ]

    return "\n".join(lines)


def build_report_json(result: Dict[str, Any], feedback: Dict[str, Dict[str, Any]]) -> str:
    m = result["metrics"]
    p = result["params"]

    summary = {
        "params": p,
        "metrics": {
            k: v
            for k, v in m.items()
            if k not in {"roc_df", "pr_df", "confusion_df", "slices_df", "class_recall_df", "reliability_df", "gates_df", "plc_df"}
        },
        "feedback": feedback,
        "n_logs": len(result["logs"]),
        "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return json.dumps(summary, indent=2, default=str)


def logs_to_csv_bytes(logs: List[Dict[str, Any]]) -> bytes:
    if not logs:
        return b""

    fieldnames = list(logs[0].keys())
    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=fieldnames)
    writer.writeheader()

    for row in logs:
        writer.writerow(row)

    return si.getvalue().encode("utf-8")


# ---------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------

def architecture_tradeoff_rows() -> List[Dict[str, str]]:
    return [
        {
            "Option": "Golden difference + morphology + tile anomaly bank",
            "Accuracy": "Good for demo and obvious geometric defects; provisional on synthetic data",
            "Latency": "Low; CPU-friendly",
            "Cost": "Low",
            "Complexity": "Medium",
            "Maintainability": "High",
            "Decision": "Chosen for local advanced demo",
        },
        {
            "Option": "Deep supervised detector only",
            "Accuracy": "Potentially higher for known classes",
            "Latency": "Medium to high",
            "Cost": "Higher GPU/inference cost",
            "Complexity": "Medium",
            "Maintainability": "Medium",
            "Decision": "Rejected as sole solution because it misses never-seen defects",
        },
        {
            "Option": "Pure unsupervised anomaly model only",
            "Accuracy": "Good for novelty, weaker for precise known-class labels",
            "Latency": "Medium",
            "Cost": "Medium",
            "Complexity": "Medium",
            "Maintainability": "Medium",
            "Decision": "Rejected as sole solution because known defects need class-specific thresholds",
        },
        {
            "Option": "Large multimodal foundation model",
            "Accuracy": "Unverified for this task",
            "Latency": "Likely too high for line speed",
            "Cost": "High",
            "Complexity": "High",
            "Maintainability": "Low",
            "Decision": "Rejected for deterministic factory inspection unless benchmarked on real line data",
        },
    ]


def data_readiness_rows() -> List[Dict[str, str]]:
    return [
        {
            "Source": "MVTec AD",
            "Licence": "Verify current licence before commercial use",
            "Access": "Public benchmark download",
            "Size": "Verify official current size",
            "Label quality": "Usually strong pixel/defect annotations",
            "Risk": "Lab bias, object bias, not identical to line cameras",
            "Mitigation": "Use only as benchmark/pretraining; validate on line data",
        },
        {
            "Source": "VisA",
            "Licence": "Verify current licence before commercial use",
            "Access": "Public benchmark download",
            "Size": "Verify official current size",
            "Label quality": "Generally good, verify per-class masks",
            "Risk": "Domain shift from production line",
            "Mitigation": "Cross-domain stress tests; do not treat as acceptance data",
        },
        {
            "Source": "Line good samples",
            "Licence": "Internal / customer data",
            "Access": "Production camera archive",
            "Size": "Target 300+ good samples per product",
            "Label quality": "High if captured under fixed optics",
            "Risk": "Near-duplicate frames, lighting drift, camera pose drift",
            "Mitigation": "Split by scene/site/camera; perceptual-hash dedupe; log pose/lighting",
        },
        {
            "Source": "Line defect samples",
            "Licence": "Internal / customer data",
            "Access": "Quality hold / reject bin",
            "Size": "Target 50+ examples per defect type",
            "Label quality": "Variable; needs double annotation",
            "Risk": "Class imbalance and label noise",
            "Mitigation": "Double-annotate 10%; measure agreement; copy-paste augmentation",
        },
        {
            "Source": "Synthetic demo data",
            "Licence": "Generated locally",
            "Access": "This app",
            "Size": "Configurable",
            "Label quality": "Perfect ground truth by construction",
            "Risk": "Not representative of real physics",
            "Mitigation": "Use only for UX, logic, and pipeline validation",
        },
    ]


def risk_register_rows() -> List[Dict[str, str]]:
    return [
        {
            "Risk": "Data leakage via near-identical frames across splits",
            "Likelihood": "Medium",
            "Impact": "High",
            "Owner": "Data engineer",
            "Early warning": "Duplicate hash matches across train/test",
            "Mitigation": "Split by scene/site/camera; hash-based duplicate check; report in README",
        },
        {
            "Risk": "Over-fitting to a single sensor or location",
            "Likelihood": "Medium",
            "Impact": "High",
            "Owner": "ML engineer",
            "Early warning": "Large drop on held-out site",
            "Mitigation": "Hold out entire sites; cross-domain test; stronger colour/geometry augmentation",
        },
        {
            "Risk": "Metric looks good but misses safety-critical cases",
            "Likelihood": "Medium",
            "Impact": "Critical",
            "Owner": "Quality engineer",
            "Early warning": "Worst-slice recall low while mean is high",
            "Mitigation": "Report recall at fixed false-positive rate and per-condition breakdowns",
        },
        {
            "Risk": "Latency regressions after export or quantisation",
            "Likelihood": "Medium",
            "Impact": "High",
            "Owner": "Deployment engineer",
            "Early warning": "p95 latency exceeds budget after export",
            "Mitigation": "Automate parity + latency CI check; keep FP16/FP32 fallback",
        },
        {
            "Risk": "Label noise capping achievable accuracy",
            "Likelihood": "High",
            "Impact": "Medium",
            "Owner": "Data lead",
            "Early warning": "Inter-annotator agreement low",
            "Mitigation": "Estimate noise ceiling from double-annotated subset; set realistic targets",
        },
        {
            "Risk": "Operator overrides silently bypass quality gate",
            "Likelihood": "Medium",
            "Impact": "High",
            "Owner": "Plant supervisor",
            "Early warning": "Override rate rises above baseline",
            "Mitigation": "Log all overrides, require reason code, review weekly, restrict permissions",
        },
    ]


def make_synthetic_manifest(params: Dict[str, Any], logs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for pt in PRODUCT_TYPES:
        sub = [l for l in logs if l.get("product_type") == pt and bool(l.get("included", False))]
        good = sum(1 for l in sub if not bool(l["is_defect"]))
        bad = len(sub) - good
        rows.append(
            {
                "split": "synthetic_demo",
                "product_type": pt,
                "good_count": good,
                "defect_count": bad,
                "hash": make_hash({"pt": pt, "params": params, "n": len(sub)}),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x * 100.0:.2f}%"


def fmt_ci(ci: Tuple[float, float]) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return "n/a"
    return f"{lo * 100.0:.2f}% to {hi * 100.0:.2f}%"


def fmt_float(x: float, digits: int = 3) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def display_result(result: Dict[str, Any]) -> None:
    m = result["metrics"]
    p = result["params"]
    logs = result["logs"]
    inc_logs = [l for l in logs if bool(l.get("included", False))]
    feedback = st.session_state.get("feedback", {})

    adjusted_logs = apply_feedback(logs, feedback)
    adjusted_metrics = compute_metrics(
        logs=adjusted_logs,
        frame_ms=result["frame_ms"],
        brightness=[],
        contrast=[],
        blur=[],
        fused_all=[],
        anomaly_all=[],
        align_all=[],
        params=p,
    )

    gallery = build_gallery(adjusted_logs, result["details"], p, limit=int(p["gallery_limit"]))

    tabs = st.tabs(
        [
            "Command Center",
            "Live Line",
            "Inspector",
            "Quality Gates",
            "Calibration",
            "Drift & PLC",
            "Failure Lab",
            "Architecture",
            "Data & Risk",
            "Rollout",
            "Tests",
            "Report",
        ]
    )

    # Command Center
    with tabs[0]:
        st.subheader("Command Center")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Products", m["n_products"])
        with c2:
            st.metric("Defects", m["n_defect"])
        with c3:
            st.metric("Escape", fmt_pct(m["escape_rate"]))
        with c4:
            st.metric("False reject", fmt_pct(m["false_reject_rate"]))
        with c5:
            st.metric("p95 latency", f"{fmt_float(m['latency_p95_ms'], 1)} ms")

        st.dataframe(m["gates_df"], use_container_width=True)

        st.markdown("#### Quick interpretation")
        if m["n_products"] < 50:
            st.warning("Small demo sample. Treat all acceptance gates as provisional.")
        if not np.isnan(m["escape_rate"]) and m["escape_rate"] >= 0.005:
            st.error("Escape gate not met in this sample. Lower threshold, improve anomaly bank, or collect harder defects.")
        if not np.isnan(m["false_reject_rate"]) and m["false_reject_rate"] >= 0.02:
            st.error("False-reject gate not met in this sample. Raise threshold, improve alignment, or reduce environment noise.")

        st.download_button(
            "Download product log CSV",
            data=logs_to_csv_bytes(logs),
            file_name="advanced_product_log.csv",
            mime="text/csv",
        )

    # Live Line
    with tabs[1]:
        st.subheader("Generated advanced production-line video")

        video_col, side_col = st.columns([2.35, 1.0])

        with video_col:
            st.video(
                result["video_bytes"],
                format=result["video_mime"],
                autoplay=True,
                muted=True,
                loop=True,
            )

            ext = ".avi" if "avi" in result["video_mime"] else ".mp4"
            st.download_button(
                "Download generated demo video",
                data=result["video_bytes"],
                file_name=f"advanced_production_line_defect_demo{ext}",
                mime=result["video_mime"],
            )

        with side_col:
            st.metric("Wall time", f"{float(result['wall_time']):.1f} s")
            st.metric("Frames", result["total_frames"])
            st.metric("Line speed", f"{int(p['line_speed'])} PPM")
            st.metric("PLC mode", p["plc_mode"])

            if result.get("last_frame_rgb") is not None:
                st.image(
                    result["last_frame_rgb"],
                    use_container_width=True,
                    caption="Last annotated frame.",
                )

        st.caption(
            "Synthetic but physically plausible demo. It validates inspection UX, scoring logic, "
            "temporal decision behavior, and dry-run PLC signaling locally."
        )

    # Inspector
    with tabs[2]:
        st.subheader("Operator inspector and hard-example queue")

        if gallery:
            st.markdown("#### Hard-example queue")
            cols = st.columns(4)
            for idx, item in enumerate(gallery[:12]):
                with cols[idx % 4]:
                    caption = (
                        f"#{item['id']} {item['decision'].upper()} | "
                        f"{PRODUCT_LABELS.get(item['product_type'], item['product_type'])} | "
                        f"GT {DEFECT_LABELS.get(item['gt'], item['gt'])} | f={item['fused']:.2f}"
                    )
                    st.image(item["image_rgb"], caption=caption, use_container_width=True)
        else:
            st.info("No hard examples were queued in this run.")

        st.divider()
        st.markdown("#### Single product deep dive")

        ids = [int(l["id"]) for l in adjusted_logs if bool(l.get("included", False))]
        if not ids:
            st.warning("No products with inspection data.")
        else:
            sel = st.selectbox("Select product ID", ids, key="detail_id")
            log = next(l for l in adjusted_logs if int(l["id"]) == int(sel))
            det = result["details"].get(int(sel))

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**Crop**")
                if det and det.get("crop") is not None:
                    st.image(cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.info("No crop.")

            with c2:
                st.markdown("**Diff heatmap**")
                if det and det.get("diff_heat") is not None:
                    st.image(cv2.cvtColor(det["diff_heat"], cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.info("No diff heatmap.")

            with c3:
                st.markdown("**Anomaly map**")
                if det and det.get("anomaly_heat") is not None:
                    st.image(cv2.cvtColor(det["anomaly_heat"], cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.info("No anomaly map.")

            with c4:
                st.markdown("**GT mask**")
                if det and det.get("gt_mask") is not None:
                    st.image(det["gt_mask"], channels="gray", use_container_width=True)
                else:
                    st.info("No GT mask.")

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Ground truth", DEFECT_LABELS.get(str(log["gt"]), str(log["gt"])))
                st.metric("Final decision", str(log["decision"]).upper())
            with mc2:
                st.metric("Predicted", DEFECT_LABELS.get(str(log["pred_class"]), str(log["pred_class"])))
                st.metric("Peak fused", fmt_float(float(log["peak_fused"]), 3))
            with mc3:
                st.metric("Peak known", fmt_float(float(log["peak_known"]), 3))
                st.metric("Alignment", fmt_float(float(log["align_mean"]), 3))

            if det and det.get("scores"):
                st.markdown("#### Class score means")
                class_means = {
                    k: float(np.mean([float(s["class_scores"][k]) for s in det["scores"]]))
                    for k in KNOWN_DEFECTS
                }

                cc = st.columns(4)
                for i, (k, v) in enumerate(class_means.items()):
                    with cc[i % 4]:
                        st.write(DEFECT_LABELS[k])
                        st.progress(min(1.0, max(0.0, v)))

                st.markdown("#### Temporal score timeline")
                fused_series = [float(s["fused"]) for s in det["scores"]]
                known_series = [float(s["known_max"]) for s in det["scores"]]
                anomaly_series = [float(s["anomaly"]) for s in det["scores"]]
                align_series = [float(s["align_conf"]) for s in det["scores"]]

                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    st.write("Fused")
                    st.line_chart(fused_series)
                with tc2:
                    st.write("Known")
                    st.line_chart(known_series)
                with tc3:
                    st.write("Anomaly")
                    st.line_chart(anomaly_series)
                with tc4:
                    st.write("Align")
                    st.line_chart(align_series)

            st.divider()
            st.markdown("#### Operator override")

            if "feedback" not in st.session_state:
                st.session_state.feedback = {}

            current_fb = st.session_state.feedback.get(str(sel), {})
            action = st.radio(
                "Override action",
                options=["keep", "good", "defect"],
                index=["keep", "good", "defect"].index(str(current_fb.get("action", "keep"))),
                horizontal=True,
                key=f"fb_action_{sel}",
            )

            cls = "unknown"
            if action == "defect":
                cls = st.selectbox(
                    "Defect class",
                    options=list(ALL_DEFECTS),
                    index=max(0, list(ALL_DEFECTS).index(str(current_fb.get("cls", "unknown")))),
                    key=f"fb_cls_{sel}",
                )

            if st.button("Save operator feedback", key=f"save_fb_{sel}"):
                st.session_state.feedback[str(sel)] = {"action": action, "cls": cls}
                st.success(f"Saved feedback for product #{sel}. Adjusted metrics are shown in Quality Gates.")

    # Quality Gates
    with tabs[3]:
        st.subheader("Quality gates and statistical evaluation")

        st.markdown("#### Automated metrics")
        st.dataframe(m["gates_df"], use_container_width=True)

        st.markdown("#### Adjusted metrics after operator feedback")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Adjusted escape", fmt_pct(adjusted_metrics["escape_rate"]))
        with ac2:
            st.metric("Adjusted false reject", fmt_pct(adjusted_metrics["false_reject_rate"]))
        with ac3:
            st.metric("Adjusted review", fmt_pct(adjusted_metrics["review_rate"]))
        with ac4:
            st.metric("Adjusted products", adjusted_metrics["n_products"])

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Overkill", fmt_pct(m["overkill_rate"]))
        with c2:
            st.metric("Underkill", fmt_pct(m["underkill_rate"]))
        with c3:
            st.metric("ROC AUC", fmt_float(m["auc"], 3))
        with c4:
            st.metric("PR AP", fmt_float(m["ap"], 3))

        if not m["roc_df"].empty:
            st.markdown("#### ROC curve")
            st.line_chart(m["roc_df"], x="fpr", y="tpr")

        if not m["pr_df"].empty:
            st.markdown("#### Precision-recall curve")
            st.line_chart(m["pr_df"], x="recall", y="precision")

        if not m["confusion_df"].empty:
            st.markdown("#### Confusion matrix")
            st.dataframe(m["confusion_df"], use_container_width=True)

        if not m["class_recall_df"].empty:
            st.markdown("#### Per-class recall")
            st.dataframe(m["class_recall_df"], use_container_width=True)

        if not m["slices_df"].empty:
            st.markdown("#### Slice breakdown")
            st.dataframe(m["slices_df"], use_container_width=True)

        if not m["reliability_df"].empty:
            st.markdown("#### Reliability curve")
            st.line_chart(m["reliability_df"], x="mean_score", y="defect_rate")

    # Calibration
    with tabs[4]:
        st.subheader("Cost-matrix threshold calibration")

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Escape cost", f"{float(p['escape_cost']):.2f}")
        with cc2:
            st.metric("False-reject cost", f"{float(p['false_reject_cost']):.2f}")
        with cc3:
            st.metric("Review cost", f"{float(p['review_cost']):.2f}")

        best, cal_df = calibrate_thresholds(adjusted_logs, p)

        if best is None:
            st.warning("Not enough labeled products to calibrate thresholds.")
        else:
            kc1, kc2, kc3, kc4 = st.columns(4)
            with kc1:
                st.metric("Best reject threshold", fmt_float(float(best["reject_threshold"]), 3))
            with kc2:
                st.metric("Best known threshold", fmt_float(float(best["known_threshold"]), 3))
            with kc3:
                st.metric("Expected cost", fmt_float(float(best["expected_cost"]), 2))
            with kc4:
                st.metric("Reviews", int(best["reviews"]))

            st.dataframe(cal_df, use_container_width=True)

            st.info(
                "To apply these thresholds, move the sidebar threshold sliders to the recommended values "
                "and press Regenerate demo."
            )

    # Drift & PLC
    with tabs[5]:
        st.subheader("Drift monitoring and PLC dry-run")

        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.metric("Brightness PSI", fmt_float(m["brightness_psi"], 3))
        with dc2:
            st.metric("Contrast PSI", fmt_float(m["contrast_psi"], 3))
        with dc3:
            st.metric("Blur PSI", fmt_float(m["blur_psi"], 3))
        with dc4:
            st.metric("Prediction PSI", fmt_float(m["prediction_psi"], 3))

        st.markdown("#### Histograms")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.write("Fused score")
            st.bar_chart(m["fused_hist"])
        with h2:
            st.write("Anomaly score")
            st.bar_chart(m["anomaly_hist"])
        with h3:
            st.write("Brightness")
            st.bar_chart(m["brightness_hist"])
        with h4:
            st.write("Latency")
            st.bar_chart(m["latency_hist"])

        st.divider()
        st.markdown("#### PLC event log")

        if not m["plc_df"].empty:
            st.dataframe(m["plc_df"], use_container_width=True, height=300)

            st.markdown("#### PLC signal timeline")
            st.line_chart(m["plc_df"], x="time_s", y="signal_num")
        else:
            st.info("No PLC events.")

    # Failure Lab
    with tabs[6]:
        st.subheader("Failure lab")

        failures = [l for l in adjusted_logs if bool(l.get("included", False)) and (bool(l["escaped"]) or bool(l["false_reject"]))]

        if not failures:
            st.success("No escapes or false rejects occurred in this adjusted sample.")
        else:
            categories = Counter(classify_failure(l, p) for l in failures)

            cat_rows = [
                {
                    "Category": k,
                    "Count": v,
                    "Targeted fix": FAILURE_FIXES.get(k, "Investigate manually."),
                }
                for k, v in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
            ]

            st.dataframe(cat_rows, use_container_width=True)

            st.markdown("#### Worst 12 failures")

            def failure_margin(l: Dict[str, Any]) -> float:
                if bool(l["escaped"]):
                    return float(p["reject_threshold"]) - float(l["peak_fused"])
                return float(l["peak_fused"]) - float(p["reject_threshold"])

            worst = sorted(failures, key=failure_margin, reverse=True)[:12]

            worst_rows = [
                {
                    "ID": int(l["id"]),
                    "Type": "Escape" if bool(l["escaped"]) else "False reject",
                    "Product": PRODUCT_LABELS.get(str(l["product_type"]), str(l["product_type"])),
                    "Ground truth": DEFECT_LABELS.get(str(l["gt"]), str(l["gt"])),
                    "Predicted": DEFECT_LABELS.get(str(l["pred_class"]), str(l["pred_class"])),
                    "Peak fused": round(float(l["peak_fused"]), 3),
                    "Peak known": round(float(l["peak_known"]), 3),
                    "Anomaly": round(float(l["anomaly_mean"]), 3),
                    "Align": round(float(l["align_mean"]), 3),
                    "Category": classify_failure(l, p),
                }
                for l in worst
            ]

            st.dataframe(worst_rows, use_container_width=True)

    # Architecture
    with tabs[7]:
        st.subheader("System architecture")

        st.markdown(
            """
            ```text
            Camera / frame grabber
                   |
                   v
            Ingestion & decoding
                   |
                   v
            Pre-processing + golden alignment
                   |
        -----------------------------
        |                           |
        v                           v
Known-defect scorer         Unsupervised tile anomaly scorer
(difference morphology)     (nearest-good feature bank)
        |                           |
        --------- score fusion ------
                  |
                  v
        Uncertainty + temporal smoothing
                  |
                  v
        Cost-aware threshold decision
                  |
                  v
        PLC dry-run reject trigger
                  |
                  v
        Observability + operator review UI
            """
        )

        st.markdown("#### Trade-off table")
        st.dataframe(pd.DataFrame(architecture_tradeoff_rows()), use_container_width=True)

        st.markdown(
            """
            Final choice: golden difference + morphology + tile anomaly bank.

            Why:
            - runs locally without GPU
            - supports known and unknown defect scoring
            - keeps latency low enough for interactive demo
            - easier to audit than a black-box detector
            - allows cost-based threshold calibration

            Limitation:
            - synthetic demo only
            - not a substitute for trained deep detectors on real production data
            - acceptance metrics are provisional until validated on frozen line data
            """
        )

    # Data & Risk
    with tabs[8]:
        st.subheader("Data readiness and risk register")

        st.markdown("#### Data readiness")
        st.dataframe(pd.DataFrame(data_readiness_rows()), use_container_width=True)

        st.markdown("#### Synthetic manifest")
        st.dataframe(make_synthetic_manifest(p, adjusted_logs), use_container_width=True)

        st.markdown(
            """
            Real data protocol:
            - write dataset card
            - split by scene/site/camera, never by frame
            - freeze test split before tuning
            - keep separate stress set
            - double-annotate 10%
            - remove near-duplicates with perceptual hashing
            - log per-class counts
            - version raw and derived data
            """
        )

        st.divider()
        st.markdown("#### Risk register")
        st.dataframe(pd.DataFrame(risk_register_rows()), use_container_width=True)

    # Rollout
    with tabs[9]:
        st.subheader("Staged rollout, runbook, security and ethics")

        st.markdown(
            """
            1. Shadow mode
               - Run inspection on live images but do not trigger PLC reject.
               - Entry: frozen test gates met provisionally, operator UI reviewed.
               - Exit: 7 days with no unexplained escape/false-reject spikes.
               - Rollback: disable shadow logging if latency or storage exceeds budget.

            2. Limited pilot
               - Enable automatic reject on one line or one station.
               - Entry: shadow-mode drift monitors stable, PLC dry-run tested.
               - Exit: escape and false-reject rates within target over production volume.
               - Rollback: switch PLC output to dry-run instantly.

            3. Production
               - Full automatic reject with manual review queue for low-confidence parts.
               - Entry: pilot exit criteria met and maintenance sign-off complete.
               - Exit: ongoing monitoring.
               - Rollback: revert to previous model/threshold tag and dry-run mode.
            """
        )

        st.markdown("#### Monitoring signals")
        st.markdown(
            """
            - Input drift: brightness/contrast/blur PSI. Alert when PSI > 0.20.
            - Prediction drift: per-hour reject rate and confidence histogram. Alert on > 30% deviation from 7-day baseline.
            - Latency: alert when p95 exceeds budget for 5 consecutive minutes.
            - Feedback loop: sample low-confidence outputs weekly for human review and retraining.
            - Override drift: alert if operator override rate exceeds baseline.
            """
        )

        st.markdown("#### One-page runbook")
        st.markdown(
            """
            - If escape alarm fires: freeze current model tag, pull last 100 rejected and passed parts, compare to golden, escalate to quality engineer.
            - If false-reject alarm fires: check lens contamination, lighting drift, dust, camera pose, and compression artifacts before changing thresholds.
            - If latency alarm fires: reduce batch size, switch to lighter preprocessing, or fallback to FP32/FP16 model variant depending on hardware.
            - If PLC communication fails: enter safe dry-run mode and alert operator. Do not silently pass unknown parts.
            - If operator override spike occurs: review reason codes, check for new defect family or fixture drift.
            """
        )

        st.markdown("#### Security, privacy, and ethics")
        st.markdown(
            """
            - Store only images needed for inspection and quality traceability.
            - Apply least-privilege access to image archives and PLC interfaces.
            - Provide an operator contest/correction path for automated reject decisions.
            - Document model limits: this system is not safe for human presence detection or safety-interlocked guarding.
            - Keep audit logs for model version, threshold, data version, PLC event, and operator override.
            """
        )

    # Tests
    with tabs[10]:
        st.subheader("Built-in and edge-case tests")

        t1, t2 = st.columns(2)

        with t1:
            if st.button("Run built-in tests", key="run_builtin_btn"):
                with st.spinner("Running built-in tests..."):
                    st.session_state.test_results = run_builtin_tests()

            test_results = st.session_state.get("test_results")
            if test_results:
                st.dataframe(test_results, use_container_width=True)
            else:
                st.info("Press Run built-in tests.")

        with t2:
            if st.button("Run edge-case tests", key="run_edge_btn"):
                with st.spinner("Running edge-case tests..."):
                    st.session_state.edge_results = run_edge_case_tests(p)

            edge_results = st.session_state.get("edge_results")
            if edge_results:
                st.dataframe(edge_results, use_container_width=True)
            else:
                st.info("Press Run edge-case tests.")

    # Report
    with tabs[11]:
        st.subheader("Exportable report")

        md = build_report_markdown(result, feedback)
        js = build_report_json(result, feedback)

        st.text_area("Markdown report preview", md, height=360)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.download_button(
                "Download Markdown report",
                data=md.encode("utf-8"),
                file_name="advanced_defect_inspection_report.md",
                mime="text/markdown",
            )
        with rc2:
            st.download_button(
                "Download JSON report",
                data=js.encode("utf-8"),
                file_name="advanced_defect_inspection_report.json",
                mime="application/json",
            )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main() -> None:
    p0 = default_params()

    st.title("Advanced Automatic Defect Inspection for Manufacturing")
    st.caption(
        "Local Streamlit demo: multi-product synthetic line, hybrid known + unknown defect detection, "
        "golden alignment, tile anomaly bank, temporal voting, cost calibration, PLC dry-run, operator review, "
        "drift monitoring, failure analysis, and exportable reports."
    )

    with st.sidebar:
        st.header("Control room")

        sim_tab, defect_tab, env_tab, model_tab, cost_tab = st.tabs(
            ["Line", "Defects", "Environment", "Model", "Cost / Ops"]
        )

        with sim_tab:
            seed = st.number_input("Random seed", value=int(p0["seed"]), step=1)
            duration = st.slider("Duration (seconds)", 4.0, 20.0, float(p0["duration"]), 0.5)
            fps = st.slider("FPS", 10, 30, int(p0["fps"]))
            line_speed = st.slider("Line speed (products/min)", 60, 600, int(p0["line_speed"]))

            st.subheader("Product mix")
            mix_plate = st.slider("Plate mix", 0.0, 1.0, float(p0["mix_plate"]), 0.05)
            mix_gear = st.slider("Gear mix", 0.0, 1.0, float(p0["mix_gear"]), 0.05)
            mix_bracket = st.slider("Bracket mix", 0.0, 1.0, float(p0["mix_bracket"]), 0.05)

        with defect_tab:
            st.subheader("Defect rates")
            scratch_rate = st.slider("Scratch", 0.0, 0.25, float(p0["scratch_rate"]), 0.01)
            crack_rate = st.slider("Crack", 0.0, 0.25, float(p0["crack_rate"]), 0.01)
            missing_rate = st.slider("Missing", 0.0, 0.25, float(p0["missing_rate"]), 0.01)
            dent_rate = st.slider("Dent", 0.0, 0.25, float(p0["dent_rate"]), 0.01)
            contamination_rate = st.slider("Contamination", 0.0, 0.25, float(p0["contamination_rate"]), 0.01)
            discoloration_rate = st.slider("Discoloration", 0.0, 0.25, float(p0["discoloration_rate"]), 0.01)
            foreign_object_rate = st.slider("Foreign object", 0.0, 0.25, float(p0["foreign_object_rate"]), 0.01)
            unknown_rate = st.slider("Unknown defect", 0.0, 0.25, float(p0["unknown_rate"]), 0.01)
            severity_bias = st.slider("Severity bias", 0.25, 1.0, float(p0["severity_bias"]), 0.01)

        with env_tab:
            st.subheader("Environment")
            lighting_drift = st.slider("Lighting drift", 0.0, 1.0, float(p0["lighting_drift"]), 0.01)
            dust = st.slider("Dust / lens contamination", 0.0, 1.0, float(p0["dust"]), 0.01)
            sensor_noise = st.slider("Sensor noise", 0.0, 1.0, float(p0["sensor_noise"]), 0.01)
            motion_blur = st.slider("Motion blur", 0.0, 1.0, float(p0["motion_blur"]), 0.01)
            compression = st.slider("Compression artifacts", 0.0, 1.0, float(p0["compression"]), 0.01)
            orientation_jitter_deg = st.slider("Orientation jitter (deg)", 0.0, 15.0, float(p0["orientation_jitter_deg"]), 0.1)

        with model_tab:
            st.subheader("Hybrid detector")
            sensitivity = st.slider("Sensitivity", 0.0, 1.0, float(p0["sensitivity"]), 0.01)
            reject_threshold = st.slider("Fused reject threshold", 0.05, 0.80, float(p0["reject_threshold"]), 0.01)
            known_threshold = st.slider("Known-defect threshold", 0.05, 0.90, float(p0["known_threshold"]), 0.01)
            review_band = st.slider("Review band", 0.01, 0.30, float(p0["review_band"]), 0.01)
            min_align_conf = st.slider("Min alignment confidence", 0.0, 0.95, float(p0["min_align_conf"]), 0.01)

            st.subheader("Fusion and temporal")
            w_known = st.slider("Known weight", 0.0, 1.0, float(p0["w_known"]), 0.01)
            w_anomaly = st.slider("Anomaly weight", 0.0, 1.0, float(p0["w_anomaly"]), 0.01)
            anomaly_top_k = st.slider("Anomaly top-k fraction", 0.02, 0.50, float(p0["anomaly_top_k"]), 0.01)
            temporal_ema = st.slider("Temporal EMA alpha", 0.05, 1.0, float(p0["temporal_ema"]), 0.01)
            vote_ratio = st.slider("Vote ratio", 0.10, 1.0, float(p0["vote_ratio"]), 0.01)
            bank_samples = st.slider("Good bank samples / product", 6, 40, int(p0["bank_samples"]))

        with cost_tab:
            st.subheader("Cost matrix")
            escape_cost = st.number_input("Escape cost", min_value=0.0, value=float(p0["escape_cost"]), step=10.0)
            false_reject_cost = st.number_input("False-reject cost", min_value=0.0, value=float(p0["false_reject_cost"]), step=1.0)
            review_cost = st.number_input("Review cost", min_value=0.0, value=float(p0["review_cost"]), step=0.5)

            st.subheader("Operations")
            latency_budget_ms = st.number_input("Latency budget (ms)", min_value=10.0, value=float(p0["latency_budget_ms"]), step=5.0)
            plc_mode = st.selectbox("PLC mode", ["Dry-run", "Shadow", "Pilot"], index=0)

        run_btn = st.button("Regenerate advanced demo", type="primary")

        st.caption(
            "Longer duration, higher FPS, and larger bank samples increase generation time. "
            "The app generates the advanced demo automatically on first launch."
        )

    params: Dict[str, Any] = {
        "seed": int(seed),
        "duration": float(duration),
        "fps": int(fps),
        "line_speed": int(line_speed),
        "mix_plate": float(mix_plate),
        "mix_gear": float(mix_gear),
        "mix_bracket": float(mix_bracket),
        "scratch_rate": float(scratch_rate),
        "crack_rate": float(crack_rate),
        "missing_rate": float(missing_rate),
        "dent_rate": float(dent_rate),
        "contamination_rate": float(contamination_rate),
        "discoloration_rate": float(discoloration_rate),
        "foreign_object_rate": float(foreign_object_rate),
        "unknown_rate": float(unknown_rate),
        "severity_bias": float(severity_bias),
        "lighting_drift": float(lighting_drift),
        "dust": float(dust),
        "sensor_noise": float(sensor_noise),
        "motion_blur": float(motion_blur),
        "compression": float(compression),
        "orientation_jitter_deg": float(orientation_jitter_deg),
        "sensitivity": float(sensitivity),
        "reject_threshold": float(reject_threshold),
        "known_threshold": float(known_threshold),
        "review_band": float(review_band),
        "min_align_conf": float(min_align_conf),
        "w_known": float(w_known),
        "w_anomaly": float(w_anomaly),
        "anomaly_top_k": float(anomaly_top_k),
        "temporal_ema": float(temporal_ema),
        "vote_ratio": float(vote_ratio),
        "bank_samples": int(bank_samples),
        "escape_cost": float(escape_cost),
        "false_reject_cost": float(false_reject_cost),
        "review_cost": float(review_cost),
        "latency_budget_ms": float(latency_budget_ms),
        "plc_mode": str(plc_mode),
        "gallery_limit": int(p0["gallery_limit"]),
    }

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    sig = json.dumps(params, sort_keys=True, default=str)

    if "result" not in st.session_state:
        with st.spinner("Generating advanced synthetic inspection video..."):
            try:
                st.session_state.result = run_simulation(params)
                st.session_state.result_sig = sig
                st.session_state.pop("startup_error", None)
            except Exception as exc:
                st.session_state.result = None
                st.session_state.startup_error = str(exc)

    elif run_btn:
        with st.spinner("Regenerating advanced demo..."):
            try:
                st.session_state.result = run_simulation(params)
                st.session_state.result_sig = sig
                st.session_state.pop("startup_error", None)
            except Exception as exc:
                st.session_state.result = None
                st.session_state.startup_error = str(exc)

    if st.session_state.get("startup_error"):
        st.error(f"Demo generation failed: {st.session_state.startup_error}")

    result = st.session_state.get("result")
    if result is None:
        st.stop()

    if sig != st.session_state.get("result_sig"):
        st.info("Controls changed. Press **Regenerate advanced demo** to apply them.")

    display_result(result)


if __name__ == "__main__":
    main()