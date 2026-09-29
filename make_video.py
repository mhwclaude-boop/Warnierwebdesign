#!/usr/bin/env python3
"""
make_video.py - "From chaos to governed AI": a 20 s camera move over scene.png

Every frame is rendered from ONE illustration by a virtual camera
(crop window + scale + roll) that follows the keyframes in the CONFIG block.

    python3 make_video.py --preview         # 960x540 check   -> output/ai_transformation_preview.mp4
    python3 make_video.py                   # final 1920x1080 -> output/ai_transformation.mp4
    python3 make_video.py --contact-sheet   # key frames      -> output/contact_sheet.jpg, output/legibility_sheet.jpg
    python3 make_video.py --stills 3.4 9.8  # single frames   -> output/stills/
    python3 make_video.py --check           # camera report (holds, speeds, bounds), no rendering

Needs numpy, opencv-python, pillow and ffmpeg/ffprobe on PATH.  The 4x master
is made once with Real-ESRGAN (pip install torch realesrgan) and cached in
build/; without it the script falls back to Lanczos 4x + a light unsharp mask.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
SCENE = "scene.png"
LOGO = "assets/deloitte_logo.png"          # used on the end card if present
FONT_DIR = "assets/fonts"
BUILD_DIR = "build"                         # cache: 4x master + model weights
OUT_FINAL = "output/ai_transformation.mp4"
OUT_PREVIEW = "output/ai_transformation_preview.mp4"

FPS = 30
DURATION = 20.0                             # -> exactly 600 frames
FINAL_SIZE = (1920, 1080)
PREVIEW_SIZE = (960, 540)
CRF = 15                                    # 14-16
PRESET = "slow"
PREVIEW_CRF, PREVIEW_PRESET = 20, "medium"

UPSCALE_WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                       "v0.1.0/RealESRGAN_x4plus.pth")


@dataclass
class Key:
    """Camera keyframe.

    cx, cy  camera centre in pixels of the ORIGINAL scene.png (1536x1024).
    zoom    1.0 = whole picture height fills the frame (full view, the sides
            are filled with a blurred extension); 3.0 = three times closer.
    rot     camera roll in degrees (+ = picture turns counter-clockwise).
    shake   handheld shake amplitude in pixels at 1080p (0 = steady-cam).
    The remaining fields describe the move that ARRIVES at this key:
    ease    cubic ease-in-out curves:
              "inout_cubic"   easeInOutCubic 4t^3 / 1-(2-2t)^3/2 - snappy, peak 3x mean speed
              "hermite_cubic" 3t^2-2t^3 (cubic Hermite)            - gliding, peak 1.5x
            also inout_sine, linear, in_cubic, out_cubic
    path    "pan"   centre and log-zoom interpolate together; arc > 0 pulls
                    the camera back mid-move (log-zoom units, 0.5 ~ 40 %).
                    Pan moves between two in-picture framings are kept
                    inside the picture.
            "pivot" pure zoom into / out of a fixed point
    """
    t: float
    cx: float
    cy: float
    zoom: float
    rot: float = 0.0
    shake: float = 0.0
    ease: str = "inout_cubic"
    path: str = "pan"
    arc: float = 0.0


K = Key
HOLD = "inout_sine"          # near-still drift while a text is on screen
GLIDE = "hermite_cubic"      # steady-cam moves on the organized side
CAMERA = [
    # ── 0.0-1.5 FULL VIEW: whole scene, very slow push-in ─────────────────
    K(0.00, 768, 512, 1.000),
    K(1.50, 768, 512, 1.045, ease="linear"),
    # ── 1.5-3.0 ZOOM TO CHAOS: dive into the far-left side ────────────────
    K(3.00, 272, 440, 3.50, rot=-0.8, shake=1.5, path="pivot"),
    # ── 3.0-8.0 CHAOS POV (handheld): desk -> wall -> alarm rack ─────────
    K(3.82, 267, 443, 3.60, rot=-1.3, shake=2.0, ease=HOLD),     # "New AI capability available"
    K(4.55, 373, 213, 2.48, rot=0.1, shake=2.2, arc=0.12),       # look up at the wall
    K(5.55, 372, 214, 2.48, rot=0.2, shake=2.4, ease=HOLD),      # CHANGE MGMT / STRATEGY?? / GAP METER / EXCUSES
    K(6.70, 290, 712, 3.30, rot=-2.4, shake=4.0, arc=0.25),      # alarm! faster, stressed turn
    K(8.00, 286, 713, 3.55, rot=-3.0, shake=6.0, ease=HOLD),     # SECURITY POLICY VIOLATION ... (creeping push-in)
    # ── 8.0-10.0 TRANSITION: swish pan across the chasm to the bridge ────
    K(9.05, 992, 640, 2.70, arc=0.55),
    K(10.00, 998, 642, 2.78, ease=HOLD),                          # AI PLATFORM FUTURE READY + FRONT RUNNER
    # ── 10.0-18.0 ORGANIZED POV (steady-cam, no shake) ───────────────────
    K(10.95, 1340, 359, 4.70, arc=0.50, ease=GLIDE),              # AI WORKFLOW + team with VALUE
    K(12.00, 1341, 360, 4.78, ease=HOLD),
    K(12.70, 1322, 560, 4.40, arc=0.35, ease=GLIDE),              # dashboards
    K(13.75, 1324, 557, 4.50, ease=HOLD),
    K(14.75, 1190, 155, 3.35, arc=0.55, ease=GLIDE),              # THE FUTURE doorway
    K(15.80, 1186, 148, 3.50, ease=HOLD),                         # walk towards the light
    K(16.90, 1208, 818, 2.80, arc=0.70, ease=GLIDE),              # WE SHAPE OUR OWN FUTURE
    K(18.00, 1210, 815, 2.95, ease=HOLD),
    # ── 18.0-20.0 OUTRO: pull back to the full image ─────────────────────
    K(19.25, 768, 512, 1.000, path="pivot"),
    K(20.00, 768, 512, 0.985, ease="linear"),
]

# Effect tracks: (time, value) points, smoothly interpolated in between.
GRADE = [(0, 0), (1.9, 0), (3.0, -1), (8.0, -1), (9.8, 1), (18.2, 1), (19.3, 0.25)]  # -1 warm/dull, +1 cool/bright
VIGNETTE = [(0, .12), (1.9, .12), (3.0, .55), (8.0, .62), (9.8, .16), (18.2, .16), (19.3, .30)]
DOF = [(0, 0), (2.5, 0), (3.2, 7.0), (7.9, 8.0), (8.6, 0)]          # edge blur sigma, px @1080p
BLOOM = [(0, 0), (8.6, 0), (9.9, .55), (18.0, .55), (19.3, .15)]    # screen glow strength
PARALLAX = [(0, 0), (8.0, 0), (8.4, .6), (9.8, 1.0), (18.0, 1.0), (18.4, 0)]
ALARM = [(0, 0), (6.3, 0), (6.8, 1.0), (8.1, 1.0), (8.7, 0)]         # pulsing red beacon on the rack
DIM = [(0, 0), (18.55, 0), (19.2, 1.0)]                              # outro dim (x DIM_AMOUNT)
DIM_AMOUNT = 0.38

# Grade presets reached at GRADE = -1 (chaos) and +1 (order).
GRADE_CHAOS = dict(wb=(1.07, 1.00, 0.84), gain=0.95, sat=0.66, contrast=1.08, lift=(0.030, 0.014, 0.0))
GRADE_ORDER = dict(wb=(0.95, 1.00, 1.06), gain=1.07, sat=1.10, contrast=1.04, lift=(0.0, 0.006, 0.020))

SHAKE_ROT_PER_PX = 0.07          # degrees of roll jitter per px of shake
PARALLAX_K = 0.035               # s   - lateral parallax (camera velocity x depth)
PARALLAX_KZ = 0.12               # zoom parallax (near layers expand faster)
SHUTTER = 0.5                    # motion blur shutter (0.5 = 180 degrees)
MB_STEP_PX, MB_MAX = 3.0, 48     # sub-frame spacing (px @1080p) and cap
ALARM_XY = (175, 597)            # red beacon on the server rack (scene px)

# Overlays (crisp vector text, rendered at output resolution)
LOWER_THIRDS = [   # (text, fade-in start, fade-out start)
    ("Without a clear data foundation, AI stays fragmented.", 3.25, 7.60),
    ("With clear data as the base, AI becomes governed, scalable and valuable.", 10.45, 17.45),
]
END_CARD_TEXT = "Deloitte helps organizations build the data and AI foundation to lead the future."
END_CARD_IN = 18.60
FADE = 0.4
ACCENT = (0x86, 0xBC, 0x25)
PANEL_RGB, PANEL_ALPHA = (10, 14, 20), 0.74
FONT_UI = ["OpenSans-Semibold.ttf", "OpenSans-SemiBold.ttf", "Inter-SemiBold.otf", "Inter-SemiBold.ttf", "Arial.ttf"]
FONT_UI_REGULAR = ["OpenSans-Regular.ttf", "Inter-Regular.otf", "Inter-Regular.ttf", "Arial.ttf"]

# Lettering that is still unreadable after upscaling is re-typeset as vector
# text, matched in position, perspective and font feel, and painted into the
# 4x master (so it moves, blurs and grades with the picture).
#   rack      the source misspells "TOKEN LIMIT EXCEEDED!" with broken letters;
#             all three red panels are redone so they match (quad = panel
#             corners TL, TR, BR, BL in scene px).
#   workflow  "UNDERSTAND" reads as "UNDERSTAHO" in the source; the five step
#             labels are redone together (box = centre x, centre y, text width,
#             cap height, baseline angle in degrees).
TEXT_STYLES = {
    "rack": dict(font=["Anton-Regular.ttf"], ink="light", fill=(252, 247, 238), shadow=(95, 12, 6, 0.45)),
    "workflow": dict(font=["OpenSans-Bold.ttf", "Inter-Bold.otf"], ink="dark", fill=(14, 36, 112),
                     glow=(236, 246, 255, 0.55)),
}
TEXT_FIXES = [
    dict(style="rack", quad=[(224, 645), (272, 621), (273, 667), (229, 695)], lines=["SECURITY", "POLICY", "VIOLATION!"]),
    dict(style="rack", quad=[(228, 715), (274, 687), (272, 721), (230, 747)], lines=["TOKEN LIMIT", "EXCEEDED!"]),
    dict(style="rack", quad=[(231, 765), (274, 737), (274, 769), (232, 795)], lines=["ACCESS", "DENIED!"]),
    dict(style="workflow", box=(1261.0, 309.7, 25.0, 5.9, 1.0), lines=["INGEST"]),
    dict(style="workflow", box=(1311.0, 310.8, 45.0, 5.8, 1.8), lines=["UNDERSTAND"]),
    dict(style="workflow", box=(1361.3, 312.7, 32.7, 5.8, 4.0), lines=["ANALYZE"]),
    dict(style="workflow", box=(1411.8, 315.7, 35.0, 5.9, 4.9), lines=["INSIGHTS"]),
    dict(style="workflow", box=(1468.0, 322.0, 27.5, 5.9, 8.4), lines=["ACTION"]),
]

# Key text regions (scene px): kept sharp by the DOF blur and checked by --check.
KEY_TEXTS = {
    "New AI capability available": (100, 436, 182, 496),
    "CHANGE MANAGEMENT IS NOT OPTIONAL": (20, 20, 124, 126),
    "STRATEGY??": (258, 176, 334, 196),
    "THE GAP METER": (468, 12, 610, 114),
    "TODAY'S EXCUSES": (626, 10, 728, 140),
    "SECURITY / TOKEN LIMIT / ACCESS DENIED": (222, 618, 276, 797),
    "AI PLATFORM FUTURE READY": (943, 478, 1051, 568),
    "FRONT RUNNER": (932, 688, 1016, 777),
    "AI WORKFLOW (Ingest > ... > Action)": (1240, 258, 1490, 330),
    "VALUE": (1280, 366, 1346, 423),
    "SECURITY / COMPLIANCE / TOKEN USAGE / MODEL HEALTH": (1168, 484, 1522, 594),
    "THE FUTURE": (1092, 8, 1284, 40),
    "WE SHAPE OUR OWN FUTURE": (1150, 772, 1420, 852),
}

# Frames for the contact sheet (time, label)
CONTACT_FRAMES = [
    (0.75, "Full view"), (2.30, "Zoom to chaos"), (3.45, "Desk / New AI capability"),
    (4.25, "Chaos: look up"), (5.25, "Wall: CM / Strategy / Gap / Excuses"), (6.25, "Alarm turn"),
    (7.40, "Server rack alarm"), (8.62, "Swish across the chasm"), (9.80, "AI Platform + Front Runner"),
    (11.55, "AI Workflow + Value"), (13.00, "Dashboards"), (14.90, "The Future"),
    (15.95, "Crane down"), (17.35, "We shape our own future"), (18.60, "Pull back"), (19.85, "End card"),
]

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def ease(name: str, s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    if name == "inout_cubic":
        return 4 * s ** 3 if s < 0.5 else 1 - (-2 * s + 2) ** 3 / 2
    if name == "hermite_cubic":
        return s * s * (3 - 2 * s)
    if name == "inout_sine":
        return -(math.cos(math.pi * s) - 1) / 2
    if name == "linear":
        return s
    if name == "in_cubic":
        return s ** 3
    if name == "out_cubic":
        return 1 - (1 - s) ** 3
    raise ValueError(f"unknown ease {name!r}")


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def track(points, t: float) -> float:
    if t <= points[0][0]:
        return float(points[0][1])
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            s = s * s * (3 - 2 * s)
            return float(v0 + (v1 - v0) * s)
    return float(points[-1][1])


def find_font(names):
    dirs = [FONT_DIR, "/usr/share/fonts/truetype/open-sans", "/usr/share/fonts/opentype/inter",
            "/usr/share/fonts/truetype/dejavu", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
            "/System/Library/Fonts/Supplemental", "C:/Windows/Fonts"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"none of the fonts {names} found (put them in {FONT_DIR}/)")


# ── camera ────────────────────────────────────────────────────────────────
@dataclass
class Cam:
    cx: float
    cy: float
    zoom: float
    rot: float
    shake: float


SCENE_W, SCENE_H = 1536, 1024      # updated from scene.png at start-up
ASPECT = 16 / 9


def _half_view(zoom):
    return SCENE_H * ASPECT / zoom / 2, SCENE_H / zoom / 2


def _inside(k: Key) -> bool:
    hw, hh = _half_view(k.zoom)
    return (k.cx - hw >= -0.5 and k.cx + hw <= SCENE_W + 0.5 and
            k.cy - hh >= -0.5 and k.cy + hh <= SCENE_H + 0.5)


def _soft_clamp(x, lo, hi, k=2.0):
    """Smooth clamp: identity well inside [lo, hi], saturates softly at the ends."""
    if lo > hi:
        return (lo + hi) / 2
    sp = lambda z: k * math.log1p(math.exp(min(z / k, 50))) if z / k < 50 else z   # softplus
    return lo + sp(x - lo) - sp(x - hi)


def _keep_inside(cx, cy, zoom):
    hw, hh = _half_view(zoom)
    return _soft_clamp(cx, hw, SCENE_W - hw), _soft_clamp(cy, hh, SCENE_H - hh)


def cam_path(t: float) -> Cam:
    """Keyframed camera at time t (no shake applied)."""
    keys = CAMERA
    if t <= keys[0].t:
        k = keys[0]
        return Cam(k.cx, k.cy, k.zoom, k.rot, k.shake)
    for k0, k1 in zip(keys, keys[1:]):
        if t <= k1.t:
            e = ease(k1.ease, (t - k0.t) / (k1.t - k0.t))
            lz0, lz1 = math.log(k0.zoom), math.log(k1.zoom)
            lz = lz0 + (lz1 - lz0) * e
            if k1.path == "pivot":
                # the view rectangle scales about a fixed point: its edges move
                # linearly with the width, so the picture never slides sideways
                w0, w1, w = 1 / k0.zoom, 1 / k1.zoom, math.exp(-lz)
                f = e if abs(w0 - w1) < 1e-9 else (w0 - w) / (w0 - w1)
            else:
                f = e
                lz -= k1.arc * math.sin(math.pi * e) ** 2
            cx, cy, z = k0.cx + (k1.cx - k0.cx) * f, k0.cy + (k1.cy - k0.cy) * f, math.exp(lz)
            if k1.path == "pan" and _inside(k0) and _inside(k1):
                cx, cy = _keep_inside(cx, cy, z)
            return Cam(cx, cy, z, k0.rot + (k1.rot - k0.rot) * e, k0.shake + (k1.shake - k0.shake) * e)
    k = keys[-1]
    return Cam(k.cx, k.cy, k.zoom, k.rot, k.shake)


_rng = np.random.default_rng(7)
SHAKE_FREQ = np.array([0.31, 0.73, 1.37, 2.61, 4.9])
SHAKE_W = np.array([1.0, 0.62, 0.34, 0.16, 0.07])
SHAKE_PH = _rng.uniform(0, 2 * np.pi, (3, len(SHAKE_FREQ)))


def shake_noise(t: float):
    """Smooth pseudo-random handheld motion in [-1, 1] for x, y, roll."""
    v = (SHAKE_W * np.sin(2 * np.pi * SHAKE_FREQ * t + SHAKE_PH)).sum(axis=1) / SHAKE_W.sum()
    return float(v[0]), float(v[1]), float(v[2])


def cam_full(t: float, src_h: float) -> Cam:
    """Keyframed camera plus handheld shake."""
    c = cam_path(t)
    if c.shake <= 0:
        return c
    nx, ny, nr = shake_noise(t)
    s = 1080.0 * c.zoom / src_h                        # 1080p px per scene px
    rot = c.rot + nr * c.shake * SHAKE_ROT_PER_PX
    a = math.radians(rot)
    dx, dy = nx * c.shake / s, ny * c.shake / s
    return Cam(c.cx + math.cos(a) * dx - math.sin(a) * dy, c.cy + math.sin(a) * dx + math.cos(a) * dy,
               c.zoom, rot, c.shake)


def to_screen(c: Cam, px, py, W, H, src_h):
    """Scene px -> output px for camera c."""
    s = H * c.zoom / src_h
    a = math.radians(c.rot)
    dx, dy = np.asarray(px) - c.cx, np.asarray(py) - c.cy
    return (math.cos(a) * dx + math.sin(a) * dy) * s + W / 2, (-math.sin(a) * dx + math.cos(a) * dy) * s + H / 2


def to_scene(c: Cam, u, v, W, H, src_h):
    s = H * c.zoom / src_h
    a = math.radians(c.rot)
    du, dv = (np.asarray(u) - W / 2) / s, (np.asarray(v) - H / 2) / s
    return c.cx + math.cos(a) * du - math.sin(a) * dv, c.cy + math.sin(a) * du + math.cos(a) * dv


# ══════════════════════════════════════════════════════════════════════════
#  4x master: Real-ESRGAN (pip) -> Lanczos 4x + unsharp mask fallback
# ══════════════════════════════════════════════════════════════════════════
def _realesrgan_x4(src_rgb):
    try:
        importlib.import_module("torchvision.transforms.functional_tensor")
    except ImportError:   # basicsr still imports this pre-0.17 torchvision module
        import torchvision.transforms.functional as tvf
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = tvf.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    weights = os.path.join(BUILD_DIR, "weights", "RealESRGAN_x4plus.pth")
    if not os.path.exists(weights):
        os.makedirs(os.path.dirname(weights), exist_ok=True)
        print(f"[upscale] downloading {UPSCALE_WEIGHTS_URL}")
        urllib.request.urlretrieve(UPSCALE_WEIGHTS_URL, weights + ".part")
        os.replace(weights + ".part", weights)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    up = RealESRGANer(scale=4, model_path=weights, model=net, tile=384, tile_pad=24, pre_pad=0,
                      half=dev.type == "cuda", device=dev)
    out, _ = up.enhance(cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR), outscale=4)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _lanczos_x4(src_rgb):
    up = cv2.resize(src_rgb, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    blur = cv2.GaussianBlur(up, (0, 0), 1.6)
    return np.clip(up + 0.6 * (up - blur), 0, 255).astype(np.uint8)


def load_scene():
    bgr = cv2.imread(SCENE, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read {SCENE}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ensure_master(upscaler="auto", force=False):
    """Create (or reuse) the cached 4x master. Returns (path, method)."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    digest = hashlib.sha256(open(SCENE, "rb").read()).hexdigest()
    meta_path = os.path.join(BUILD_DIR, "scene_x4.json")
    cands = {"realesrgan": os.path.join(BUILD_DIR, "scene_x4_realesrgan.png"),
             "lanczos": os.path.join(BUILD_DIR, "scene_x4_lanczos.png")}
    if not force and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("sha256") == digest and os.path.exists(meta["path"]) and upscaler in ("auto", meta["kind"]):
            return meta["path"], meta["method"]
    if not force and not os.path.exists(meta_path):
        for kind in (["realesrgan", "lanczos"] if upscaler == "auto" else [upscaler]):
            if os.path.exists(cands[kind]):   # adopt a master made earlier by hand
                meta = dict(sha256=digest, kind=kind, path=cands[kind],
                            method="Real-ESRGAN x4plus" if kind == "realesrgan" else "Lanczos 4x + unsharp mask")
                json.dump(meta, open(meta_path, "w"), indent=1)
                return meta["path"], meta["method"]
    src = load_scene()
    t0 = time.time()
    kind = "lanczos"
    if upscaler in ("auto", "realesrgan"):
        try:
            print("[upscale] Real-ESRGAN x4plus (one-time, a few minutes on CPU)...")
            master, kind, method = _realesrgan_x4(src), "realesrgan", "Real-ESRGAN x4plus"
        except Exception as e:  # noqa: BLE001 - any failure means "not available"
            if upscaler == "realesrgan":
                raise
            print(f"[upscale] Real-ESRGAN unavailable ({type(e).__name__}: {e}); using Lanczos 4x + unsharp mask")
    if kind == "lanczos":
        master, method = _lanczos_x4(src), "Lanczos 4x + unsharp mask"
    cv2.imwrite(cands[kind], cv2.cvtColor(master, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    json.dump(dict(sha256=digest, kind=kind, path=cands[kind], method=method), open(meta_path, "w"), indent=1)
    print(f"[upscale] {method}: {master.shape[1]}x{master.shape[0]} in {time.time() - t0:.0f}s")
    return cands[kind], method


# ── re-typeset unreadable signs ──────────────────────────────────────────
def _fit_font(path, lines, w, h, gap=0.30):
    size = int(h)
    while size > 6:
        f = ImageFont.truetype(path, size)
        cap = f.getbbox("H")[3] - f.getbbox("H")[1]
        width = max(f.getlength(l) for l in lines)
        if width <= w and cap * len(lines) + cap * gap * (len(lines) - 1) <= h:
            return f, cap
        size -= 1
    raise ValueError("text does not fit")


def _hand_lettering(lines, W, H, st, seed, ss=3):
    """White hand-lettered text (RGBA float, W x H) fitted into a flat sign."""
    font_path = find_font(st["font"])
    rng = np.random.default_rng(seed)
    Ws, Hs = W * ss, H * ss
    font, cap = _fit_font(font_path, lines, Ws * 0.84, Hs * 0.78)
    gap = cap * 0.30
    layer = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
    y = (Hs - (cap * len(lines) + gap * (len(lines) - 1))) / 2
    top = font.getbbox("H")[1]
    sr, sg, sb, sa = st["shadow"]
    for line in lines:
        x = (Ws - font.getlength(line)) / 2
        for i, ch in enumerate(line):
            if ch == " ":
                continue
            cx = x + font.getlength(line[:i])
            g = Image.new("RGBA", (int(cap * 2.2), int(cap * 2.2)), (0, 0, 0, 0))
            ImageDraw.Draw(g).text((cap * 0.35, cap * 0.35 - top), ch, font=font, fill=tuple(st["fill"]) + (255,))
            g = g.rotate(rng.uniform(-2.0, 2.0), resample=Image.BICUBIC, center=(cap * 0.6, cap * 0.85))
            pos = (max(0, int(round(cx - cap * 0.35))),
                   max(0, int(round(y - cap * 0.35 + rng.uniform(-0.04, 0.04) * cap))))
            layer.alpha_composite(g, pos)
            sh = Image.new("RGBA", g.size, (sr, sg, sb, 0))
            sh.putalpha(g.getchannel("A").point(lambda a: int(a * sa)))
            shadow.alpha_composite(sh, (pos[0] + int(cap * 0.05), pos[1] + int(cap * 0.07)))
        y += cap + gap
    arr = np.asarray(Image.alpha_composite(shadow, layer)).astype(np.float32)
    # slight ink wobble so the letters sit in the hand-drawn illustration
    nx = cv2.GaussianBlur(rng.standard_normal((Hs, Ws)).astype(np.float32), (0, 0), 7 * ss)
    ny = cv2.GaussianBlur(rng.standard_normal((Hs, Ws)).astype(np.float32), (0, 0), 7 * ss)
    nx *= 0.9 * ss / (nx.std() + 1e-6)
    ny *= 0.9 * ss / (ny.std() + 1e-6)
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    arr = cv2.remap(arr, gx + nx, gy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)


def _label_text(text, W, H, bw, bh, st, ss=4):
    """Single-line label (RGBA float, W x H): cap height bh, width <= bw, centred, soft glow."""
    font_path = find_font(st["font"])
    ref = ImageFont.truetype(font_path, 100)
    cap100 = ref.getbbox("H")[3] - ref.getbbox("H")[1]
    font = ImageFont.truetype(font_path, max(4, int(round(100 * bh * ss / cap100))))
    top = font.getbbox("H")[1]
    cap = font.getbbox("H")[3] - top
    tw = font.getlength(text)
    m = int(cap)
    tl = Image.new("L", (int(tw) + 2 * m, int(cap) + 2 * m), 0)
    ImageDraw.Draw(tl).text((m, m - top), text, font=font, fill=255)
    arr = np.asarray(tl, np.float32) / 255
    squeeze = min(1.0, bw * ss / tw)                      # condense only, never stretch
    arr = cv2.resize(arr, (max(1, int(round(arr.shape[1] * squeeze))), arr.shape[0]), interpolation=cv2.INTER_AREA)
    Ws, Hs = W * ss, H * ss
    mask = np.zeros((Hs, Ws), np.float32)
    x0 = int(round((Ws - arr.shape[1]) / 2))
    y0 = int(round((Hs - cap) / 2 - m))
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + arr.shape[1], Ws), min(y0 + arr.shape[0], Hs)
    mask[ya:yb, xa:xb] = arr[ya - y0:yb - y0, xa - x0:xb - x0]
    gr, gg, gb, ga = st["glow"]
    glow = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.35 * bh * ss) * 2.2, 0, 1) * ga
    rgb = np.empty((Hs, Ws, 3), np.float32)
    rgb[:] = (gr, gg, gb)
    a = glow + mask * (1 - glow)
    rgb = (np.float32((gr, gg, gb)) * glow[..., None] * (1 - mask[..., None])
           + np.float32(st["fill"]) * mask[..., None]) / np.maximum(a, 1e-6)[..., None]
    arr = np.dstack([rgb, a * 255]).astype(np.float32)
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)


def _fix_quad(fix):
    if "quad" in fix:
        return np.float32(fix["quad"]), None
    cx, cy, w, h, ang = fix["box"]
    a = math.radians(ang)
    u = np.float32([math.cos(a), math.sin(a)])
    n = np.float32([-math.sin(a), math.cos(a)])
    hw, hh = w / 2 + 0.55 * h, h / 2 + 0.40 * h
    c = np.float32([cx, cy])
    return np.float32([c - hw * u - hh * n, c + hw * u - hh * n, c + hw * u + hh * n, c - hw * u + hh * n]), (w, h, hw, hh)


def apply_text_fixes(master, density):
    for n, fix in enumerate(TEXT_FIXES):
        st = TEXT_STYLES[fix["style"]]
        quad, box = _fix_quad(fix)
        q = (quad + 0.5) * density - 0.5
        W = int(round(np.linalg.norm(q[1] - q[0])))
        H = int(round(np.linalg.norm(q[3] - q[0])))
        flat_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        Hm = cv2.getPerspectiveTransform(flat_pts, q)
        flat = cv2.warpPerspective(master, Hm, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)
        hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV)
        if st["ink"] == "light":
            letters = ((hsv[..., 1] < 115) & (hsv[..., 2] > 115)).astype(np.uint8) * 255
            letters = cv2.dilate(letters, np.ones((7, 7), np.uint8))
        else:
            letters = ((hsv[..., 2] < 170) & (hsv[..., 1] > 60)).astype(np.uint8) * 255
            letters = cv2.dilate(letters, np.ones((5, 5), np.uint8))
        bg = cv2.inpaint(flat, letters, 7, cv2.INPAINT_TELEA)
        soft = cv2.GaussianBlur(bg, (0, 0), 4)
        m = cv2.GaussianBlur(letters.astype(np.float32) / 255, (0, 0), 3)[..., None]
        bg = bg * (1 - m) + soft * m
        if box is None:
            txt = _hand_lettering(fix["lines"], W, H, st, seed=11 + n)
        else:
            w, h, hw, hh = box
            txt = _label_text(fix["lines"][0], W, H, w * density, h * density, st)
        a = txt[..., 3:4] / 255.0
        new_flat = np.clip(bg * (1 - a) + txt[..., :3] * a, 0, 255).astype(np.uint8)
        # paste back through the same perspective, inside the quad only
        x0, y0 = np.floor(q.min(0)).astype(int) - 2
        x1, y1 = np.ceil(q.max(0)).astype(int) + 3
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ Hm
        patch = cv2.warpPerspective(new_flat, T, (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        inner = q + (q.mean(0) - q) * np.float32(0.035 if box is None else 0.02)
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillPoly(mask, [np.round((inner - [x0, y0]) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2 if box is None else 2.5)[..., None]
        roi = master[y0:y1, x0:x1].astype(np.float32)
        master[y0:y1, x0:x1] = np.clip(roi * (1 - mask) + patch * mask, 0, 255).astype(np.uint8)
    return master


# ══════════════════════════════════════════════════════════════════════════
#  Overlays
# ══════════════════════════════════════════════════════════════════════════
def _rounded_panel(size, radius, rgb, alpha):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=rgb + (int(255 * alpha),))
    return im


def build_lower_third(text, W, H):
    k = H / 1080
    font = ImageFont.truetype(find_font(FONT_UI), round(40 * k))
    maxw = W - 2 * round(120 * k) - round(76 * k)
    lines = [text]
    if font.getlength(text) > maxw:            # wrap into two balanced lines
        words = text.split()
        best = min(range(1, len(words)), key=lambda i: abs(font.getlength(" ".join(words[:i])) -
                                                           font.getlength(" ".join(words[i:]))))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    asc, desc = font.getmetrics()
    lh = round((asc + desc) * 1.02)
    pad_x, pad_y, acc = round(34 * k), round(20 * k), round(8 * k)
    tw = max(font.getlength(l) for l in lines)
    bw, bh = int(acc + 2 * pad_x + tw), int(2 * pad_y + lh * len(lines))
    x0, y0 = round(120 * k), H - round(92 * k) - bh
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(_rounded_panel((bw, bh), round(6 * k), PANEL_RGB, PANEL_ALPHA), (x0, y0))
    d = ImageDraw.Draw(layer)
    d.rectangle([x0, y0, x0 + acc - 1, y0 + bh - 1], fill=ACCENT + (255,))
    for i, l in enumerate(lines):
        d.text((x0 + acc + pad_x, y0 + pad_y + i * lh), l, font=font, fill=(255, 255, 255, 255))
    return np.asarray(layer)


def _balanced_lines(text, font, maxw):
    words = text.split()
    if font.getlength(text) <= maxw:
        return [text]
    best = min(range(1, len(words)), key=lambda i: max(font.getlength(" ".join(words[:i])),
                                                       font.getlength(" ".join(words[i:]))))
    return [" ".join(words[:best]), " ".join(words[best:])]


def build_end_card(W, H):
    k = H / 1080
    font = ImageFont.truetype(find_font(FONT_UI), round(52 * k))
    small = ImageFont.truetype(find_font(FONT_UI), round(20 * k))
    tiny = ImageFont.truetype(find_font(FONT_UI_REGULAR), round(15 * k))
    pw = round(1320 * k)
    lines = _balanced_lines(END_CARD_TEXT, font, pw - round(120 * k))
    asc, desc = font.getmetrics()
    lh = round((asc + desc) * 1.12)
    logo_h, gap1, rule_h, gap2 = round(92 * k), round(30 * k), round(6 * k), round(34 * k)
    pad = round(58 * k)
    ph = pad + logo_h + gap1 + rule_h + gap2 + lh * len(lines) + pad - round(10 * k)
    x0, y0 = (W - pw) // 2, (H - ph) // 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(_rounded_panel((pw, ph), round(12 * k), PANEL_RGB, PANEL_ALPHA + 0.04), (x0, y0))
    d = ImageDraw.Draw(layer)
    y = y0 + pad
    if os.path.exists(LOGO):
        logo = Image.open(LOGO).convert("RGBA")
        sc = min(logo_h / logo.height, round(420 * k) / logo.width)
        logo = logo.resize((max(1, round(logo.width * sc)), max(1, round(logo.height * sc))), Image.LANCZOS)
        layer.alpha_composite(logo, ((W - logo.width) // 2, y + (logo_h - logo.height) // 2))
    else:   # clearly marked placeholder - no invented logo
        bw, bh = round(380 * k), logo_h
        bx, by = (W - bw) // 2, y
        col = (201, 207, 214, 255)
        dash, gapd, lw = round(12 * k), round(8 * k), max(1, round(2 * k))
        for xx in range(bx, bx + bw, dash + gapd):
            d.line([xx, by, min(xx + dash, bx + bw), by], fill=col, width=lw)
            d.line([xx, by + bh - 1, min(xx + dash, bx + bw), by + bh - 1], fill=col, width=lw)
        for yy in range(by, by + bh, dash + gapd):
            d.line([bx, yy, bx, min(yy + dash, by + bh)], fill=col, width=lw)
            d.line([bx + bw - 1, yy, bx + bw - 1, min(yy + dash, by + bh)], fill=col, width=lw)
        t1, t2 = "LOGO PLACEHOLDER", "add assets/deloitte_logo.png"
        d.text((W / 2, by + bh * 0.40), t1, font=small, fill=col, anchor="mm")
        d.text((W / 2, by + bh * 0.72), t2, font=tiny, fill=(150, 160, 170, 255), anchor="mm")
    y += logo_h + gap1
    rw = round(110 * k)
    d.rectangle([(W - rw) // 2, y, (W + rw) // 2 - 1, y + rule_h - 1], fill=ACCENT + (255,))
    y += rule_h + gap2
    for i, l in enumerate(lines):
        d.text((W / 2, y + i * lh), l, font=font, fill=(255, 255, 255, 255), anchor="ma")
    return np.asarray(layer)


def overlay_state(t, t_in, t_out=None):
    """(opacity, slide in px@1080p) for an overlay fading in at t_in and out at t_out."""
    a = min(max((t - t_in) / FADE, 0.0), 1.0)
    slide = (1 - ease("out_cubic", a)) * 16
    if t_out is not None:
        a *= 1 - min(max((t - t_out) / FADE, 0.0), 1.0)
    return a, slide


# ══════════════════════════════════════════════════════════════════════════
#  Renderer
# ══════════════════════════════════════════════════════════════════════════
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


class Renderer:
    def __init__(self, size, master_path):
        self.W, self.H = size
        src = load_scene()
        self.src_h, self.src_w = src.shape[:2]
        master = cv2.cvtColor(cv2.imread(master_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dens0 = master.shape[1] / self.src_w
        master = apply_text_fixes(master, dens0)
        self.levels, self.dens = [master], [dens0]
        while self.dens[-1] > 0.6:
            lv = self.levels[-1]
            self.levels.append(cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA))
            self.dens.append(self.dens[-1] / 2)
        self._build_background()
        self._build_glow()
        self.depth_d = 0.25
        dh, dw = int(self.src_h * self.depth_d), int(self.src_w * self.depth_d)
        yy = (np.arange(dh, dtype=np.float32) + 0.5) / dh
        self.depth = np.repeat((yy ** 1.15)[:, None], dw, axis=1)          # 0 far (top) .. 1 near (bottom)
        # per-output-size tables
        W, H = self.W, self.H
        uu = (np.arange(W, dtype=np.float32) + 0.5 - W / 2) / (W / 2)
        vv = (np.arange(H, dtype=np.float32) + 0.5 - H / 2) / (H / 2)
        self.r_ell = np.sqrt(uu[None, :] ** 2 + vv[:, None] ** 2) / math.sqrt(2)   # corners = 1
        self.vig_shape = smoothstep(0.42, 1.12, self.r_ell) ** 1.35
        self.dof_shape = smoothstep(0.50, 1.00, self.r_ell)
        # parallax fades out towards the frame border so the border never samples outside the picture
        self.edge_win = np.minimum.outer(smoothstep(0, 0.2, np.minimum(vv + 1, 1 - vv)),
                                         smoothstep(0, 0.2, np.minimum(uu + 1, 1 - uu))).astype(np.float32)
        self.lower_thirds = [(build_lower_third(txt, W, H), a, b) for txt, a, b in LOWER_THIRDS]
        self.end_card = build_end_card(W, H)

    # ── static layers ────────────────────────────────────────────────────
    def _build_background(self):
        """Blurred mirror extension used where the view leaves the picture."""
        d = 0.25
        base = cv2.resize(self.levels[0], (int(self.src_w * d), int(self.src_h * d)), interpolation=cv2.INTER_AREA)
        px, py = int(700 * d), int(360 * d)
        big = cv2.copyMakeBorder(base, py, py, px, px, cv2.BORDER_REFLECT_101).astype(np.float32) / 255
        big = cv2.GaussianBlur(big, (0, 0), 38 * d)
        l = big @ LUMA
        big = (l[..., None] + (big - l[..., None]) * 0.55) * 0.52
        # soft shadow around the picture
        hh, ww = big.shape[:2]
        ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
        xw, yw = (xs + 0.5) / d - 700, (ys + 0.5) / d - 360
        ox = np.maximum(np.maximum(-xw, xw - self.src_w), 0)
        oy = np.maximum(np.maximum(-yw, yw - self.src_h), 0)
        dist = np.sqrt(ox ** 2 + oy ** 2)
        big *= (1 - 0.55 * np.exp(-dist / 30))[..., None]
        self.bg, self.bg_d, self.bg_off = big.astype(np.float32), d, (700, 360)

    def _build_glow(self):
        """Mask of glowing screens / neon / light (scene px density 1)."""
        lv = self.levels[[abs(x - 1) for x in self.dens].index(min(abs(x - 1) for x in self.dens))]
        lv = cv2.resize(lv, (self.src_w, self.src_h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(lv, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
        h, s, v = hsv[..., 0] * 360 / 256, hsv[..., 1] / 255, hsv[..., 2] / 255
        blue = ((h > 170) & (h < 250)) * smoothstep(0.30, 0.60, s) * smoothstep(0.55, 0.85, v)
        green = ((h > 80) & (h < 165)) * smoothstep(0.35, 0.65, s) * smoothstep(0.50, 0.80, v)
        warm = ((h > 25) & (h < 65)) * smoothstep(0.10, 0.30, s) * smoothstep(0.88, 0.98, v)
        m = np.maximum(np.maximum(blue, green), warm * 0.8)
        self.glow = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)

    # ── sampling ─────────────────────────────────────────────────────────
    def _maps(self, c: Cam, W, H, t, parallax):
        c = Cam(float(c.cx), float(c.cy), float(c.zoom), float(c.rot), float(c.shake))
        s = H * c.zoom / self.src_h
        a = math.radians(c.rot)
        du = (np.arange(W, dtype=np.float32) + 0.5 - W / 2) / s
        dv = (np.arange(H, dtype=np.float32) + 0.5 - H / 2) / s
        ca, sa = np.float32(math.cos(a)), np.float32(math.sin(a))
        xw = np.float32(c.cx) + ca * du[None, :] - sa * dv[:, None]
        yw = np.float32(c.cy) + sa * du[None, :] + ca * dv[:, None]
        if parallax > 1e-3:
            h = 1 / 120
            p0, p1 = cam_path(t - h), cam_path(t + h)
            vx, vy = (p1.cx - p0.cx) / (2 * h), (p1.cy - p0.cy) / (2 * h)
            zr = (math.log(p1.zoom) - math.log(p0.zoom)) / (2 * h)
            if abs(vx) + abs(vy) > 1 or abs(zr) > 0.01:
                sub = 8
                xs, ys = np.ascontiguousarray(xw[::sub, ::sub]), np.ascontiguousarray(yw[::sub, ::sub])
                dep = cv2.remap(self.depth, xs * self.depth_d - 0.5, ys * self.depth_d - 0.5,
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                dc = float(np.interp(c.cy / self.src_h, [0, 1], [0, 1]) ** 1.15)
                rel = (dep - dc) * self.edge_win[::sub, ::sub] * parallax
                dx = PARALLAX_K * vx * rel - PARALLAX_KZ * zr * rel * (xs - c.cx)
                dy = PARALLAX_K * vy * rel - PARALLAX_KZ * zr * rel * (ys - c.cy)
                xw = xw + cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR)
                yw = yw + cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR)
        return xw, yw, s

    def _sample_level(self, k, xw, yw, s):
        f = self.dens[k]
        lv = self.levels[k]
        X, Y = xw * f - 0.5, yw * f - 0.5
        r = s / f
        sigma = 0.5 * math.sqrt(max(0.0, 1 / (r * r) - 1)) if r < 0.999 else 0.0
        pad = int(4 + 3 * sigma) + 2
        x0 = max(int(math.floor(float(X.min()))) - pad, 0)
        y0 = max(int(math.floor(float(Y.min()))) - pad, 0)
        x1 = min(int(math.ceil(float(X.max()))) + pad, lv.shape[1])
        y1 = min(int(math.ceil(float(Y.max()))) + pad, lv.shape[0])
        if x1 <= x0 or y1 <= y0:
            return np.zeros(xw.shape + (3,), np.uint8)
        crop = lv[y0:y1, x0:x1]
        if sigma > 0.05:
            crop = cv2.GaussianBlur(crop, (0, 0), sigma)
        return cv2.remap(crop, X - x0, Y - y0, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    def render_base(self, c: Cam, t, parallax, W=None, H=None):
        W, H = W or self.W, H or self.H
        xw, yw, s = self._maps(c, W, H, t, parallax)
        # pyramid level with density just above the output scale; cross-fade near switches
        k = 0
        while k + 1 < len(self.dens) and self.dens[k + 1] >= s:
            k += 1
        r = s / self.dens[k]
        img = self._sample_level(k, xw, yw, s).astype(np.float32)
        if k + 1 < len(self.dens) and r < 0.6:
            w = float(smoothstep(0.5, 0.6, r))
            img2 = self._sample_level(k + 1, xw, yw, s).astype(np.float32)
            img = img * w + img2 * (1 - w)
        img *= 1 / 255
        # blurred extension where the view leaves the picture
        m = 1.0
        if xw.min() < 0 or yw.min() < 0 or xw.max() > self.src_w or yw.max() > self.src_h:
            inside = np.minimum(np.minimum(xw, self.src_w - xw), np.minimum(yw, self.src_h - yw))
            alpha = np.clip(inside * s + 0.5, 0, 1)[..., None]
            ox, oy = self.bg_off
            bg = cv2.remap(self.bg, (xw + ox) * self.bg_d - 0.5, (yw + oy) * self.bg_d - 0.5,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img = img * alpha + bg * (1 - alpha)
        return img, xw, yw, s

    def subframes(self, t):
        """Number of motion-blur sub-frames needed at time t."""
        W, H = self.W, self.H
        dt = SHUTTER / FPS
        ca, cb = cam_full(t - dt / 2, self.src_h), cam_full(t + dt / 2, self.src_h)
        u = np.array([0, W, 0, W, W / 2, 0, W, W / 2, W / 2], np.float64)
        v = np.array([0, 0, H, H, H / 2, H / 2, H / 2, 0, H], np.float64)
        px, py = to_scene(ca, u, v, W, H, self.src_h)
        u2, v2 = to_screen(cb, px, py, W, H, self.src_h)
        disp = float(np.max(np.hypot(u2 - u, v2 - v))) * 1080 / H
        return int(min(max(math.ceil(disp / MB_STEP_PX), 1), MB_MAX))

    # ── full frame ───────────────────────────────────────────────────────
    def frame(self, t):
        W, H = self.W, self.H
        par = track(PARALLAX, t)
        n = self.subframes(t)
        if n == 1:
            img, xw, yw, s = self.render_base(cam_full(t, self.src_h), t, par)
        else:
            dt = SHUTTER / FPS
            acc = None
            for i in range(n):
                ts = t + ((i + 0.5) / n - 0.5) * dt
                im, xw_i, yw_i, s_i = self.render_base(cam_full(ts, self.src_h), ts, par)
                acc = im if acc is None else acc + im
                if i == n // 2:
                    xw, yw, s = xw_i, yw_i, s_i
            img = acc / n
        cam = cam_full(t, self.src_h)
        k = H / 1080

        bloom = track(BLOOM, t)
        if bloom > 1e-3:
            sub = 4
            g = cv2.remap(self.glow, np.ascontiguousarray(xw[::sub, ::sub]) - 0.5,
                          np.ascontiguousarray(yw[::sub, ::sub]) - 0.5, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            small = cv2.resize(img, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA) * g[..., None]
            gl = cv2.GaussianBlur(small, (0, 0), 3 * k) * 0.55 + cv2.GaussianBlur(small, (0, 0), 11 * k) * 0.75
            gl = cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)
            img = 1 - (1 - img) * (1 - np.clip(gl * bloom, 0, 1))

        dof = track(DOF, t)
        if dof > 0.05:
            sub = 4
            prot = np.zeros((xw[::sub, ::sub]).shape, np.float32)
            xs, ys = xw[::sub, ::sub], yw[::sub, ::sub]
            for (x0, y0, x1, y1) in KEY_TEXTS.values():
                prot = np.maximum(prot, ((xs > x0 - 6) & (xs < x1 + 6) & (ys > y0 - 6) & (ys < y1 + 6)).astype(np.float32))
            prot = cv2.resize(cv2.GaussianBlur(prot, (0, 0), 5 * k), (W, H), interpolation=cv2.INTER_LINEAR)
            half = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            half = cv2.GaussianBlur(half, (0, 0), dof * k / 2)
            blur = cv2.resize(half, (W, H), interpolation=cv2.INTER_LINEAR)
            m = (self.dof_shape * (1 - prot) * min(1.0, dof / 2))[..., None]
            img = img + (blur - img) * m

        alarm = track(ALARM, t)
        if alarm > 1e-3:
            pulse = max(0.0, math.sin(2 * math.pi * 1.6 * t)) ** 2
            bx, by = to_screen(cam, ALARM_XY[0], ALARM_XY[1], W, H, self.src_h)
            R = 26 * s
            x0, x1 = int(max(bx - 3 * R, 0)), int(min(bx + 3 * R, W))
            y0, y1 = int(max(by - 3 * R, 0)), int(min(by + 3 * R, H))
            if x1 > x0 and y1 > y0:
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                gl = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (R * R))
                img[y0:y1, x0:x1] += (alarm * pulse * 0.55 * gl)[..., None] * np.float32([1.0, 0.16, 0.06])
            img += (alarm * pulse * 0.07 * self.vig_shape)[..., None] * np.float32([1.0, 0.05, 0.02])

        img = self.grade(img, track(GRADE, t))
        img *= (1 - track(VIGNETTE, t) * self.vig_shape)[..., None]
        dim = track(DIM, t)
        if dim > 0:
            img *= 1 - DIM_AMOUNT * dim

        for layer, t_in, t_out in self.lower_thirds:
            a, slide = overlay_state(t, t_in, t_out)
            if a > 0:
                img = self._composite(img, layer, a, int(round(slide * k)))
        a, slide = overlay_state(t, END_CARD_IN)
        if a > 0:
            img = self._composite(img, self.end_card, a, int(round(slide * k)))

        noise = np.random.default_rng(int(t * 1000)).random((H, W, 3), dtype=np.float32) - 0.5
        return np.clip(img * 255 + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def grade(img, g):
        if abs(g) < 1e-3:
            return np.clip(img, 0, 1)
        P, a = (GRADE_CHAOS if g < 0 else GRADE_ORDER), abs(g)
        wb = (1 + (np.float32(P["wb"]) - 1) * a) * (1 + (P["gain"] - 1) * a)
        img = img * wb
        l = (img @ LUMA)[..., None]
        img = l + (img - l) * (1 + (P["sat"] - 1) * a)
        img = (img - 0.5) * (1 + (P["contrast"] - 1) * a) + 0.5
        img = img + np.float32(P["lift"]) * a * (1 - img)
        knee = 0.86            # soft highlight roll-off instead of hard clipping
        img = np.where(img > knee, knee + (1 - knee) * np.tanh((img - knee) / (1 - knee)), img)
        return np.clip(img, 0, 1)

    @staticmethod
    def _composite(img, layer, opacity, dy):
        if dy:
            layer = np.roll(layer, dy, axis=0)
            layer[:dy] = 0
        a = layer[..., 3:4].astype(np.float32) * (opacity / 255)
        return img * (1 - a) + layer[..., :3].astype(np.float32) * (1 / 255) * a


# ══════════════════════════════════════════════════════════════════════════
#  Camera report (--check)
# ══════════════════════════════════════════════════════════════════════════
def hold_windows(src_h, W=1920, H=1080, still_px_s=90.0, min_scale=2.0):
    """For every key text: longest window where it is fully in frame, readable
    (>= min_scale px per scene px at 1080p) and the camera path is near-still.
    Returns {name: (seconds, t_start, t_end)}."""
    n = int(round(DURATION * FPS))
    out = {}
    for name, (x0, y0, x1, y1) in KEY_TEXTS.items():
        best, cur, start, best_span = 0, 0, 0.0, (None, None)
        for i in range(n):
            t = i / FPS
            c0, c1 = cam_path(t - 0.5 / FPS), cam_path(t + 0.5 / FPS)
            u, v = to_screen(c0, [x0, x1, x0, x1], [y0, y0, y1, y1], W, H, src_h)
            inside = u.min() >= 0 and u.max() <= W and v.min() >= 0 and v.max() <= H
            u2, v2 = to_screen(c1, [x0, x1, x0, x1], [y0, y0, y1, y1], W, H, src_h)
            speed = float(np.max(np.hypot(u2 - u, v2 - v))) * FPS
            readable = H * c0.zoom / src_h >= min_scale
            if inside and readable and speed < still_px_s:
                if cur == 0:
                    start = t
                cur += 1
                if cur > best:
                    best, best_span = cur, (start, t)
            else:
                cur = 0
        out[name] = (best / FPS, *best_span)
    return out


def camera_report(src_w, src_h):
    W, H = FINAL_SIZE
    print("\nKey text holds (in frame, readable scale, camera path near-still < 90 px/s @1080p):")
    for name, (secs, t0, t1) in hold_windows(src_h).items():
        if secs:
            x0, y0, x1, y1 = KEY_TEXTS[name]
            sc = H * cam_path((t0 + t1) / 2).zoom / src_h
            print(f"  {name:52s} {secs:4.2f}s  ({t0:5.2f}-{t1:5.2f}s, sign height {(y1 - y0) * sc:4.0f}px)")
        else:
            print(f"  {name:52s}  -- never held")
    print("\nPeak screen speed per move (px/frame @1080p, before motion blur) and view bounds:")
    for k0, k1 in zip(CAMERA, CAMERA[1:]):
        peak, pc, out = 0.0, 0.0, 0.0
        for i in range(int(k0.t * FPS), int(k1.t * FPS) + 1):
            t = i / FPS
            c0, c1 = cam_path(t), cam_path(t + 1 / FPS)
            u = np.array([0, W, 0, W, W / 2]); v = np.array([0, 0, H, H, H / 2])
            px, py = to_scene(c0, u, v, W, H, src_h)
            u2, v2 = to_screen(c1, px, py, W, H, src_h)
            d = np.hypot(u2 - u, v2 - v)
            peak, pc = max(peak, float(np.max(d))), max(pc, float(d[-1]))
            cf = cam_full(t, src_h)
            px, py = to_scene(cf, u[:4], v[:4], W, H, src_h)
            out = max(out, float(np.max(np.maximum.reduce([-px, px - src_w, -py, py - src_h]))))
        flag = "" if out <= 0.5 else f"   view leaves picture by {out:.1f} scene px"
        print(f"  {k0.t:5.2f} -> {k1.t:5.2f}s  zoom {k0.zoom:4.2f} -> {k1.zoom:4.2f}   "
              f"peak centre {pc:6.1f}  corners {peak:6.1f}{flag}")


# ══════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════
_R = None


def _worker_init(size, master_path, scene_wh):
    global _R, SCENE_W, SCENE_H
    SCENE_W, SCENE_H = scene_wh
    cv2.setNumThreads(1)
    _R = Renderer(size, master_path)


def _worker_frame(i):
    return _R.frame(i / FPS).tobytes()


def encode(out_path, size, master_path, workers, crf, preset):
    W, H = size
    n = int(round(DURATION * FPS))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
           "-g", str(2 * FPS), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-movflags", "+faststart", "-frames:v", str(n), out_path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    import multiprocessing as mp
    # spawn, not fork: forking after OpenCV/BLAS have started threads deadlocks the workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(size, master_path, (SCENE_W, SCENE_H))) as pool:
        for i, buf in enumerate(pool.imap(_worker_frame, range(n), chunksize=2)):
            ff.stdin.write(buf)
            if i % 30 == 29 or i == n - 1:
                el = time.time() - t0
                print(f"\r  frame {i + 1}/{n}  {el:5.0f}s  (eta {el / (i + 1) * (n - i - 1):4.0f}s)", end="", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"\n  wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    probe(out_path)


def probe(path):
    if not shutil.which("ffprobe"):
        return
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    info = json.loads(out)
    st, fmt = info["streams"][0], info["format"]
    audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    print(f"  ffprobe: {st['codec_name']} ({st.get('profile')}) {st['width']}x{st['height']} {st['pix_fmt']} "
          f"{st['r_frame_rate']} fps, {st['nb_read_frames']} frames, duration {float(fmt['duration']):.3f}s, "
          f"audio streams: {len(audio.splitlines()) if audio else 0}")


def contact_sheet(size, master_path):
    R = Renderer(size, master_path)
    W, H = size
    os.makedirs("output/keyframes", exist_ok=True)
    font = ImageFont.truetype(find_font(FONT_UI), 22)
    thumbs = []
    for t, label in CONTACT_FRAMES:
        f = R.frame(t)
        Image.fromarray(f).save(f"output/keyframes/{t:05.2f}s.png")
        th = Image.fromarray(f).resize((640, 360), Image.LANCZOS)
        cell = Image.new("RGB", (640, 396), (24, 24, 24))
        cell.paste(th, (0, 36))
        ImageDraw.Draw(cell).text((6, 6), f"{t:5.2f}s  {label}", font=font, fill=(255, 255, 255))
        thumbs.append(cell)
        print(f"  {t:5.2f}s {label}")
    cols, gap = 4, 8
    rows = math.ceil(len(thumbs) / cols)
    cw, ch = thumbs[0].size
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap), (24, 24, 24))
    for i, im in enumerate(thumbs):
        sheet.paste(im, (gap + (i % cols) * (cw + gap), gap + (i // cols) * (ch + gap)))
    sheet.save("output/contact_sheet.jpg", quality=90)

    # 1:1 crops of every key text in the middle of its hold -> legibility check
    crops = []
    for name, (secs, t0, t1) in hold_windows(R.src_h, W, H).items():
        if not secs:
            continue
        x0, y0, x1, y1 = KEY_TEXTS[name]
        t = round((t0 + t1) / 2 * FPS) / FPS
        f = R.frame(t)
        u, v = to_screen(cam_full(t, R.src_h), [x0, x1, x0, x1], [y0, y0, y1, y1], W, H, R.src_h)
        cw_, ch_ = min(W, max(560, int(u.max() - u.min()) + 60)), min(H, max(280, int(v.max() - v.min()) + 60))
        xa = int(min(max((u.min() + u.max()) / 2 - cw_ / 2, 0), W - cw_))
        ya = int(min(max((v.min() + v.max()) / 2 - ch_ / 2, 0), H - ch_))
        crop = Image.fromarray(f[ya:ya + ch_, xa:xa + cw_])
        lab = Image.new("RGB", (crop.width, crop.height + 34), (24, 24, 24))
        lab.paste(crop, (0, 34))
        ImageDraw.Draw(lab).text((6, 5), f"{name}  @{t:.2f}s, held {secs:.2f}s (1:1 at {W}x{H})",
                                 font=ImageFont.truetype(find_font(FONT_UI), 19), fill=(255, 255, 255))
        crops.append(lab)
    maxw = 1940
    rows_, row, x = [], [], 0
    for c in crops:
        if x + c.width > maxw - 10 and row:
            rows_.append(row)
            row, x = [], 0
        row.append(c)
        x += c.width + 10
    rows_.append(row)
    sheet = Image.new("RGB", (maxw, sum(max(c.height for c in r) + 10 for r in rows_) + 10), (40, 40, 40))
    y = 10
    for r in rows_:
        x = 10
        for c in r:
            sheet.paste(c, (x, y))
            x += c.width + 10
        y += max(c.height for c in r) + 10
    sheet.save("output/legibility_sheet.jpg", quality=92)
    print("  wrote output/contact_sheet.jpg, output/legibility_sheet.jpg, output/keyframes/*.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="960x540 quick render")
    ap.add_argument("--contact-sheet", action="store_true", help="export key frames + legibility crops")
    ap.add_argument("--stills", nargs="*", type=float, help="render single frames at these times")
    ap.add_argument("--check", action="store_true", help="print camera report and exit")
    ap.add_argument("--upscaler", choices=["auto", "realesrgan", "lanczos"], default="auto")
    ap.add_argument("--reupscale", action="store_true", help="ignore the cached 4x master")
    ap.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    ap.add_argument("--out", help="output path override")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    src = load_scene()
    global SCENE_W, SCENE_H
    SCENE_H, SCENE_W = src.shape[:2]
    if args.check:
        camera_report(src.shape[1], src.shape[0])
        return
    master_path, method = ensure_master(args.upscaler, args.reupscale)
    print(f"master: {master_path} ({method})")
    size = PREVIEW_SIZE if args.preview else FINAL_SIZE
    if args.stills is not None:
        R = Renderer(size, master_path)
        os.makedirs("output/stills", exist_ok=True)
        for t in args.stills:
            p = f"output/stills/{t:05.2f}s.png"
            Image.fromarray(R.frame(t)).save(p)
            print("  wrote", p)
        return
    if args.contact_sheet:
        contact_sheet(size, master_path)
        return
    out = args.out or (OUT_PREVIEW if args.preview else OUT_FINAL)
    crf, preset = (PREVIEW_CRF, PREVIEW_PRESET) if args.preview else (CRF, PRESET)
    print(f"rendering {out} at {size[0]}x{size[1]}, {FPS} fps, {DURATION}s, x264 {preset} crf {crf}")
    encode(out, size, master_path, args.workers, crf, preset)


if __name__ == "__main__":
    main()
