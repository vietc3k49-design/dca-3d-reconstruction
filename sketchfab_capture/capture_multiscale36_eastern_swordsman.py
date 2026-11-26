#!/usr/bin/env python3
"""
Step 1: Smart Multi-Scale 36-View 1536x1536 RGBA Capture for Eastern Female Swordsman.
Strictly 36 views, single shared PINHOLE camera (fixed FOV), dual-background exact RGBA matting:
  - Group A (20 views): Full-body anchor sphere (r=6.28m) — 12 @ +12°, 6 @ -6° (shoes/4 legs), 2 @ +36°
  - Group B ( 8 views): Upper-body / Face / Hairpin / Sword close-ups (r=2.65m, 1,372 px/m)
  - Group C ( 4 views): Plum blossom tree & pot close-ups (r=2.60m, 1,398 px/m -> 3.5-4.5 px twigs)
  - Group D ( 4 views): High-angle +44° tabletop / 90° corners / sword close-ups (r=2.95m)
"""

from __future__ import annotations

import json
import os
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_sketchfab import (
    DEFAULT_URL,
    LocalHTMLServer,
    alpha_bbox,
    build_preview_grid,
    build_viewer_html,
    compute_camera_matrices,
    dual_background_matte,
    enter_matte_mode,
    extract_model_info,
    launch_browser,
    orbit_eye,
    sf_grab_rgb,
    sf_set_bg_color,
    sf_set_camera,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "workspace_runs" / "eastern_swordsman_multiscale36")))
CAP_DIR = OUT_DIR / "capture_1536"
IMG_DIR = CAP_DIR / "images"
GT_DIR = OUT_DIR / "gt_eval"

RES_W = 1536
RES_H = 1536

# Exact Sketchfab world targets (Z-up meters) calibrated from scene 3D landmarks
T_FULL = np.array([-0.327704, 0.215656, 0.642498], dtype=np.float64)
T_UPPER = np.array([-0.060, 0.230, 1.220], dtype=np.float64)
T_PLUM = np.array([-0.380, 0.020, 1.040], dtype=np.float64)
T_TABLE = np.array([-0.220, 0.140, 0.800], dtype=np.float64)


def build_36_multiscale_plan() -> list[dict[str, Any]]:
    """Construct the exact 36-view multi-scale camera plan."""
    specs: list[dict[str, Any]] = []

    # Group A: 20 Full-body views (r = 6.28 m)
    for az in range(0, 360, 30):  # 12 views @ +12 deg
        specs.append({"group": "full_mid", "target": T_FULL, "radius": 6.28, "elev_deg": 12.0, "az_deg": float(az)})
    for az in (15, 75, 135, 195, 255, 315):  # 6 views @ -6 deg (feet, skirt hem, under table, 4 legs)
        specs.append({"group": "full_low", "target": T_FULL, "radius": 6.28, "elev_deg": -6.0, "az_deg": float(az)})
    for az in (285, 105):  # 2 views @ +36 deg (high full-body bridge)
        specs.append({"group": "full_hi", "target": T_FULL, "radius": 6.28, "elev_deg": 36.0, "az_deg": float(az)})

    # Group B: 8 Upper-body / Face / Hairpin / Sword optical close-ups (r = 2.65 m)
    upper_poses = [
        (225.0, 10.0),  # right profile/3-4
        (255.0, 16.0),  # right 3/4 frontal
        (275.0, 6.0),   # direct front eye-level (eyes/nose/lips/collar)
        (295.0, 18.0),  # hero front-left 3/4 high
        (315.0, 6.0),   # hero 3/4 eye-level (matches reference hero angle)
        (345.0, 14.0),  # left 3/4 profile (nose bridge, sword hilt, hairpin)
        (65.0, 14.0),   # back-left 3/4 (ponytail, golden crown)
        (145.0, 14.0),  # back-right 3/4 (cloak, hair)
    ]
    for az, el in upper_poses:
        specs.append({"group": "close_face", "target": T_UPPER, "radius": 2.65, "elev_deg": el, "az_deg": az})

    # Group C: 4 Plum Blossom Tree & Pot optical close-ups (r = 2.60 m)
    plum_poses = [
        (245.0, 12.0),  # unobstructed front-right
        (290.0, 18.0),  # front-left + overlap with character
        (190.0, 15.0),  # pure table-end side view
        (130.0, 20.0),  # rear view of branches & pot
    ]
    for az, el in plum_poses:
        specs.append({"group": "close_plum", "target": T_PLUM, "radius": 2.60, "elev_deg": el, "az_deg": az})

    # Group D: 4 High-angle +44 deg Tabletop / 90 deg Corners / Sword views (r = 2.95 m)
    table_poses = [
        (275.0, 44.0),  # front top-down
        (350.0, 44.0),  # left top-down over sword & tabletop
        (100.0, 44.0),  # rear top-down over tabletop & back corners
        (195.0, 44.0),  # right/table-end top-down
    ]
    for az, el in table_poses:
        specs.append({"group": "close_table", "target": T_TABLE, "radius": 2.95, "elev_deg": el, "az_deg": az})

    assert len(specs) == 36, f"Expected 36 views, got {len(specs)}"
    return specs


def main() -> int:
    t0 = time.perf_counter()
    for d in (OUT_DIR, CAP_DIR, IMG_DIR, GT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    uid, _ = extract_model_info(DEFAULT_URL)
    specs = build_36_multiscale_plan()
    html = build_viewer_html(uid=uid, transparent=False)

    print("=" * 72, flush=True)
    print(f"[Step 1] Multi-Scale 36-View RGBA Capture ({RES_W}x{RES_H}) -> {CAP_DIR}", flush=True)
    print("=" * 72, flush=True)

    with LocalHTMLServer(html) as srv:
        with sync_playwright() as pw:
            cache_dir = OUT_DIR / "_browser_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            browser_args = [
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
            ]
            context = None
            for ch in ("msedge", "chrome", None):
                try:
                    kw: dict[str, Any] = {
                        "headless": True,
                        "args": browser_args,
                        "viewport": {"width": 960, "height": 960},
                        "device_scale_factor": 1,
                    }
                    if ch:
                        kw["channel"] = ch
                    context = pw.chromium.launch_persistent_context(str(cache_dir), **kw)
                    print(f"[Browser] Persistent context launched ({ch or 'chromium'}, headless=True)", flush=True)
                    break
                except Exception as e:
                    print(f"[Browser] Channel {ch} failed: {e}", flush=True)
            if context is None:
                raise RuntimeError("Could not launch Chromium/Edge persistent context")
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.set_viewport_size({"width": 960, "height": 960})
                page.goto(f"http://127.0.0.1:{srv.port}/", wait_until="domcontentloaded", timeout=60000)

                t_load0 = time.time()
                while True:
                    st = page.evaluate(
                        "() => ({ready: !!window.sfReady, err: window.sfError, "
                        "mp: window.modelProgress || 0, tp: window.textureProgress || 0})"
                    )
                    elapsed_load = time.time() - t_load0
                    print(
                        f"  [Load +{elapsed_load:5.1f}s] ready={st['ready']} model={st['mp']*100:5.1f}% tex={st['tp']*100:5.1f}%",
                        flush=True,
                    )
                    if st["err"]:
                        raise RuntimeError(f"Sketchfab Viewer error: {st['err']}")
                    if st["ready"]:
                        break
                    if elapsed_load > 360.0:
                        raise TimeoutError("Timed out (>360s) waiting for Sketchfab viewerready")
                    time.sleep(5.0)

                page.evaluate(
                    """() => new Promise(resolve => {
                        const api = window.sfApi;
                        api.setEnableCameraConstraints(false, { preventCameraConstraintsFocus: true }, () => {
                            api.hideAnnotationTooltips(() => {
                                api.setTextureQuality('hd', () => {
                                    api.pause(() => resolve(true));
                                });
                            });
                        });
                    })"""
                )
                t_wait = time.time()
                while time.time() - t_wait < 90.0:
                    tp = float(page.evaluate("window.textureProgress || 0"))
                    if tp >= 0.99:
                        break
                    print(f"  [HD Tex +{time.time()-t_wait:4.1f}s] tex={tp*100:5.1f}%", flush=True)
                    time.sleep(2.5)
                time.sleep(1.2)

                cam_init = page.evaluate("() => new Promise(res => window.sfApi.getCameraLookAt((e, c) => res(c)))")
                fov_deg = float(page.evaluate("() => new Promise(res => window.sfApi.getFov((e, f) => res(f)))"))
                pos0 = np.asarray(cam_init["position"], dtype=np.float64)
                target0 = np.asarray(cam_init["target"], dtype=np.float64)
                rel0 = pos0 - target0
                base_az_rad = math.atan2(float(rel0[1]), float(rel0[0]))
                print(
                    f"[Init] base_az={math.degrees(base_az_rad):.2f}°, fovy={fov_deg:.4f}°, "
                    f"pos0={np.round(pos0, 3).tolist()}, target0={np.round(target0, 3).tolist()}",
                    flush=True,
                )

                matte_env = enter_matte_mode(page)
                if (matte_env.get("corner_diff_max") or 0.0) < 245.0:
                    raise RuntimeError(f"Dual-background matte failed verification: {matte_env}")

                frames: list[dict[str, Any]] = []
                img_paths: list[Path] = []
                cur_bg = "w"  # enter_matte_mode ends on white [1,1,1]

                for idx, sp in enumerate(specs):
                    grp = sp["group"]
                    tgt = sp["target"]
                    rad = float(sp["radius"])
                    elev_deg = float(sp["elev_deg"])
                    az_deg = float(sp["az_deg"])
                    phi = base_az_rad + math.radians(az_deg)
                    eye = orbit_eye(tgt, rad, phi, elev_deg)

                    sf_set_camera(page, eye, tgt.tolist(), 0.0)
                    time.sleep(0.28)

                    # Read exact Sketchfab camera back
                    cam_now = page.evaluate("() => new Promise(res => window.sfApi.getCameraLookAt((e, c) => res(c)))")
                    fov_now = float(page.evaluate("() => new Promise(res => window.sfApi.getFov((e, f) => res(f)))"))
                    mats = compute_camera_matrices(cam_now["position"], cam_now["target"])

                    # Alternate starting background to halve background switch latency (exact same pose!)
                    if cur_bg == "w":
                        img_w = sf_grab_rgb(page, RES_W, RES_H)
                        sf_set_bg_color(page, [0.0, 0.0, 0.0])
                        time.sleep(0.26)
                        img_b = sf_grab_rgb(page, RES_W, RES_H)
                        cur_bg = "b"
                    else:
                        img_b = sf_grab_rgb(page, RES_W, RES_H)
                        sf_set_bg_color(page, [1.0, 1.0, 1.0])
                        time.sleep(0.26)
                        img_w = sf_grab_rgb(page, RES_W, RES_H)
                        cur_bg = "w"

                    rgba = dual_background_matte(img_b, img_w, shadow_alpha_cutoff=0.22)
                    fname = f"view_{idx:03d}_{grp}_el{int(round(elev_deg)):+03d}_az{int(round(az_deg)) % 360:03d}.png"
                    out_p = IMG_DIR / fname
                    Image.fromarray(rgba, "RGBA").save(out_p, compress_level=1)
                    img_paths.append(out_p)

                    a01 = rgba[..., 3].astype(np.float32) * (1.0 / 255.0)
                    bb = alpha_bbox(a01)
                    margins = (
                        [round(bb[0] / RES_W, 4), round(1 - bb[2] / RES_W, 4), round(bb[1] / RES_H, 4), round(1 - bb[3] / RES_H, 4)]
                        if bb is not None
                        else None
                    )
                    fg_frac = float((a01 > 0.1).mean())

                    fovy_rad = math.radians(fov_now)
                    fl_y = (RES_H * 0.5) / math.tan(fovy_rad * 0.5)
                    fovx_rad = 2.0 * math.atan((RES_W * 0.5) / fl_y)

                    frames.append(
                        {
                            "file_path": f"images/{fname}",
                            "view_index": idx,
                            "group": grp,
                            "width": RES_W,
                            "height": RES_H,
                            "azimuth_deg": round(az_deg, 3),
                            "elevation_deg": round(elev_deg, 3),
                            "radius": round(rad, 4),
                            "fovy_deg": fov_now,
                            "fovx_deg": math.degrees(fovx_rad),
                            "fl_x": fl_y,
                            "fl_y": fl_y,
                            "cx": RES_W * 0.5,
                            "cy": RES_H * 0.5,
                            "eye": cam_now["position"],
                            "target": cam_now["target"],
                            "transform_matrix": mats["c2w_opengl"],
                            "c2w_colmap": mats["c2w_colmap"],
                            "w2c_colmap_R": mats["w2c_colmap_R"],
                            "w2c_colmap_t": mats["w2c_colmap_t"],
                            "alpha_margins_lrtb": margins,
                            "fg_fraction": round(fg_frac, 4),
                        }
                    )
                    print(
                        f"  [{idx + 1:02d}/36] {fname:42s} r={rad:.2f}m fg={fg_frac*100:4.1f}% margins={margins}",
                        flush=True,
                    )

                meta_obj = {
                    "model_uid": uid,
                    "source_url": DEFAULT_URL,
                    "method": "multiscale_36_orbit",
                    "resolution": [RES_W, RES_H],
                    "initial_camera": cam_init,
                    "alpha_matte": True,
                    "matte_env": matte_env,
                    "frames": frames,
                }
                meta_text = json.dumps(meta_obj, indent=2)
                (CAP_DIR / "cameras_meta.json").write_text(meta_text, encoding="utf-8")
                (GT_DIR / "cameras_meta.json").write_text(meta_text, encoding="utf-8")

                preview_path = CAP_DIR / "preview_grid_36.png"
                build_preview_grid(img_paths, preview_path, cols=6, thumb_w=280)
            finally:
                context.close()

    elapsed = time.perf_counter() - t0
    print("=" * 72, flush=True)
    print(f"[Step 1 DONE] Captured 36 multi-scale {RES_W}x{RES_H} RGBA images in {elapsed:.1f}s!", flush=True)
    print(f"  Preview grid: {CAP_DIR / 'preview_grid_36.png'}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
