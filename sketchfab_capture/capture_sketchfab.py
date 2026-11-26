#!/usr/bin/env python3
"""
Chương trình tự động mở mô hình 3D trên Sketchfab, tự động kéo/xoay/zoom ở nhiều góc độ
và chụp ảnh sạch (ẩn toàn bộ UI/annotations) ở nhiều độ phân giải khác nhau để phục vụ
tái tạo 3D (COLMAP SIFT / Bundle Adjustment / 3D Gaussian Splatting / NeRF).

Hỗ trợ 2 phương thức điều khiển camera (--method):
  1. orbit (Mặc định - Khuyên dùng cho tái tạo 3D):
     Quỹ đạo cầu 360° quanh tâm mô hình với nhiều vòng góc cao (elevation) và mức zoom,
     kèm hiệu ứng xoay trực quan trên trình duyệt.
  2. drag:
     Giả lập thao tác nhấn giữ chuột kéo (mouse drag) ngang/dọc và lăn chuột (mouse wheel)
     trực tiếp trên khung nhìn 3D như người dùng thật.

Hỗ trợ 2 chế độ mở trang (--viewer-source):
  1. api (Mặc định): Nhúng Sketchfab Viewer API chính thức giúp ẩn sạch 100% nhãn số (1,2,3...),
     bật texture HD, tắt giới hạn góc khóa camera, xuất ảnh WebGL gốc ở mọi độ phân giải
     và lưu kèm thông số camera chuẩn (transforms.json & cameras_meta.json).
  2. webpage: Mở trực tiếp đường link trang web Sketchfab, dùng CSS ẩn thanh công cụ và
     kéo chuột chụp màn hình khung canvas theo từng độ phân giải cửa sổ.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import math
import re
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

DEFAULT_URL = "https://sketchfab.com/3d-models/eastern-female-swordsman-db24b935f7ca476fad65176b291b1552"


def extract_model_info(url_or_uid: str) -> tuple[str, str]:
    """Trích xuất (model_uid, model_slug) từ link Sketchfab bất kỳ hoặc UID 32 ký tự."""
    s = url_or_uid.strip().rstrip("/")
    if re.fullmatch(r"[0-9a-fA-F]{32}", s):
        return s.lower(), f"sketchfab_{s[:8].lower()}"

    # Dạng: https://sketchfab.com/3d-models/eastern-female-swordsman-db24b935f7ca476fad65176b291b1552
    m = re.search(r"/3d-models/([^/?#]+)-([0-9a-fA-F]{32})", s)
    if m:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", m.group(1)).strip("_")
        return m.group(2).lower(), slug

    # Dạng: https://sketchfab.com/models/db24b935f7ca476fad65176b291b1552
    m2 = re.search(r"([0-9a-fA-F]{32})", s)
    if m2:
        uid = m2.group(1).lower()
        return uid, f"sketchfab_{uid[:8]}"

    raise ValueError(f"Không tìm thấy mã UID 32 ký tự của Sketchfab trong: {url_or_uid}")


def parse_resolutions(res_list: list[str]) -> list[tuple[int, int]]:
    """Phân tích danh sách độ phân giải dạng ['800x600', '1280x720', '1920x1080']."""
    out: list[tuple[int, int]] = []
    for item in res_list:
        m = re.fullmatch(r"(\d+)[xX](\d+)", item.strip())
        if not m:
            raise ValueError(f"Độ phân giải không hợp lệ '{item}' (ví dụ hợp lệ: 1280x720)")
        w, h = int(m.group(1)), int(m.group(2))
        if w < 64 or h < 64:
            raise ValueError(f"Độ phân giải quá nhỏ: {w}x{h}")
        out.append((w, h))
    return out


def compute_camera_matrices(
    eye: list[float],
    target: list[float],
    up_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """
    Tính ma trận ngoại tham camera từ eye, target, up_world=[0,0,1] (hệ tọa độ Z-up của Sketchfab):
      - c2w_opengl: 4x4 chuẩn NeRF / Three.js / OpenGL (+X phải, +Y lên, -Z hướng nhìn)
      - c2w_colmap: 4x4 chuẩn OpenCV / COLMAP (+X phải, +Y xuống, +Z hướng nhìn)
      - w2c_colmap_R, w2c_colmap_t: R (3x3) và t (3,) chuẩn COLMAP (X_cam = R * X_world + t)
    """
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    up = np.asarray(up_world, dtype=np.float64)

    forward = t - e
    dist = float(np.linalg.norm(forward))
    if dist < 1e-9:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        dist = 1.0
    else:
        forward = forward / dist

    right = np.cross(forward, up)
    r_norm = float(np.linalg.norm(right))
    if r_norm < 1e-6:
        # Trường hợp nhìn thẳng từ đỉnh cực Bắc/Nam
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, alt_up)
        r_norm = float(np.linalg.norm(right))
    right = right / r_norm
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    c2w_opengl = np.eye(4, dtype=np.float64)
    c2w_opengl[:3, 0] = right
    c2w_opengl[:3, 1] = cam_up
    c2w_opengl[:3, 2] = -forward
    c2w_opengl[:3, 3] = e

    c2w_colmap = np.eye(4, dtype=np.float64)
    c2w_colmap[:3, 0] = right
    c2w_colmap[:3, 1] = -cam_up
    c2w_colmap[:3, 2] = forward
    c2w_colmap[:3, 3] = e

    R_w2c = c2w_colmap[:3, :3].T
    t_w2c = -R_w2c @ e

    return {
        "distance": dist,
        "c2w_opengl": c2w_opengl.tolist(),
        "c2w_colmap": c2w_colmap.tolist(),
        "w2c_colmap_R": R_w2c.tolist(),
        "w2c_colmap_t": t_w2c.tolist(),
    }


def build_viewer_html(uid: str, transparent: bool = False) -> str:
    """Tạo trang HTML cục bộ kết nối Sketchfab Viewer API với chế độ sạch không giao diện rác."""
    trans_flag = 1 if transparent else 0
    bg_color = "transparent" if transparent else "#1e1e1e"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sketchfab Multi-View Capture ({uid})</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {bg_color};
    }}
    #api-frame {{
      border: 0;
      width: 100vw;
      height: 100vh;
      display: block;
    }}
  </style>
</head>
<body>
  <iframe id="api-frame" allow="autoplay; fullscreen; xr-spatial-tracking"></iframe>
  <script src="https://static.sketchfab.com/api/sketchfab-viewer-1.12.1.js"></script>
  <script>
    window.sfReady = false;
    window.sfError = null;
    window.modelProgress = 0;
    window.textureProgress = 0;

    const iframe = document.getElementById('api-frame');
    const client = new Sketchfab('1.12.1', iframe);

    client.init('{uid}', {{
      autostart: 1,
      preload: 1,
      transparent: {trans_flag},
      ui_annotations: 0,
      annotations_visible: 0,
      ui_controls: 0,
      ui_infos: 0,
      ui_stop: 0,
      ui_watermark: 0,
      ui_watermark_link: 0,
      ui_hint: 0,
      ui_ar: 0,
      ui_help: 0,
      ui_settings: 0,
      ui_vr: 0,
      ui_fullscreen: 0,
      ui_inspector: 0,
      dnt: 1,
      success: function(api) {{
        window.sfApi = api;
        api.start();
        api.addEventListener('modelLoadProgress', function(ev) {{
          if (ev && typeof ev.progress === 'number') window.modelProgress = ev.progress;
        }});
        api.addEventListener('textureLoadProgress', function(ev) {{
          if (ev && typeof ev.progress === 'number') window.textureProgress = ev.progress;
        }});
        api.addEventListener('viewerready', function() {{
          window.sfReady = true;
        }});
      }},
      error: function(err) {{
        window.sfError = String(err || 'Sketchfab init error');
      }}
    }});
  </script>
</body>
</html>
"""


class LocalHTMLServer:
    """HTTP server nội bộ (127.0.0.1) để phục vụ trang nhúng Sketchfab API không bị chặn CORS."""

    def __init__(self, html_content: str) -> None:
        self.html_bytes = html_content.encode("utf-8")
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(parent.html_bytes)))
                self.end_headers()
                self.wfile.write(parent.html_bytes)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "LocalHTMLServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def launch_browser(playwright_obj: Any, headless: bool, channel: str | None):
    """Khởi chạy trình duyệt ưu tiên Chrome/Edge có sẵn trên máy với WebGL GPU."""
    browser_args = [
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
    ]
    channels = [channel] if channel else ["chrome", "msedge", None]
    last_err = None
    for ch in channels:
        try:
            kwargs: dict[str, Any] = {"headless": headless, "args": browser_args}
            if ch:
                kwargs["channel"] = ch
            browser = playwright_obj.chromium.launch(**kwargs)
            print(f"[Browser] Đã mở trình duyệt ({ch or 'chromium'} - version {browser.version}, headless={headless})")
            return browser
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Không thể khởi chạy trình duyệt. Lỗi cuối: {last_err}")


def save_data_url_png(data_url: str, out_path: Path) -> None:
    """Giải mã chuỗi data:image/png;base64,... và lưu ra file ảnh."""
    _, b64 = data_url.split(",", 1)
    out_path.write_bytes(base64.b64decode(b64))


# ---------------------------------------------------------------------------------------------
# V2: Sketchfab API helpers (có timeout để không bao giờ treo), Dual-Background Matting, Auto Full-body
# ---------------------------------------------------------------------------------------------

def sf_call(page: Page, body: str, timeout_ms: int = 15000) -> Any:
    """Chạy đoạn JS gọi Sketchfab API; JS phải gọi `done(value)`. Trả {'__timeout': True} nếu callback không về."""
    js = (
        "() => new Promise((resolve) => {"
        f" const timer = setTimeout(() => resolve({{__timeout: true}}), {int(timeout_ms)});"
        " const done = (v) => { clearTimeout(timer); resolve(v === undefined ? null : v); };"
        f" try {{ {body} }} catch (e) {{ clearTimeout(timer); resolve({{__error: String(e)}}); }}"
        " })"
    )
    return page.evaluate(js)


def sf_set_camera(page: Page, eye: list[float], target: list[float], duration: float = 0.0) -> None:
    sf_call(
        page,
        f"window.sfApi.setCameraLookAt({json.dumps(eye)}, {json.dumps(target)}, {duration:.3f}, () => done(true));",
        20000,
    )


def sf_set_bg_color(page: Page, color: list[float]) -> None:
    sf_call(page, f"window.sfApi.setBackground({{color: {json.dumps(color)}}}, () => done(true));", 10000)


def sf_grab_rgb(page: Page, w: int, h: int) -> np.ndarray:
    """Chụp ảnh WebGL gốc -> mảng float32 (H, W, 3)."""
    d = sf_call(
        page,
        f"window.sfApi.getScreenShot({int(w)}, {int(h)}, 'image/png', (e, r) => e ? done({{__error: String(e)}}) : done(r));",
        40000,
    )
    if not isinstance(d, str):
        raise RuntimeError(f"getScreenShot thất bại: {d}")
    im = Image.open(io.BytesIO(base64.b64decode(d.split(",", 1)[1]))).convert("RGB")
    return np.asarray(im, dtype=np.float32)


def sample_bg_color(page: Page) -> list[float]:
    """Ước lượng màu nền gốc (giữa cạnh trên) để khôi phục khi chỉ dùng --auto-fullbody."""
    try:
        a = sf_grab_rgb(page, 160, 120)
        px = a[1:4, 70:90].reshape(-1, 3).mean(axis=0) / 255.0
        return [round(float(c), 4) for c in px]
    except Exception:
        return [0.12, 0.12, 0.12]


def suppress_floor_shadow(alpha: np.ndarray, cutoff: float, grow_px: int = 3) -> np.ndarray:
    """Xoá vùng alpha thấp (< cutoff) KHÔNG nằm sát vật thể đặc (alpha>=0.9 nở ra grow_px) => bỏ bóng đổ sàn nhưng
    giữ viền tóc/hoa mỏng bám sát thân mô hình."""
    if cutoff <= 0:
        return alpha
    from PIL import ImageFilter

    solid = Image.fromarray(((alpha >= 0.9) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * grow_px + 1))
    near_solid = np.asarray(solid) > 0
    out = alpha.copy()
    out[(alpha < cutoff) & ~near_solid] = 0.0
    return out


def dual_background_matte(
    img_black: np.ndarray, img_white: np.ndarray, shadow_alpha_cutoff: float = 0.0
) -> np.ndarray:
    """Dual-background matting: alpha = 1 - mean(I_white - I_black)/255 ; RGB = I_black / alpha (unpremultiplied).
    Trả mảng uint8 (H, W, 4) RGBA."""
    diff = (img_white - img_black).mean(axis=-1)
    alpha = np.clip(1.0 - diff / 255.0, 0.0, 1.0)
    alpha = suppress_floor_shadow(alpha, shadow_alpha_cutoff)
    alpha[alpha < 1.5 / 255.0] = 0.0  # khử nhiễu dither nền => alpha == 0 tuyệt đối
    rgb = np.clip(img_black / np.maximum(alpha, 1e-4)[..., None], 0.0, 255.0)
    a8 = np.rint(alpha * 255.0).astype(np.uint8)
    rgb[a8 == 0] = 0.0
    return np.dstack([np.rint(rgb).astype(np.uint8), a8])


def alpha_bbox(alpha01: np.ndarray, thr: float = 0.1) -> tuple[int, int, int, int] | None:
    m = alpha01 > thr
    ys = np.where(m.any(axis=1))[0]
    xs = np.where(m.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def enter_matte_mode(page: Page) -> dict[str, Any]:
    """Tắt bóng đổ sàn (environment shadow) + vignette/post-FX để nền đơn sắc cho alpha chính xác tuyệt đối."""
    info: dict[str, Any] = {}
    info["shadow_off"] = sf_call(
        page, "window.sfApi.setEnvironment({shadowEnabled: false}, (e) => done(e ? String(e) : 'OK'));", 8000
    )
    info["postfx"] = "unverified"
    for mode, pp in (("vignette_only_off", {"vignetteEnable": False}), ("all_postfx_off", {"enable": False})):
        sf_call(page, f"window.sfApi.setPostProcessing({json.dumps(pp)}, () => done(true));", 8000)
        diffs = []
        sf_set_bg_color(page, [0.0, 0.0, 0.0])
        time.sleep(0.4)
        b = sf_grab_rgb(page, 128, 96)
        sf_set_bg_color(page, [1.0, 1.0, 1.0])
        time.sleep(0.4)
        w = sf_grab_rgb(page, 128, 96)
        d = (w - b).mean(axis=-1)
        diffs = [d[0, 0], d[0, -1], d[-1, 0], d[-1, -1]]
        if max(diffs) >= 250.0:
            info["postfx"] = mode
            break
    info["corner_diff_max"] = float(max(diffs)) if diffs else None
    print(f"[Matte] shadow_off={info['shadow_off']} postfx={info['postfx']} corner_diff={info['corner_diff_max']}")
    return info


def exit_matte_mode(page: Page, info: dict[str, Any], bg_color: list[float]) -> None:
    """Khôi phục gần đúng trạng thái hiển thị gốc (dùng khi chỉ bật --auto-fullbody không --alpha-matte)."""
    sf_call(page, "window.sfApi.setEnvironment({shadowEnabled: true}, () => done(true));", 8000)
    pp = {"vignetteEnable": True} if info.get("postfx") == "vignette_only_off" else {"enable": True}
    sf_call(page, f"window.sfApi.setPostProcessing({json.dumps(pp)}, () => done(true));", 8000)
    sf_set_bg_color(page, bg_color)


def orbit_eye(target: np.ndarray, r: float, az_rad: float, elev_deg: float) -> list[float]:
    th = math.radians(elev_deg)
    return [
        float(target[0] + r * math.cos(th) * math.cos(az_rad)),
        float(target[1] + r * math.cos(th) * math.sin(az_rad)),
        float(target[2] + r * math.sin(th)),
    ]


def probe_silhouettes(
    page: Page,
    poses: list[tuple[list[float], list[float]]],
    sizes: list[tuple[int, int]],
    settle: float,
) -> list[list[tuple[int, int, int, int] | None]]:
    """Với mỗi (eye,target) và kích thước ảnh: chụp nền đen + nền trắng -> bbox silhouette (alpha>0.1)."""
    shots: dict[tuple[str, int, int], np.ndarray] = {}
    for name, col in (("b", [0.0, 0.0, 0.0]), ("w", [1.0, 1.0, 1.0])):
        sf_set_bg_color(page, col)
        time.sleep(0.3)
        for i, (eye, tgt) in enumerate(poses):
            sf_set_camera(page, eye, tgt, 0.0)
            time.sleep(settle)
            for j, (w, h) in enumerate(sizes):
                shots[(name, i, j)] = sf_grab_rgb(page, w, h)
    out: list[list[tuple[int, int, int, int] | None]] = []
    for i in range(len(poses)):
        row = []
        for j in range(len(sizes)):
            alpha = np.clip(1.0 - (shots[("w", i, j)] - shots[("b", i, j)]).mean(axis=-1) / 255.0, 0.0, 1.0)
            row.append(alpha_bbox(alpha))
        out.append(row)
    return out


def calibrate_fullbody(
    page: Page,
    target0: np.ndarray,
    radius0: float,
    base_az_rad: float,
    fov_deg: float,
    elevations: list[float],
    resolutions: list[tuple[int, int]],
    margin: float,
    settle: float,
    probe_w: int = 384,
) -> dict[str, Any]:
    """
    Auto full-body framing bằng silhouette probing (dual nền đen/trắng):
      A) Zoom rộng 2.2x, 4 góc cardinal 0/90/180/270° @elev 0: căn tâm bbox -> dịch target (đặc biệt target_z)
      B) Quét 8 azimuth × các elevation yêu cầu × mọi tỷ lệ khung: co/giãn base_radius để silhouette nằm trọn
         trong biên `margin` ở cả 4 cạnh.
    """
    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    sizes: list[tuple[int, int]] = []
    for (w, h) in resolutions:
        s = (probe_w, max(64, int(round(probe_w * h / w))))
        if s not in sizes:
            sizes.append(s)
    target = np.asarray(target0, dtype=np.float64).copy()
    r = float(radius0) * 2.2
    cards = (0.0, 90.0, 180.0, 270.0)
    W0, H0 = sizes[0]

    print(f"[AutoFullbody] Bước A: căn tâm với zoom rộng 2.2x (radius={r:.3f}), 4 góc cardinal ...")
    for it in range(6):
        poses = [(orbit_eye(target, r, base_az_rad + math.radians(a), 0.0), target.tolist()) for a in cards]
        bbs = [row[0] for row in probe_silhouettes(page, poses, [sizes[0]], settle)]
        if any(b is None for b in bbs):
            r *= 1.6
            print(f"  [A{it}] không thấy vật thể -> r={r:.3f}")
            continue
        if any(b[0] <= 1 or b[1] <= 1 or b[2] >= W0 - 1 or b[3] >= H0 - 1 for b in bbs):
            r *= 1.4
            print(f"  [A{it}] silhouette chạm biên -> nới r={r:.3f}")
            continue
        scale = 2.0 * r * tan_half / H0
        horiz = np.zeros(3)
        zsum = 0.0
        for (eye, tgt), b in zip(poses, bbs):
            right = np.asarray(compute_camera_matrices(eye, tgt)["c2w_opengl"])[:3, 0]
            off_x = (b[0] + b[2]) * 0.5 - W0 * 0.5
            off_y = (b[1] + b[3]) * 0.5 - H0 * 0.5
            horiz += right * off_x * scale
            zsum += -off_y * scale
        delta = horiz / 2.0
        delta[2] = zsum / 4.0
        target += delta
        print(f"  [A{it}] dịch target {np.round(delta, 4).tolist()} -> target={np.round(target, 4).tolist()}")
        if float(np.linalg.norm(delta)) < 0.004 * r:
            break

    elevs = sorted({max(-85.0, min(85.0, float(e))) for e in elevations} | {0.0})
    azs = [45.0 * k for k in range(8)]
    fit_report: list[dict[str, Any]] = []
    print(f"[AutoFullbody] Bước B: calibrate base_radius (margin={margin * 100:.0f}%) trên {len(azs) * len(elevs)} góc × {len(sizes)} khung ...")
    ratio = 1.0
    for it in range(8):
        metas = [(e, a) for e in elevs for a in azs]
        poses = [(orbit_eye(target, r, base_az_rad + math.radians(a), e), target.tolist()) for e, a in metas]
        res = probe_silhouettes(page, poses, sizes, settle)
        # (1) cân bằng dọc: lề trên nhỏ nhất vs lề dưới nhỏ nhất trên mọi góc -> dịch target_z (chống cắt chân/chân bàn)
        tops, bots = [], []
        for row in res:
            for (w, h), b in zip(sizes, row):
                if b is not None:
                    tops.append(b[1] / h)
                    bots.append(1.0 - b[3] / h)
        s_frac = 0.0
        if tops:
            s_frac = (min(tops) - min(bots)) * 0.5  # >0: nội dung cần dịch LÊN trong ảnh
        ratio = 0.0
        fit_report = []
        for (e, a), row in zip(metas, res):
            for (w, h), b in zip(sizes, row):
                if b is None:
                    continue
                y0 = b[1] - s_frac * h
                y1 = b[3] - s_frac * h
                rx = max(abs(b[0] - w * 0.5), abs(b[2] - w * 0.5)) / ((0.5 - margin) * w)
                ry = max(abs(y0 - h * 0.5), abs(y1 - h * 0.5)) / ((0.5 - margin) * h)
                ratio = max(ratio, rx, ry)
                fit_report.append(
                    {
                        "elev": e, "az": a, "size": [w, h],
                        "margin_left": round(b[0] / w, 4), "margin_top": round(b[1] / h, 4),
                        "margin_right": round(1 - b[2] / w, 4), "margin_bottom": round(1 - b[3] / h, 4),
                    }
                )
        print(f"  [B{it}] r={r:.4f} target_z={target[2]:.4f} vshift={s_frac:+.3f}H fit_ratio(after shift)={ratio:.3f}")
        # tâm dọc: nội dung dịch lên s_frac*H px  <=>  hạ target xuống s_frac * 2 r tan(fov/2)
        target[2] -= s_frac * 2.0 * r * tan_half
        if ratio <= 0.0 or (0.96 <= ratio <= 1.0 and abs(s_frac) < 0.01):
            break
        if not (0.96 <= ratio <= 1.0):
            r *= max(ratio, 0.3) * (1.0 if ratio < 1.0 else 1.01)

    if ratio > 1.0:
        r *= ratio * 1.01  # an toàn: vòng lặp hết lượt mà vẫn chưa đạt
        print(f"  [B*] nới thêm an toàn -> r={r:.4f}")

    return {
        "target": target.tolist(),
        "base_radius": float(r),
        "initial_radius_scale": 2.2,
        "frame_margin": margin,
        "final_fit_ratio": float(ratio),
        "probe_sizes": [list(s) for s in sizes],
        "fit_report": fit_report,
    }


def _composite_on_checker(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Ghép ảnh RGBA lên nền ô vuông xám để xem trước alpha."""
    w, h = size
    rgba = im.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
    tile = 16
    yy, xx = np.mgrid[0:h, 0:w]
    chk = (((yy // tile) + (xx // tile)) % 2).astype(np.uint8)
    bg = np.where(chk[..., None] == 0, 70, 105).astype(np.uint8).repeat(3, axis=2)
    out = Image.fromarray(bg, "RGB")
    out.paste(rgba, (0, 0), rgba)
    return out


def build_preview_grid(image_paths: list[Path], out_path: Path, cols: int = 6, thumb_w: int = 320) -> None:
    """Tạo ảnh tổng hợp (contact sheet) để kiểm tra nhanh các góc chụp."""
    if not image_paths:
        return
    sample = Image.open(image_paths[0])
    aspect = sample.height / max(1, sample.width)
    thumb_h = int(round(thumb_w * aspect))
    rows = math.ceil(len(image_paths) / cols)

    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(grid)

    for idx, p in enumerate(image_paths):
        r, c = divmod(idx, cols)
        with Image.open(p) as im:
            if im.mode == "RGBA":
                rgb = _composite_on_checker(im, (thumb_w, thumb_h))
            else:
                rgb = im.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x0, y0 = c * thumb_w, r * thumb_h
        grid.paste(rgb, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1], outline=(60, 60, 70), width=1)
        draw.rectangle([x0 + 4, y0 + 4, x0 + 135, y0 + 24], fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 7), p.stem, fill=(235, 235, 235))

    grid.save(out_path)


def perform_mouse_drag(page: Page, dx: float, dy: float, steps: int = 18, pause_s: float = 0.015) -> None:
    """Thực hiện thao tác nhấn giữ chuột trái và kéo mượt trên tâm màn hình."""
    vp = page.viewport_size or {"width": 1280, "height": 720}
    cx, cy = vp["width"] * 0.5, vp["height"] * 0.5
    # Đảm bảo điểm bắt đầu và kết thúc nằm gọn trong vùng khung nhìn
    start_x = max(80.0, min(vp["width"] - 80.0, cx - dx * 0.5))
    start_y = max(80.0, min(vp["height"] - 80.0, cy - dy * 0.5))

    page.mouse.move(start_x, start_y)
    page.mouse.down(button="left")
    for i in range(1, steps + 1):
        frac = i / steps
        page.mouse.move(start_x + dx * frac, start_y + dy * frac)
        time.sleep(pause_s)
    page.mouse.up(button="left")


def capture_via_api(
    page: Page,
    uid: str,
    url_original: str,
    out_dir: Path,
    resolutions: list[tuple[int, int]],
    method: str,
    num_azimuth: int,
    elevations: list[float],
    zoom_scales: list[float],
    fov_override: float | None,
    transparent: bool,
    settle_time: float,
    anim_duration: float,
    auto_fullbody: bool = False,
    frame_margin: float = 0.09,
    target_offset: list[float] | None = None,
    alpha_matte: bool = False,
    shadow_alpha_cutoff: float = 0.35,
    matte_settle: float = 0.3,
) -> dict[str, Any]:
    """Chụp đa góc độ & đa độ phân giải thông qua Sketchfab Viewer API."""
    transparent = transparent and not alpha_matte
    html = build_viewer_html(uid=uid, transparent=transparent)

    with LocalHTMLServer(html) as srv:
        init_w, init_h = resolutions[0]
        page.set_viewport_size({"width": init_w, "height": init_h})
        print(f"[Sketchfab] Đang tải mô hình UID={uid} từ {url_original} ...")
        page.goto(f"http://127.0.0.1:{srv.port}/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("window.sfReady === true || window.sfError !== null", timeout=90000)

        err = page.evaluate("window.sfError")
        if err:
            raise RuntimeError(f"Sketchfab Viewer trả về lỗi: {err}")

        # Chờ tải xong texture HD (tối đa 25s)
        print("[Sketchfab] Viewer đã sẵn sàng. Đang bật chất lượng HD, tắt giới hạn góc quay & chờ nạp kết cấu...")
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

        if transparent:
            page.evaluate(
                """() => new Promise(resolve => {
                    window.sfApi.setBackground({ transparent: true }, () => resolve(true));
                })"""
            )

        if fov_override is not None:
            page.evaluate(
                f"() => new Promise(resolve => window.sfApi.setFov({float(fov_override)}, () => resolve(true)))"
            )

        t_wait0 = time.time()
        while time.time() - t_wait0 < 20.0:
            tex_p = float(page.evaluate("window.textureProgress || 0"))
            if tex_p >= 0.99:
                break
            time.sleep(0.4)
        time.sleep(max(1.0, settle_time))

        cam_init = page.evaluate("() => new Promise(res => window.sfApi.getCameraLookAt((e, c) => res(c)))")
        fov_deg = float(page.evaluate("() => new Promise(res => window.sfApi.getFov((e, f) => res(f)))"))

        pos0 = np.asarray(cam_init["position"], dtype=np.float64)
        target0 = np.asarray(cam_init["target"], dtype=np.float64)
        rel0 = pos0 - target0
        base_radius = float(np.linalg.norm(rel0))
        base_azimuth_rad = math.atan2(float(rel0[1]), float(rel0[0]))
        base_elev_deg = math.degrees(math.asin(np.clip(float(rel0[2]) / max(base_radius, 1e-8), -1.0, 1.0)))

        print(
            f"[Camera gốc] target={np.round(target0, 4).tolist()}, "
            f"radius={base_radius:.4f}, elev0={base_elev_deg:.2f}°, fovy={fov_deg:.2f}°"
        )

        # ---- V2: Auto full-body framing / Alpha matte -------------------------------------------
        framing: dict[str, Any] | None = None
        matte_env: dict[str, Any] | None = None
        if alpha_matte or auto_fullbody:
            orig_bg = sample_bg_color(page)
            matte_env = enter_matte_mode(page)
        if auto_fullbody:
            if method != "orbit":
                print("[AutoFullbody] CẢNH BÁO: chỉ hỗ trợ --method orbit, bỏ qua auto-fullbody.")
            else:
                framing = calibrate_fullbody(
                    page, target0, base_radius, base_azimuth_rad, fov_deg,
                    elevations, resolutions, frame_margin, max(0.2, settle_time * 0.5),
                )
                target0 = np.asarray(framing["target"], dtype=np.float64)
                base_radius = float(framing["base_radius"])
        if target_offset:
            target0 = target0 + np.asarray(target_offset, dtype=np.float64)
            print(f"[Target offset] +{target_offset} -> target={np.round(target0, 4).tolist()}")
        if auto_fullbody and not alpha_matte and matte_env is not None:
            exit_matte_mode(page, matte_env, orig_bg)
        if framing is not None or target_offset:
            print(f"[Framing] target={np.round(target0, 4).tolist()}, base_radius={base_radius:.4f}")
            # chuyển camera về tư thế đã căn khung để bắt đầu chụp
            sf_set_camera(page, orbit_eye(target0, base_radius, base_azimuth_rad, 0.0), target0.tolist(), 0.0)

        # Tạo thư mục cho từng độ phân giải
        res_dirs: dict[tuple[int, int], Path] = {}
        res_records: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for w, h in resolutions:
            rdir = out_dir / f"res_{w}x{h}"
            (rdir / "images").mkdir(parents=True, exist_ok=True)
            res_dirs[(w, h)] = rdir
            res_records[(w, h)] = []

        # Xây dựng danh sách các góc chụp
        pose_specs: list[dict[str, Any]] = []
        view_idx = 0
        for z_idx, z_scale in enumerate(zoom_scales):
            for el_idx, elev_deg in enumerate(elevations):
                # Đảo pha nhẹ giữa các vòng góc cao để phủ đều mặt cầu hơn (tốt hơn cho SIFT / MVS)
                az_shift = (180.0 / num_azimuth) * (el_idx % 2)
                for az_idx in range(num_azimuth):
                    az_deg = (360.0 * az_idx / num_azimuth) + az_shift
                    pose_specs.append(
                        {
                            "view_index": view_idx,
                            "zoom_index": z_idx,
                            "zoom_scale": float(z_scale),
                            "elev_index": el_idx,
                            "elev_deg": float(elev_deg),
                            "az_index": az_idx,
                            "az_deg": float(az_deg),
                        }
                    )
                    view_idx += 1

        total_views = len(pose_specs)
        print(
            f"[Kế hoạch] Tổng số góc chụp: {total_views} góc "
            f"({len(zoom_scales)} mức zoom × {len(elevations)} vòng cao độ × {num_azimuth} góc xoay ngang) "
            f"× {len(resolutions)} độ phân giải = {total_views * len(resolutions)} ảnh."
        )

        # Đặt về góc chuẩn trước khi bắt đầu nếu dùng drag
        prev_zoom = zoom_scales[0]
        prev_elev = base_elev_deg

        for spec in pose_specs:
            v_idx = spec["view_index"]
            az_deg = spec["az_deg"]
            elev_deg = max(-85.0, min(85.0, spec["elev_deg"]))
            z_scale = spec["zoom_scale"]

            if method == "orbit":
                phi = base_azimuth_rad + math.radians(az_deg)
                theta = math.radians(elev_deg)
                r = base_radius * z_scale
                eye = [
                    float(target0[0] + r * math.cos(theta) * math.cos(phi)),
                    float(target0[1] + r * math.cos(theta) * math.sin(phi)),
                    float(target0[2] + r * math.sin(theta)),
                ]
                tgt = target0.tolist()
                page.evaluate(
                    f"""() => new Promise(resolve => {{
                        window.sfApi.setCameraLookAt({json.dumps(eye)}, {json.dumps(tgt)}, {anim_duration:.3f}, () => resolve(true));
                    }})"""
                )
                time.sleep(anim_duration + settle_time)
            else:
                # Chế độ kéo chuột thật (mouse drag + mouse wheel)
                vp = page.viewport_size or {"width": init_w, "height": init_h}
                if v_idx == 0:
                    # Căn góc cao ban đầu
                    d_elev = elev_deg - prev_elev
                    if abs(d_elev) > 1.0:
                        perform_mouse_drag(page, dx=0.0, dy=d_elev * 4.5, steps=15)
                        prev_elev = elev_deg
                else:
                    # Lăn chuột nếu đổi mức zoom
                    if abs(z_scale - prev_zoom) > 1e-4:
                        wheel_delta = (z_scale - prev_zoom) * 900.0
                        page.mouse.move(vp["width"] * 0.5, vp["height"] * 0.5)
                        page.mouse.wheel(0, wheel_delta)
                        prev_zoom = z_scale
                        time.sleep(0.25)

                    # Đổi vòng góc cao nếu chuyển sang vòng mới
                    if abs(elev_deg - prev_elev) > 1e-3:
                        d_elev = elev_deg - prev_elev
                        perform_mouse_drag(page, dx=0.0, dy=d_elev * 4.5, steps=16)
                        prev_elev = elev_deg
                        time.sleep(0.2)

                    # Kéo ngang xoay sang góc tiếp theo
                    step_px = -(vp["width"] * 0.62) * (360.0 / num_azimuth) / 180.0
                    perform_mouse_drag(page, dx=step_px, dy=0.0, steps=18)

                time.sleep(settle_time)

            # Đọc lại tọa độ camera chính xác thực tế từ Sketchfab sau khi xoay/kéo
            cam_now = page.evaluate("() => new Promise(res => window.sfApi.getCameraLookAt((e, c) => res(c)))")
            fov_now = float(page.evaluate("() => new Promise(res => window.sfApi.getFov((e, f) => res(f)))"))
            mats = compute_camera_matrices(cam_now["position"], cam_now["target"])

            fname = f"view_{v_idx:03d}_z{spec['zoom_index']}_el{int(round(elev_deg)):+03d}_az{int(round(az_deg)) % 360:03d}.png"

            # Chụp ở tất cả các độ phân giải yêu cầu tại cùng đúng góc nhìn này
            matte_imgs: dict[tuple[int, int], np.ndarray] = {}
            if alpha_matte:
                # Dual-background: nền đen rồi nền trắng ở CÙNG một camera -> RGBA chính xác tới sub-pixel
                blacks: dict[tuple[int, int], np.ndarray] = {}
                sf_set_bg_color(page, [0.0, 0.0, 0.0])
                time.sleep(matte_settle)
                for w, h in resolutions:
                    blacks[(w, h)] = sf_grab_rgb(page, w, h)
                sf_set_bg_color(page, [1.0, 1.0, 1.0])
                time.sleep(matte_settle)
                for w, h in resolutions:
                    whites = sf_grab_rgb(page, w, h)
                    matte_imgs[(w, h)] = dual_background_matte(blacks[(w, h)], whites, shadow_alpha_cutoff)

            for w, h in resolutions:
                img_path = res_dirs[(w, h)] / "images" / fname
                extra: dict[str, Any] = {}
                if alpha_matte:
                    rgba = matte_imgs[(w, h)]
                    Image.fromarray(rgba, "RGBA").save(img_path)
                    bb = alpha_bbox(rgba[..., 3].astype(np.float32) / 255.0)
                    if bb is not None:
                        extra["alpha_margins_lrtb"] = [
                            round(bb[0] / w, 4), round(1 - bb[2] / w, 4),
                            round(bb[1] / h, 4), round(1 - bb[3] / h, 4),
                        ]
                    extra["alpha_zero_fraction"] = round(float((rgba[..., 3] == 0).mean()), 4)
                else:
                    data_url = page.evaluate(
                        f"""() => new Promise((resolve, reject) => {{
                            window.sfApi.getScreenShot({w}, {h}, 'image/png', (err, result) => {{
                                if (err) reject(String(err));
                                else resolve(result);
                            }});
                        }})"""
                    )
                    save_data_url_png(data_url, img_path)

                fovy_rad = math.radians(fov_now)
                fl_y = (h * 0.5) / math.tan(fovy_rad * 0.5)
                fl_x = fl_y
                fovx_rad = 2.0 * math.atan((w * 0.5) / fl_x)

                res_records[(w, h)].append(
                    {
                        "file_path": f"images/{fname}",
                        "view_index": v_idx,
                        "width": w,
                        "height": h,
                        "azimuth_deg": round(az_deg, 3),
                        "elevation_deg": round(elev_deg, 3),
                        "zoom_scale": z_scale,
                        "fovy_deg": fov_now,
                        "fovx_deg": math.degrees(fovx_rad),
                        "fl_x": fl_x,
                        "fl_y": fl_y,
                        "cx": w * 0.5,
                        "cy": h * 0.5,
                        "eye": cam_now["position"],
                        "target": cam_now["target"],
                        "transform_matrix": mats["c2w_opengl"],
                        "c2w_colmap": mats["c2w_colmap"],
                        "w2c_colmap_R": mats["w2c_colmap_R"],
                        "w2c_colmap_t": mats["w2c_colmap_t"],
                        **extra,
                    }
                )

            print(
                f"  [{v_idx + 1:02d}/{total_views:02d}] Đã chụp {fname} "
                f"ở {len(resolutions)} độ phân giải ({', '.join(f'{w}x{h}' for w, h in resolutions)})"
            )

        # Ghi file transforms.json (chuẩn NeRF / 3DGS) & cameras_meta.json + ảnh preview cho từng độ phân giải
        summary_per_res = []
        for w, h in resolutions:
            rdir = res_dirs[(w, h)]
            frames = res_records[(w, h)]
            fovy_rad = math.radians(frames[0]["fovy_deg"])
            fl_y = frames[0]["fl_y"]
            fovx_rad = 2.0 * math.atan((w * 0.5) / fl_y)

            transforms_json = {
                "model_uid": uid,
                "source_url": url_original,
                "camera_model": "OPENCV",
                "w": w,
                "h": h,
                "fl_x": fl_y,
                "fl_y": fl_y,
                "cx": w * 0.5,
                "cy": h * 0.5,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "camera_angle_x": fovx_rad,
                "camera_angle_y": fovy_rad,
                "frames": [
                    {
                        "file_path": fr["file_path"],
                        "transform_matrix": fr["transform_matrix"],
                    }
                    for fr in frames
                ],
            }
            (rdir / "transforms.json").write_text(json.dumps(transforms_json, indent=2), encoding="utf-8")
            (rdir / "cameras_meta.json").write_text(
                json.dumps(
                    {
                        "model_uid": uid,
                        "source_url": url_original,
                        "method": method,
                        "resolution": [w, h],
                        "initial_camera": cam_init,
                        "alpha_matte": bool(alpha_matte),
                        "frames": frames,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            img_files = [rdir / fr["file_path"] for fr in frames]
            preview_path = rdir / "preview_grid.png"
            build_preview_grid(img_files, preview_path, cols=min(6, max(2, num_azimuth // 2)))
            summary_per_res.append(
                {
                    "resolution": f"{w}x{h}",
                    "images_dir": str(rdir / "images"),
                    "transforms_json": str(rdir / "transforms.json"),
                    "preview_grid": str(preview_path),
                    "num_images": len(frames),
                }
            )

        manifest = {
            "model_uid": uid,
            "source_url": url_original,
            "viewer_source": "api",
            "method": method,
            "total_views": total_views,
            "alpha_matte": bool(alpha_matte),
            "auto_fullbody": bool(framing is not None),
            "target_offset": target_offset,
            "framing": ({k: v for k, v in framing.items() if k != "fit_report"} if framing else None),
            "resolutions": summary_per_res,
        }
        if framing:
            (out_dir / "framing_calibration.json").write_text(json.dumps(framing, indent=2), encoding="utf-8")
        (out_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


def capture_via_webpage_direct(
    page: Page,
    url_original: str,
    out_dir: Path,
    resolutions: list[tuple[int, int]],
    num_azimuth: int,
    elevations: list[float],
    settle_time: float,
) -> dict[str, Any]:
    """Chế độ dự phòng: Mở trực tiếp trang viewer Sketchfab, ẩn UI bằng CSS và kéo chuột chụp canvas."""
    uid, _ = extract_model_info(url_original)
    embed_url = (
        f"https://sketchfab.com/models/{uid}/embed"
        "?autostart=1&preload=1&ui_annotations=0&annotations_visible=0"
        "&ui_controls=0&ui_infos=0&ui_stop=0&ui_watermark=0&ui_watermark_link=0"
        "&ui_hint=0&ui_ar=0&ui_help=0&ui_settings=0&ui_vr=0&ui_fullscreen=0&ui_inspector=0&dnt=1"
    )
    canvas_sel = "canvas.canvas, canvas:not(#qrCode)"

    init_w, init_h = resolutions[0]
    page.set_viewport_size({"width": init_w, "height": init_h})
    print(f"[Webpage] Đang mở trực tiếp trang viewer: {embed_url} (từ {url_original}) ...")
    page.goto(embed_url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(canvas_sel, state="visible", timeout=60000)
    page.wait_for_selector("main.viewer.model-loaded", state="attached", timeout=60000)
    time.sleep(5.0)

    # Ẩn toàn bộ nút bấm giao diện, tiêu đề, watermark, banner và nhãn annotation trên trang Sketchfab
    page.add_style_tag(
        content="""
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        .titlebar, .landing, .overlays, .osd, .fallback, .embed-sandboxed-message,
        .loading-container, .hotspots, .hotspot, .hotspot-content, .gui, .gui-wrap,
        .annotation, .annotations-autopilot, .controls, .general-controls,
        .model-name, .watermark, .overlay-link, .viewer-hint, .main-progress-display,
        header, footer, aside, [class*="Popup"], [class*="Banner"] {
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }
        """
    )
    time.sleep(1.5)

    summary_per_res = []
    for w, h in resolutions:
        rdir = out_dir / f"res_{w}x{h}"
        (rdir / "images").mkdir(parents=True, exist_ok=True)
        page.set_viewport_size({"width": w, "height": h})
        time.sleep(1.2)

        saved: list[Path] = []
        v_idx = 0
        prev_elev = 0.0
        for elev_deg in elevations:
            d_elev = elev_deg - prev_elev
            if abs(d_elev) > 1.0:
                perform_mouse_drag(page, dx=0.0, dy=d_elev * 4.0, steps=16)
                prev_elev = elev_deg
                time.sleep(0.3)

            for az_idx in range(num_azimuth):
                if az_idx > 0:
                    step_px = -(w * 0.62) * (360.0 / num_azimuth) / 180.0
                    perform_mouse_drag(page, dx=step_px, dy=0.0, steps=18)
                time.sleep(settle_time)
                fname = f"view_{v_idx:03d}_el{int(round(elev_deg)):+03d}_az{int(round(360 * az_idx / num_azimuth)):03d}.png"
                out_img = rdir / "images" / fname
                page.screenshot(path=str(out_img), type="png")
                saved.append(out_img)
                v_idx += 1
                print(f"  [{w}x{h}] Đã chụp {fname}")

        preview_path = rdir / "preview_grid.png"
        build_preview_grid(saved, preview_path, cols=min(6, max(2, num_azimuth // 2)))
        summary_per_res.append(
            {
                "resolution": f"{w}x{h}",
                "images_dir": str(rdir / "images"),
                "preview_grid": str(preview_path),
                "num_images": len(saved),
            }
        )

    manifest = {
        "source_url": url_original,
        "viewer_source": "webpage",
        "resolutions": summary_per_res,
    }
    (out_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tự động mở link Sketchfab, xoay/kéo mô hình 3D và chụp ảnh đa góc độ, đa độ phân giải."
    )
    p.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Link mô hình 3D trên Sketchfab (mặc định: Eastern Female Swordsman).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Thư mục lưu ảnh đầu ra (mặc định: ./output/<tên_model>).",
    )
    p.add_argument(
        "--resolutions",
        nargs="+",
        default=["800x600", "1280x720", "1920x1080"],
        help="Danh sách các độ phân giải cần chụp, cách nhau bởi dấu cách (VD: 800x600 1280x720 1920x1080).",
    )
    p.add_argument(
        "--num-azimuth",
        type=int,
        default=12,
        help="Số góc xoay ngang trên mỗi vòng 360 độ (VD: 12 = mỗi bước 30°, 24 = mỗi bước 15°).",
    )
    p.add_argument(
        "--elevations",
        nargs="+",
        type=float,
        default=[-5.0, 15.0, 35.0],
        help="Danh sách các góc cao độ tính bằng độ (VD: -5 15 35).",
    )
    p.add_argument(
        "--zoom-scales",
        nargs="+",
        type=float,
        default=[1.0, 0.75],
        help="Hệ số khoảng cách camera so với ban đầu (1.0 = toàn cảnh, 0.75 = zoom gần chi tiết).",
    )
    p.add_argument(
        "--method",
        choices=["orbit", "drag"],
        default="orbit",
        help="orbit: xoay quỹ đạo cầu chuẩn xác (khuyên dùng cho COLMAP/3DGS); drag: giả lập nhấn kéo chuột.",
    )
    p.add_argument(
        "--viewer-source",
        choices=["api", "webpage"],
        default="api",
        help="api: dùng Sketchfab Viewer API sạch 100%% UI/annotations; webpage: mở trang gốc và kéo chuột.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Ghi đè góc mở camera dọc FOV (độ), ví dụ 35. Để trống để giữ nguyên FOV gốc của tác giả.",
    )
    p.add_argument(
        "--transparent",
        action="store_true",
        help="Bật nền trong suốt (PNG alpha) nếu mô hình/tài khoản hỗ trợ.",
    )
    p.add_argument(
        "--settle-time",
        type=float,
        default=0.35,
        help="Thời gian nghỉ (giây) sau mỗi lần xoay để khung hình ổn định nét hoàn toàn trước khi chụp.",
    )
    p.add_argument(
        "--anim-duration",
        type=float,
        default=0.15,
        help="Thời gian chuyển động mượt giữa hai góc camera (giây) ở chế độ orbit.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Chạy ngầm không hiện cửa sổ trình duyệt (mặc định mở cửa sổ trực quan để quan sát).",
    )
    p.add_argument(
        "--browser-channel",
        type=str,
        default="",
        help="Chỉ định trình duyệt ('chrome' hoặc 'msedge'). Mặc định tự chọn Chrome hoặc Edge có sẵn.",
    )
    p.add_argument(
        "--auto-fullbody",
        action="store_true",
        help="Tự động căn khung 100%% toàn thân/cảnh (silhouette probing nền đen/trắng, zoom rộng 2.2x, 4 góc cardinal): "
        "dịch target (đặc biệt target_z) & calibrate base_radius để không bị cắt chân/chân bàn/đỉnh cây.",
    )
    p.add_argument(
        "--frame-margin",
        type=float,
        default=0.09,
        help="Biên an toàn mỗi cạnh khung hình khi --auto-fullbody (0.09 = 9%%; khuyên dùng 0.08-0.10).",
    )
    p.add_argument(
        "--target-offset",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=None,
        help="Dịch tâm quỹ đạo (target) thêm dx dy dz theo hệ toạ độ Sketchfab (Z-up), áp dụng sau auto-fullbody.",
    )
    p.add_argument(
        "--alpha-matte",
        action="store_true",
        help="Xuất ảnh PNG RGBA 4 kênh bằng Dual-Background Matting (nền đen + nền trắng): "
        "alpha = 1 - mean(I_white - I_black)/255, RGB = I_black / alpha.",
    )
    p.add_argument(
        "--shadow-alpha-cutoff",
        type=float,
        default=0.35,
        help="Với --alpha-matte: xoá vùng alpha < cutoff tách rời khỏi vật thể (bóng đổ sàn); 0 = tắt. "
        "Viền tóc/hoa bám sát thân mô hình được giữ nguyên.",
    )
    p.add_argument(
        "--matte-settle",
        type=float,
        default=0.3,
        help="Thời gian chờ (giây) sau mỗi lần đổi màu nền khi matting.",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    uid, slug = extract_model_info(args.url)
    resolutions = parse_resolutions(args.resolutions)

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (script_dir / "output" / slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" SKETCHFAB MULTI-ANGLE & MULTI-RESOLUTION AUTO CAPTURE")
    print("=" * 72)
    print(f" - Link gốc       : {args.url}")
    print(f" - Model UID      : {uid}")
    print(f" - Chế độ mở      : {args.viewer_source} | Điều khiển: {args.method}")
    print(f" - Độ phân giải   : {', '.join(f'{w}x{h}' for w, h in resolutions)}")
    print(f" - Thư mục lưu    : {out_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    with sync_playwright() as pw:
        browser = launch_browser(pw, headless=args.headless, channel=args.browser_channel or None)
        try:
            context = browser.new_context(device_scale_factor=1)
            page = context.new_page()

            if args.viewer_source == "api":
                manifest = capture_via_api(
                    page=page,
                    uid=uid,
                    url_original=args.url,
                    out_dir=out_dir,
                    resolutions=resolutions,
                    method=args.method,
                    num_azimuth=args.num_azimuth,
                    elevations=args.elevations,
                    zoom_scales=args.zoom_scales,
                    fov_override=args.fov,
                    transparent=args.transparent,
                    settle_time=args.settle_time,
                    anim_duration=args.anim_duration,
                    auto_fullbody=args.auto_fullbody,
                    frame_margin=args.frame_margin,
                    target_offset=args.target_offset,
                    alpha_matte=args.alpha_matte,
                    shadow_alpha_cutoff=args.shadow_alpha_cutoff,
                    matte_settle=args.matte_settle,
                )
            else:
                if args.auto_fullbody or args.alpha_matte:
                    print("[CẢNH BÁO] --auto-fullbody/--alpha-matte chỉ hỗ trợ --viewer-source api; bỏ qua.")
                manifest = capture_via_webpage_direct(
                    page=page,
                    url_original=args.url,
                    out_dir=out_dir,
                    resolutions=resolutions,
                    num_azimuth=args.num_azimuth,
                    elevations=args.elevations,
                    settle_time=args.settle_time,
                )
        finally:
            browser.close()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"[HOÀN TẤT] Thời gian chạy: {elapsed:.1f}s")
    for r in manifest["resolutions"]:
        print(f"  + Độ phân giải {r['resolution']:>9s}: {r['num_images']} ảnh -> {r['images_dir']}")
        print(f"    Bảng xem trước         : {r['preview_grid']}")
    print(f"  + File tổng hợp manifest : {out_dir / 'capture_manifest.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
