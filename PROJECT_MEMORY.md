# Project Memory — Đồ án Tái dựng 3D & Thuật toán DCA (`c:\doan`)

**Cập nhật lần cuối:** 2026-11-17 (Hoàn tất Pha 0, Pha 1 & Pha 2 — Sẵn sàng cho Pha 3)

---

## 1. Thông tin chung & Mục tiêu dự án
- **Mục tiêu của người dùng:** Nghiên cứu thuật toán **DCA** của thầy hướng dẫn và áp dụng vào mã nguồn để xây dựng **bản đồ/mô hình 3D hoàn chỉnh** (mục tiêu cuối: **lon bò húc**; bộ dữ liệu thực nghiệm đang có sẵn: **12 ảnh 360° con gà đèn ngủ** tại `c:\doan\data\ga_den_ngu`).
- **Giao tiếp & quy ước làm việc:**
  - Người dùng yêu cầu **báo cáo tiến độ `%` thường xuyên** theo từng mốc khi thực hiện các tác vụ nhiều bước.
  - Tài liệu học tập trực quan đã tạo cho người dùng: `c:\doan\huong_dan_3d_dca.html` (và bản artifact tương ứng).
  - Kế hoạch triển khai 5 Pha: `c:\doan\KE_HOACH_DCA_3D_BO_HUC.md`.

---

## 2. Trạng thái Codebase & Dữ liệu hiện tại
- **Thư mục mã nguồn:** `c:\doan\reconstruction_dca_3d` (đã commit Pha 0 & Pha 1 tại `9e6863e`, tiếp tục commit chốt Pha 2 trên nhánh `main` — working tree sạch, sẵn sàng cho Pha 3).
- **Code đã nâng cấp ở Pha 1 & Pha 2 (vẫn giữ nguyên mặc định `--optimizer adam` và `star` mode tương thích 100% với 6/6 test):**
  - Thêm mới: `code/reconstruction/dca.py`, `code/run_dca_study.py`, `code/check_ring_init.py`.
  - Đã cập nhật ở Pha 1: `code/reconstruction/point_terms.py`, `losses.py`, `engine.py`, `contact.py` (`hessian_data`), `axis_completion.py`, `completion.py`, `hidden_completion.py`, `code/reconstruct_images.py` (`--optimizer dca`, `--cauchy-scales`).
  - Đã cập nhật ở Pha 2:
    - `code/reconstruction/inputs.py`: `_candidate_pairs` + `feature_pairs(..., pair_mode='ring'|'all'|'star', ring_window=2)` ghép vòng tròn lân cận + seed RANSAC từng cặp.
    - `code/reconstruction/initialization.py`: `initialize(..., init_mode='ring', ring_direction='auto')` lan truyền pose theo đồ thị cạnh SIFT (Prim/BFS nối tiếp, lọc trung vị tỷ lệ baseline), tự nhận diện chiều quay vòng tròn (`_ring_sign_from_calibrated`), lấp các góc thiếu bằng `_place_ring_cameras` theo tỷ lệ chiều cao mask, và bổ sung giả thuyết `uniform_ring_*` trong `camera_hypotheses`.
    - `code/reconstruction/engine.py` & `records.py`: `_build_feature_tracks` gộp track đa góc nhìn qua Union-Find không xung đột view và neo `ray_anchors` theo camera quan sát `v_ref` của từng track (không cứng hóa view 0).
    - `code/reconstruction/rotations.py` & `code/reconstruct_images.py`: thêm `--init-ring`, `--match-mode`, `--ring-window`, `--ring-direction`, vẽ `feature_matches.png` cho >2 ảnh, và thêm `up=(0., 0., 1.)` cho `look_at` khi xuất `model_views.png` (fix góc `90°/270°` bị xoay ngang).
- **Thư mục kết quả thực nghiệm đã lưu (`c:\doan\outputs\`):**
  - `baseline_raw_inputs/` (Baseline 1 Adam trên `raw_inputs/`)
  - `baseline_chicken_3views/` (Baseline 2 Adam trên 3 ảnh mặt trước con gà)
  - `dca_study/` (đối chứng `dca_study.json` + ảnh so sánh `Adam vs L-BFGS-B vs Joint DCA` sạch & bơm 25% outlier)
  - `dca_chicken_3views/` (`Joint DCA + Homotopy` trên 3 ảnh mặt trước con gà)
  - `dca_chicken_12views_ring/` (Pha 2: `Joint DCA + --init-ring` trên đủ 12 ảnh 360° `ga_den_ngu`, `361.9s`)
- **Môi trường Python đã dựng xong (Pha 0):**
  - Virtualenv: `c:\doan\.venv` (Python `3.12.13` qua `uv`).
  - Đã cài đủ 15 gói theo `code/requirements-reconstruction.txt`: `numpy==2.5.4`, `scipy==1.18.1`, `pillow==13.0.0`, `matplotlib==3.11.2`, `opencv-python-headless==4.10.0.84`, `ipctk==1.6.1`, `libigl==2.6.3`.
  - **5/5 bộ test gốc PASS:** `check_point_uncertainty.py` (`4.34e-14`), `check_smooth_fields.py` (`1.72e-10`), `check_contact.py` (`2.26e-09`), `check_contact_integration.py` (`2.09e-11`), `check_hidden_completion.py`.
- **Bộ dữ liệu đầu vào con gà đèn ngủ (`c:\doan\data\ga_den_ngu`):**
  - 12 ảnh + 12 mask `512×512` (cắt khung vuông đồng nhất `656×656`, căn tâm, tách nền tự động bằng `RMBG-1.4 + GrabCut` trên GPU RTX 3050).
  - SIFT trên `images/` tại `384px`: mặt trước ghép được (`1↔2`: 8 inliers, `1↔3`: 9 inliers, `2↔3`: 17 inliers, `4↔5`: 9 inliers); mặt lưng/hông trơn (`5..12`) chỉ đạt `0–7` inliers/cặp (`17–39` keypoints/ảnh).
- **Kết quả Baseline Pha 0 (`Adam-Armijo` + `L2 squared point loss`, chưa có DCA):**
  1. **Baseline 1 (`c:\doan\outputs\baseline_raw_inputs`, `raw_inputs/` 2 ảnh, `211.9s`):**
     - `56` tracks / `112` obs; Mask IoU: `96.89%`, `97.84%`; SIFT RMS: `1.433 px` (median `0.255 px`); `0` self-intersections (`libigl`); `0` visibility violations. Góc sau `90°/180°/270°` vẫn lồi lõm.
  2. **Baseline 2 (`c:\doan\outputs\baseline_chicken_3views`, `ga_den_ngu` 3 ảnh mặt trước `view_01..03`, `184.7s`):**
     - `10` tracks / `27` obs; Mask IoU tụt còn `75.58% / 82.34% / 81.27%`; SIFT RMS vọt lên `25.543 px` (`max=100.49 px`), `8` visibility violations, `55` contact pairs, `signed_to_hull_volume_ratio=0.131`, lưới bị xé thành gai nhọn (vẫn giữ `0` self-intersections nhờ IPC).
     - **Nguyên nhân đã xác định rõ:** Có 2 track outlier nặng (`track 9`: ghép nhầm mào ở ảnh 1 sang cánh trái ở ảnh 3, sai số `100.49 px`; `track 8`: ghép nhầm chân phải sang chân trái, sai số `50.54 px`). Do code gốc dùng bình phương L2 thuần (chưa có **Cauchy Robust Loss** của DCA), 2 outlier này phá hỏng hoàn toàn việc chọn giả thuyết camera (`centred_crop_focal_x2`) và kéo méo lưới!
- **Điểm mạnh giữ lại từ code gốc:**
  - Cập nhật xoay camera trên bản đồ chuẩn đại số Lie $g \leftarrow \exp([\omega]_\times) g$ (`engine.py`, `rotations.py`).
  - Ràng buộc tầm nhìn chạm mặt đầu (`viscost` / First-hit visibility trong `losses.py`).
  - Chống tự cắt lưới 3 lớp bằng `ipctk` (Barrier + CCD `safe_fraction`) và `libigl` (`find_self_intersections`) trong `contact.py`.

---

## 3. Hiểu biết về Thuật toán DCA của Thầy (từ PDF 24 trang đã đọc)
- **Tài liệu:** *Single-view recovery of road geometry: a family of projections uniform in the inverse viewing distance, its orthographic limit, and a nonconvex variational problem* (25/10/2026, 24 trang).
- **Phạm vi bản thân bài PDF:** Mô hình hình học đường phố từ 1 ảnh (mặt đường phẳng $\Pi$, 3 điểm tụ trực giao, vạch kẻ đường, tam giác $OAB$). Bản thân mô hình đường phố 1 ảnh **không tự dựng được lon bò húc**.
- **Thuật toán DCA cốt lõi cần trích xuất từ Mục 11 & Phụ lục B:**
  1. **Cauchy Robust Loss $\rightarrow$ IRLS weights:** $F(\Theta) = \frac{1}{2}\sum_i c^2 \log(1 + r_i^2/c^2)$ với trọng số $w_i = \frac{1}{1 + r_i^2/c^2}$ giúp loại bỏ điểm SIFT nhiễu giả (outliers do vỏ lon nhôm bóng lóa sáng).
  2. **Phân cực cục bộ trên $SO(3)$:** Cập nhật $g \leftarrow \exp([\omega]_\times) g$ trong không gian tiếp tuyến quanh đơn vị.
  3. **Bài toán con lồi (Eq. 34):** $\min_d \frac{1}{2}\sum_i w_i (r_i + J_i d)^2 + \frac{1}{2}\mu \|d\|_D^2$, $D = \operatorname{diag}(J^\top W J)$.
  4. **Điều chỉnh $\mu$ thích nghi:** Từ chối / vi phạm khả thi $\rightarrow \mu \times 4$; chấp nhận ($F_{\text{new}} \le F - 10^{-4}\mu \|d\|_D^2$) $\rightarrow \mu \div 3$.
  5. **Graduated Non-Convexity (Homotopy):** Giảm dần $c = 30 \to 10 \to 3$.
  6. **Quy tắc bắt buộc:** Phải dùng **Joint DCA** (tối ưu đồng thời Camera & Mesh), **không** dùng Block-cyclic DCA (đã thất bại `0/40` trong Bảng 6 của PDF).

---

## 4. Môi trường & Cấu hình Phần cứng đã kiểm tra (2026-11-17)
- **OS:** Windows 11 Home Single Language 64-bit.
- **CPU:** AMD Ryzen 5 7640HS (6 cores / 12 threads).
- **RAM:** `15.29 GB` tổng (khi chạy nhớ đóng app nền để trống ít nhất **3.5 – 4.5 GB** cho bản đầy đủ Pha 2–4; Peak RAM ước tính bản chuẩn là ~0.5–1.2 GB, bản nâng cấp đã lọc AABB là ~1.5–2.8 GB).
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop (`6144 MiB` VRAM). Thuật toán chạy trên CPU; GPU chỉ dùng nếu cần chạy thêm AI tách mask ngoài.
- **Ổ cứng:** `C:` trống `93.4 GB`, `D:` `23.6 GB`, `E:` `27.3 GB`.
- **Python hiện có:**
  - `uv.exe` (`0.12.2`) đã cài tại `C:\Users\viet cong\.local\bin\uv.exe`.
  - Môi trường chạy chính của dự án: `c:\doan\.venv` (Python `3.12.13` với đủ 15 gói chuẩn từ `requirements-reconstruction.txt`).
  - Python `3.11.9` hệ thống có `torch 2.5.1+cu121`, `transformers`, `cv2`, `onnxruntime` (dùng cho bước tiền xử lý `RMBG-1.4` trên GPU).

---

## 5. Tiến độ & Bước tiếp theo
- **Pha 0 (ĐÃ HOÀN TẤT 100%):** Đổi tên thư mục sang `reconstruction_dca_3d`, dựng `.venv` Python `3.12.13`, chạy 5/5 `check_*.py` PASS, chạy xong Baseline 1 (`raw_inputs/`) và Baseline 2 (`ga_den_ngu` 3 ảnh mặt trước).
- **Pha 1 (ĐÃ HOÀN TẤT 100%):**
  - Đã cài **Cauchy Robust Loss + trọng số IRLS** (`point_terms.py`, `losses.py`, `engine.py`) với sai số gradient FD `3.36e-14`; bảo toàn 100% tương thích 5/5 `check_*.py`.
  - Đã viết bộ giải **Joint DCA** (`code/reconstruction/dca.py`): tối ưu đồng thời `(X, C, g, lf, pp, h)`, phân cực $SO(3)$, giải bài toán con Eq. 34 bằng Matrix-Free Jacobi-PCG, tích hợp Hessian thưa của `ipctk` (`contact.py:hessian_data`, đối xứng `6.332e-16`), điều chỉnh $\mu$ thích nghi ($\times 4 / \div 3$), Homotopy $30 \to 10 \to 3$ px, giảm chấn grazing visibility (`grazing_boost` / `view_boost`) và phanh CCD.
  - Đã nối `--optimizer dca` vào `reconstruct_images.py`, `axis_completion.py`, `completion.py`, `hidden_completion.py`.
  - **Kết quả đối chứng `run_dca_study.py` (`c:\doan\outputs\dca_study\dca_study.json`):**
    - `raw_inputs` sạch + IPC: `Adam` (`98.40%` IoU, `0.799 px`, `37.39s`) vs `Joint DCA` (`97.44%` IoU, `0.815 px`, `17.90s` — nhanh hơn `2.1x`, `0` tự cắt).
    - `raw_inputs` **bơm 25% outlier SIFT (45–95 px)** + IPC: `Adam` vọt sai số điểm sạch lên **`5.428 px`** (`IoU 95.47%`), trong khi **`Joint DCA` giữ nguyên `cleanRMS = 0.973 px` (`IoU 97.57%`)** — chính xác hơn **5.6x**.
  - **Kết quả trên 3 ảnh mặt trước con gà (`c:\doan\outputs\dca_chicken_3views`, `120.2s`):**
    - Mask IoU tăng từ `75.58% / 82.34% / 81.27%` (Baseline 2 Adam) lên **`94.14% / 93.73% / 94.44%`** (**+14.37% IoU**), `out = 0.431 px`, `in = 0.226 px`, `median = 1.185 px`, **`fraction_within_3px = 92.59%`** (25/27 obs, 2 outlier bị hạ trọng số xuống `0.0006`), `visibility_violations = 0`, `intersecting_pairs = 0`.
- **Pha 2 (ĐÃ HOÀN TẤT 100%):**
  - Đã nâng cấp ghép SIFT vòng tròn 360° (`inputs.py`), lan truyền pose theo đồ thị cạnh + lấp vòng tròn 360° (`initialization.py`), gộp track Union-Find + neo `ray_anchors` đa góc nhìn (`engine.py`, `records.py`), và CLI `--init-ring` (`reconstruct_images.py`).
  - **Kiểm thử Pha 2:** 5/5 test gốc PASS + `check_ring_init.py` PASS (lan truyền vòng tròn 8 góc `step=45.0 deg`, 18 track chỉ ở mặt sau, sai số FD gradient `1.28e-09`, và fallback `uniform_ring`).
  - **Kết quả thực nghiệm đủ 12 ảnh 360° con gà (`c:\doan\outputs\dca_chicken_12views_ring`, `361.9s`):**
    - 12 camera trải đều 360° (`azim = 0.0° .. 336.7°`), **Mask IoU trung bình `90.47%`** (`88.26% – 93.94%`), `out = 1.203 px`, `in = 0.617 px`, `median_reprojection = 0.611 px`, **`fraction_within_3px = 91.67%`**, `visibility_violations = 0`, `intersecting_pairs = 0` (`libigl`), `signed_to_hull_volume_ratio = 0.318` (khối 3D kín `0.60 × 0.64 × 1.00`, hết sụp mặt sau như bản 3 góc).
    - **Ghi chú kỹ thuật chuyển sang Pha 3:** Ở mức `162` đỉnh và `smooth_weight = 6.78e-10` (sau `/width^2` trong `weak_priors.py`), bề mặt còn nhiều nếp gợn đa giác thô và `contact_pairs = 72` — đây là đúng hạng mục Pha 3 (`subdivide 162 → 642 → 2562`, tăng mẫu biên mask, lọc AABB `losses.py`, đa luồng `ipctk`, và tinh chỉnh `smooth_weight`).
- **Bước tiếp theo — Pha 3:**
  - Subdivide đa phân giải (`162 → 642 → 2562` đỉnh), mở đa luồng `ipctk`, lọc AABB 2D trong `losses.py` (+ `dca.py`), tăng mật độ mẫu biên mask trong `engine.py`, và tinh chỉnh `smooth_weight`.
- **Pha sau:**
  - **Pha 4:** Viết `texture.py` (bake màu RGB lên `.obj`/`.glb`) và chuẩn hóa kích thước thật (`--reference-height-mm`).
