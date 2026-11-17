# Project Memory — Đồ án Tái dựng 3D & Thuật toán DCA (`c:\doan`)

**Cập nhật lần cuối:** 2026-11-17

---

## 1. Thông tin chung & Mục tiêu dự án
- **Mục tiêu của người dùng:** Nghiên cứu thuật toán **DCA** của thầy hướng dẫn và áp dụng vào mã nguồn để xây dựng **bản đồ/mô hình 3D cho một lon bò húc**.
- **Giao tiếp & quy ước làm việc:**
  - Người dùng yêu cầu **báo cáo tiến độ `%` thường xuyên** theo từng mốc khi thực hiện các tác vụ nhiều bước.
  - Tài liệu học tập trực quan đã tạo cho người dùng: `c:\doan\huong_dan_3d_dca.html` (và bản artifact tương ứng).
  - Kế hoạch triển khai đã chốt: `c:\doan\KE_HOACH_DCA_3D_BO_HUC.md`.

---

## 2. Trạng thái Codebase hiện tại (`c:\doan\reconstruction_generic_code_20261114 (1)`)
- **Toàn bộ file code `.py` hiện còn nguyên bản (chưa chỉnh sửa).**
- **Bản chất code gốc:**
  - Đây là gói phân tích lỗi (`failure-analysis`) tái dựng lưới 3D tổng quát từ ảnh 2D (xuất phát từ Icosphere 42 $\rightarrow$ 162 đỉnh / 320 tam giác), chạy trên **CPU**.
  - **CẢNH BÁO QUAN TRỌNG:** Code gốc **KHÔNG PHẢI** thuật toán DCA (`START_HERE.txt`, `README_CONTACT.txt`, `LBFGS_CHECK_SPEC.txt` đều ghi rõ *"This is NOT a DCA implementation"*). Bộ giải hiện tại trong `code/reconstruction/optimizer.py` là **Adam-preconditioned Armijo descent + IPC CCD**, và bản đối chứng trong `lbfgs.py` là **L-BFGS-B**.
- **Điểm mạnh giữ lại từ code gốc:**
  - Cập nhật xoay camera trên bản đồ chuẩn đại số Lie $g \leftarrow \exp([\omega]_\times) g$ (`engine.py`, `rotations.py`).
  - Ràng buộc tầm nhìn chạm mặt đầu (`viscost` / First-hit visibility trong `losses.py`).
  - Chống tự cắt lưới 3 lớp bằng `ipctk` (Barrier + CCD `safe_fraction`) và `libigl` (`find_self_intersections`) trong `contact.py`.
- **Giới hạn của code gốc đối với bài toán lon bò húc:**
  1. `inputs.py` (`feature_pairs`) chỉ ghép SIFT giữa `view 0` với các ảnh khác $\rightarrow$ không ghép được ảnh vòng tròn 360° mặt sau.
  2. Chỉ có 2 ảnh $\rightarrow$ ~106/162 đỉnh ở vùng khuất bị phồng/lõm (`result.json` mẫu lon đạt Mask IoU 98.93% & 98.82%, SIFT RMS 0.304 px, nhưng mặt sau vẫn móp).
  3. Lưới 162 đỉnh quá thô; `geometry.py` (`export_mesh`) chỉ xuất lưới đơn sắc, **không có màu/texture** và **không có kích thước thật** (`metric_scale_known: False`).
  4. `losses.py` (`silhouette_terms`) broadcast `K × M × T`, và `contact.py:20` khóa `ipctk.set_num_threads(1)`.

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
  - `uv.exe` đã cài tại `C:\Users\viet cong\.local\bin\uv.exe`.
  - Python `3.12.13` (trùng phiên bản chuẩn của `requirements-reconstruction.txt`) đã có tại `C:\Users\viet cong\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe`, nhưng **chưa cài thư viện** (`numpy`, `scipy`, `PIL`, `matplotlib`, `cv2`, `ipctk`, `igl` đều `False`).
  - Python `3.11.9` mặc định có `numpy`, `scipy`, `PIL`, `matplotlib`, `cv2`, nhưng thiếu `ipctk` và `igl`.

---

## 5. Quyết định & Bước tiếp theo khi mở phiên mới
- **Kế hoạch đã chốt (5 Pha trong `c:\doan\KE_HOACH_DCA_3D_BO_HUC.md`):**
  - **Pha 0:** Đổi tên thư mục (nếu người dùng đồng ý), tạo `.venv` Python 3.12 qua `uv`, cài `requirements-reconstruction.txt`, chạy `check_*.py` & baseline `reconstruct_images.py`.
  - **Pha 1:** Cài đặt `dca.py` (Joint DCA + Cauchy IRLS + Homotopy + giữ CCD) và chạy đối chứng với Adam / L-BFGS-B.
  - **Pha 2:** Nâng cấp ghép SIFT vòng tròn 360° & khởi tạo đa góc nhìn cho 8–12 ảnh lon bò húc.
  - **Pha 3:** Subdivide đa phân giải (162 → 642 → 2562 đỉnh), mở đa luồng `ipctk`, tối ưu `losses.py` (lọc AABB).
  - **Pha 4:** Viết `texture.py` (bake màu nhãn lon lên `.obj`/`.glb`) và chuẩn hóa kích thước thật (`--reference-height-mm`).
- **Thông tin cần hỏi lại / xác nhận khi bắt đầu Pha 0 ở phiên sau:**
  1. Xác nhận đổi tên thư mục `reconstruction_generic_code_20261114 (1)` $\rightarrow$ `reconstruction_dca_3d`.
  2. Bộ ảnh chụp thật lon bò húc (8–12 góc) đã sẵn sàng chưa hay bắt đầu trên `raw_inputs/` trước.
