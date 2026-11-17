"""Spec: complete six-case scientific report with labeled geometry and raw errors.

Each row links its own observations, cameras, mesh and detailed reprojections.
Different training objectives are not ranked by total objective value.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reconstruction.ablation import case_grid
from reconstruction.raster import render_mesh
from reconstruction.rotations import look_at


def load_results(root):
    rows=[]
    for case in case_grid():
        path=Path(root)/case['label']/'ablation.json'
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['label']!=case['label']:raise ValueError(f'Mislabeled result: {path}')
        rows.append(row)
    return rows


def geometry_figure(root,rows):
    payloads=[json.loads((root/r['label']/'result.json').read_text(encoding='utf-8')) for r in rows]
    radius=max(np.max(np.linalg.norm(np.array(p['display_vertices']),axis=1)) for p in payloads)
    C=np.array([1.,-1.,np.sqrt(2.)])*max(2.,2*radius)
    fig,axs=plt.subplots(2,3,figsize=(14,10))
    for ax,row,payload in zip(axs.ravel(),rows,payloads):
        image=render_mesh(np.array(payload['display_vertices']),np.array(payload['faces']),C,look_at(C,np.zeros(3)),
                          640,[320,320],(640,640))
        image.save(root/row['label']/'common_display_view.png')
        ax.imshow(image);ax.axis('off')
        ax.set_title(f'Camera prior {row["camera_coefficient"]:g}; point tolerance {row["point_tolerance_pixels"]:g} px\n'
                     f'Depth/width {row["depth_to_width"]:.2f}; transverse ratio {row["transverse_ratio"]:.2f}\n'
                     f'RMS {row["feature_rms_px"]:.3f} px; min IoU {min(row["mask_IoU"])*100:.2f}%',fontsize=9)
    fig.suptitle(f'Same free mesh, seed, fixed axis and {rows[0]["observations"]} observations; shape epsilon = 0.001\n'
                 'Novel views at one common display camera; height = 1 relative unit; no measured 3D truth',fontsize=12)
    fig.subplots_adjust(top=.86,bottom=.02,left=.02,right=.98,hspace=.30,wspace=.12)
    fig.savefig(root/'geometry_overview.png',dpi=145);plt.close(fig)


def diagnostics_figure(root,rows):
    fig,axs=plt.subplots(2,2,figsize=(12,7));x=np.arange(3)
    metrics=[('feature_rms_px','Raw point RMS (px)'),('depth_to_width','Depth / width (display axes)'),
             ('transverse_ratio','Transverse solid-moment ratio'),('c8_rms','C8 surface RMS (dimensionless)')]
    for ax,(key,title) in zip(axs.ravel(),metrics):
        for start,label in [(0,'Exact point fitting'),(3,'1 px observation disk')]:
            ax.plot(x,[r[key] for r in rows[start:start+3]],'o-',label=label)
        ax.set_xticks(x,['1','0.001','0']);ax.set_xlabel('Camera-prior coefficient (px squared units)')
        ax.set_title(title);ax.grid(alpha=.25)
    axs[0,0].legend();fig.suptitle('Controlled objective study: shape epsilon stays 0.001 in every case')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/'diagnostics.png',dpi=145);plt.close(fig)


def build_html(root,rows):
    body=[]
    for r in rows:
        href=html.escape(r['label'],quote=True)
        body.append(f'<tr><td>{r["camera_coefficient"]:g}</td><td>{r["point_tolerance_pixels"]:g}</td>'
            f'<td>{r["feature_rms_px"]:.4f}</td><td>{r["feature_max_px"]:.3f}</td>'
            f'<td>{" / ".join(f"{v*100:.2f}%" for v in r["mask_IoU"])}</td>'
            f'<td>{r["depth_to_width"]:.3f}</td><td>{r["transverse_ratio"]:.3f}</td>'
            f'<td>{r["c8_rms"]:.3f}</td><td>{r["first_hit_violations"]}</td>'
            f'<td><a href="{href}/index.html">Ảnh &amp; mô hình</a> · '
            f'<a href="{href}/reconstruction.glb">GLB</a> · <a href="{href}/result.json">JSON</a></td></tr>')
    table=''.join(body);host=html.escape(rows[0]['host'])
    audit=root/'intersection_audit.json';warning=''
    if audit.exists():
        values=json.loads(audit.read_text(encoding='utf-8'))
        counts=[r['transverse_nonadjacent_intersections'] for r in values]
        warning=(f'<div class="note"><b>Phát hiện lỗi hình học: {min(counts)}–{max(counts)} cặp tam giác tự cắt trong mỗi mô hình.</b> '
                 'Không được coi các mô hình này là vật rắn hợp lệ hoặc dùng mômen gần 1 làm bằng chứng đối xứng vật lý. '
                 '<a href="intersection_audit.json">Danh sách cặp mặt tự cắt</a>.</div>')
    text=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Đối chứng prior camera và sai số điểm ảnh</title><style>body{{max-width:1350px;margin:30px auto;padding:0 20px;
font:16px/1.6 system-ui;color:#263b48}}img{{max-width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd}}
.note{{padding:18px;background:#fff1c9}}table{{border-collapse:collapse;font-size:14px}}a{{color:#146c8c}}</style>
<h1>Vì sao prior đối xứng nhỏ chưa sửa được độ sâu?</h1>
<p>Sáu cấu hình, cùng ảnh, cùng trạng thái khởi đầu, cùng trục đề xuất từ ảnh và cùng số bước.
Chỉ đổi mức phạt camera và bán kính sai số điểm ảnh. Giữ toàn bộ quan sát và kiểm tra first-hit.
Mọi cấu hình giữ epsilon đối xứng = 0.001 sau quy đổi về đơn vị pixel²; không dùng prior độ gọn.</p>
<div class="note">C8 (quay 45°) là prior được giả định, không phải thuộc tính đã chứng minh từ ảnh.
Dung sai 1 px cũng là giả định về độ chính xác quan sát. Không có kích thước/camera chuẩn hoặc 3D chuẩn.
Khớp ảnh tốt và giảm residual prior không chứng minh dựng đúng hình thật hay nghiệm duy nhất.</div>
{warning}<h2>Hai ảnh đầu vào</h2><p>Đây là ảnh đưa vào thuật toán; các hình xám phía dưới là mô hình ước lượng được chiếu từ góc nhìn mới.</p>
<img src="{rows[0]['label']}/input_01.png" width="300"><img src="{rows[0]['label']}/input_02.png" width="300">
<h2>Mô hình từ cùng góc nhìn mới</h2><img src="geometry_overview.png">
<h2>Đo cùng các sai số thô</h2><img src="diagnostics.png">
<table><tr><th>Prior camera</th><th>Dung sai điểm (px)</th><th>RMS thô (px)</th><th>Max thô (px)</th>
<th>IoU hai ảnh</th><th>Sâu/ngang</th><th>Tỉ số mômen ngang</th><th>C8 RMS</th><th>First-hit sai</th><th>Chi tiết</th></tr>{table}</table>
<p>Sâu/ngang được đo trong hệ hiển thị gắn với camera đầu; tỉ số mômen ngang được đo quanh cùng trục prior.
Hai số này khác nhau và đều không phải sai số 3D so với vật thật. Không xếp hạng các cấu hình bằng tổng cost vì chúng tối ưu các hàm khác nhau.</p>
<h2>Phương pháp và phạm vi</h2><p>Lưới tự do, camera/tiêu cự/tâm ảnh chưa biết;
chiếu phối cảnh, mask hai chiều lấy mẫu và điểm tương ứng SIFT sau lọc hình học.
Với residual điểm r (pixel), cost điểm là trung bình max(||r|| − σ, 0)², nhân hệ số 1.5.
σ=0 khôi phục chính xác cost cũ. Điểm nằm trong đĩa sai số vẫn được giữ trong ràng buộc che khuất.</p>
<p>Tối ưu hiện tại: Adam làm tiền điều kiện, tìm bước giảm Armijo, cập nhật quay trên SO(3),
làm mới liên kết bề mặt theo từng khối; augmented Lagrangian cho first-hit khi cần.
Đây chưa phải hiện thực DCA. Kiểm tra mask lấy mẫu và first-hit thưa chưa chứng minh song ánh toàn bộ vùng nhìn thấy.</p>
<p>Máy thực chạy: <b>{host}</b> (3090 dự phòng); dữ liệu tại ổ G. Xeon ưu tiên đã vào được màn hình,
nhưng chưa chạy job qua SSH được. <a href="summary.csv">CSV</a> · <a href="summary.json">Toàn bộ số đo</a>.</p></html>'''
    (root/'index.html').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();rows=load_results(root)
    with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    geometry_figure(root,rows);diagnostics_figure(root,rows);build_html(root,rows)
    print('Saved complete six-case report:',root/'index.html')


if __name__=='__main__':main()
