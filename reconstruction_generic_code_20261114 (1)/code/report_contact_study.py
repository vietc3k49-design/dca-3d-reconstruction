"""Complete, labeled report of contact-safe camera-prior sensitivity, with failure cases."""
import argparse
import csv
import html
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from reconstruction.raster import render_mesh
from reconstruction.rotations import look_at


def load_contacts(root):
    records=[]
    for name,weight in [('results',1.),('camera_weak/results',.001),('camera_zero/results',0.)]:
        folder=Path(root)/name
        row=json.loads((folder/'contact_audit.json').read_text(encoding='utf-8'))
        payload=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if row['camera_coefficient']!=weight:raise ValueError('Mislabeled camera-prior case')
        records.append((name,row,payload))
    return records


def figures(root,records,filename='contact_comparison.png'):
    radius=max(np.linalg.norm(np.array(p['display_vertices']),axis=1).max() for _,_,p in records)
    distance=3.2*radius;fig,axs=plt.subplots(2,3,figsize=(13,9))
    for column,(folder,row,p) in enumerate(records):
        X=np.array(p['display_vertices']);F=np.array(p['faces'])
        for line,direction in enumerate(([1.,-1.,np.sqrt(2.)],[1.,0.,.25])):
            C=np.array(direction);C*=distance/np.linalg.norm(C)
            image=render_mesh(X,F,C,look_at(C,np.zeros(3)),640,[320,320],(640,640))
            image.save(root/folder/f'comparison_view_{line}.png')
            ax=axs[line,column];ax.imshow(image);ax.axis('off')
            if line==0:
                ax.set_title(f'Camera coefficient {row["camera_coefficient"]:g}\n'
                    f'RMS {row["feature_rms_px"]:.3f} px; min IoU {min(row["mask_IoU"])*100:.2f}%\n'
                    f'Depth/width {row["depth_to_width"]:.2f}; crossings {row["contact_audit"]["intersecting_pairs"]}',fontsize=10)
            else:ax.set_title('Side view: inspect unobserved protrusions',fontsize=10)
    fig.suptitle('Same free-mesh algorithm; shape epsilon = 0.001; all 112 observations retained\n'
                 'Gray geometry only. Common display cameras; height normalized to 1 relative unit. No 3D ground truth.',fontsize=12)
    fig.subplots_adjust(top=.84,bottom=.02,left=.02,right=.98,hspace=.18,wspace=.12)
    fig.savefig(root/filename,dpi=145);plt.close(fig)


def hidden_section(root):
    if not (root/'hidden_c1').exists():return ''
    records=[];table=[];summaries=[]
    for name,weight in [('hidden_c1',1.),('hidden_weak',.001),('hidden_zero',0.)]:
        folder=root/name/'results'
        row=json.loads((folder/'hidden_audit.json').read_text(encoding='utf-8'))
        p=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        records.append((f'{name}/results',row,p));summaries.append(row)
        table.append(f'<tr><td>{weight:g}</td><td>{row["before_objective_px_squared"]:.6f} → '
            f'{row["after_objective_px_squared"]:.6f}</td><td>{row["feature_rms_px"]:.3f}</td>'
            f'<td>{min(row["mask_IoU"])*100:.2f}%</td><td>{row["depth_to_width"]:.3f}</td>'
            f'<td>{row["c8_rms"]:.3f}</td><td>{row["contact_audit"]["intersecting_pairs"]}</td>'
            f'<td><a href="{name}/results/index.html">Ảnh</a> · <a href="{name}/results/reconstruction.glb">GLB</a> · '
            f'<a href="{name}/results/hidden_audit.json">Số đo</a></td></tr>')
    figures(root,records,'hidden_comparison.png')
    (root/'hidden_comparison.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    return ('<h2>Bước tối ưu riêng phần khuất: giữ nguyên cost và epsilon</h2>'
        '<p>Phân loại theo mô hình hiện tại; chỉ cập nhật các đỉnh khuất ở mọi ảnh và không mang điểm ghép. '
        'Các phép chiếu của điểm ghép được giữ nguyên chính xác; camera và phần đã quan sát được giữ theo cùng gauge. '
        'Vẫn đánh giá toàn bộ mask, first-hit và IPC. Đây là đổi thứ tự cập nhật tọa độ, không thêm prior theo loại vật.</p>'
        '<img src="hidden_comparison.png"><table><tr><th>Prior camera</th><th>Cùng cost trước → sau (px²)</th>'
        '<th>RMS điểm (px)</th><th>IoU thấp hơn</th><th>Sâu/ngang</th><th>C8 RMS</th><th>Tự cắt</th><th>Kết quả</th></tr>'+
        ''.join(table)+'</table><p>Cost chỉ được so sánh trước/sau trong từng hàng. '
        'Vùng khuất vẫn được hoàn thiện theo prior và khởi tạo; đây không phải phép đo 3D thật.</p>')


def optimizer_table(root):
    source=root.parent/'optimizer_study_20261114';rows=[]
    if not source.exists():return
    for folder in sorted((source/'study/results_adam').iterdir()):
        if not folder.is_dir():continue
        a=json.loads((folder/'ablation.json').read_text(encoding='utf-8'))
        b=json.loads((source/'study/results_lbfgs'/folder.name/'ablation.json').read_text(encoding='utf-8'))
        rows.append(dict(case=folder.name,before=a['starting_objective_px_squared'],
            adam_after=a['final_objective_px_squared'],lbfgs_after=b['final_objective_px_squared'],
            adam_steps=a['accepted_steps'],lbfgs_steps=b['accepted_steps'],
            adam_depth_width=a['depth_to_width'],lbfgs_depth_width=b['depth_to_width']))
    with (root/'optimizer_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_report(root,records):
    rows=[];flat=[]
    for folder,r,p in records:
        flat.append({k:r[k] for k in ('camera_coefficient','feature_rms_px','feature_max_px','depth_to_width',
                     'transverse_ratio','c8_rms','image_cost_px_squared','shape_cost_px_squared','camera_cost_px_squared')})
        flat[-1].update(mask_IoU_1=r['mask_IoU'][0],mask_IoU_2=r['mask_IoU'][1],
            self_intersections=r['contact_audit']['intersecting_pairs'],first_hit_violations=r['first_hit_violations'])
        rows.append(f'<tr><td>{r["camera_coefficient"]:g}</td><td>{r["feature_rms_px"]:.3f}</td>'
            f'<td>{r["feature_max_px"]:.3f}</td><td>{r["mask_IoU"][0]*100:.2f}% / {r["mask_IoU"][1]*100:.2f}%</td>'
            f'<td>{r["contact_audit"]["intersecting_pairs"]}</td><td>{r["first_hit_violations"]}</td>'
            f'<td>{r["depth_to_width"]:.3f}</td><td>{r["transverse_ratio"]:.3f}</td><td>{r["c8_rms"]:.3f}</td>'
            f'<td><a href="{folder}/index.html">Đối chiếu ảnh</a> · <a href="{folder}/reconstruction.glb">GLB</a> · '
            f'<a href="{folder}/result.json">JSON</a> · <a href="{folder}/verification.json">Kiểm tra lại</a></td></tr>')
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (root/'comparison.json').write_text(json.dumps(flat,indent=2),encoding='utf-8')
    table=''.join(rows);hidden=hidden_section(root)
    content=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kiểm tra lại dựng 3D: cost, tự giao cắt và bộ tối ưu</title><style>
body{{max-width:1350px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#253b48}}
img{{max-width:100%}}.note{{padding:18px;background:#fff0cc}}td,th{{padding:9px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;font-size:14px}}a{{color:#126b89}}code{{background:#edf1f3;padding:2px 5px}}</style>
<h1>Đã sửa lỗi tính toán; chưa phục hồi đúng hình 3D</h1>
<p>Ngày 14-11-2026. Chạy trên DESKTOP-L6QJ2RK (3090 dự phòng), ổ G; tiến trình dựng hình khoảng 130 MB RAM.
Xeon ưu tiên vào được màn hình nhưng chưa xác thực SSH để chạy job. Không chạy fit nặng trên máy làm việc.</p>
<div class="note"><b>Kết quả chưa đủ làm bằng chứng dựng đúng vật thật.</b> Lưới mới không tự cắt theo kiểm tra libigl;
ảnh chiếu lại khớp tốt, nhưng các góc nhìn mới vẫn có phần nhô và mặt gấp. Không có camera/kích thước/3D chuẩn,
không có kết quả sai số cm/mm, không có chứng minh nghiệm duy nhất. Tỉ số mômen gần 1 không đồng nghĩa bề mặt đạt C8.</div>
<h2>Hai ảnh đầu vào</h2><img src="results/input_01.png" width="300"><img src="results/input_02.png" width="300">
{hidden}<h2>Đối chứng trước bước tối ưu riêng phần khuất</h2><p>Cả ba cấu hình dùng cùng lưới tự do, cùng khởi tạo trung tính và cùng quan sát.
Chỉ đổi trọng số prior camera. Không dùng kích thước lon, phương trình trụ tròn xoay hoặc mẫu hình học theo loại vật.</p>
<img src="contact_comparison.png">
<table><tr><th>Prior camera</th><th>RMS thô (px)</th><th>Max (px)</th><th>IoU hai ảnh</th><th>Tự cắt</th>
<th>First-hit sai</th><th>Sâu/ngang</th><th>Mômen ngang</th><th>C8 RMS</th><th>Kết quả</th></tr>{table}</table>
<p><a href="comparison.csv">CSV</a> · <a href="comparison.json">Số đo đầy đủ</a>.
Sâu/ngang trong hệ hiển thị gắn với camera đầu; chiều cao được chuẩn hóa để xem. Các độ đo này không phải sai số so với vật thật.
Không xếp hạng ba cấu hình bằng tổng cost vì các hàm mục tiêu khác nhau.</p>
<h2>Nguyên nhân đã kiểm chứng</h2>
<p><b>1. Tự giao cắt bị bỏ lọt.</b> Sáu mô hình cũ có 30–554 cặp mặt tự cắt theo kiểm tra độc lập;
bản chặn giao cắt đơn giản vẫn bỏ lọt 3 cặp. Không thể dùng mômen thể tích của các lưới này để chứng nhận đối xứng vật rắn.
<a href="results/baseline_intersection_audit.json">Các cặp mặt và kiểm tra đối chứng</a>.</p>
<p><b>2. Nội suy trường khoảng cách làm kẹt bước.</b> Ở một nghiệm cũ, đạo hàm dự báo −0.0608 nhưng bước 10⁻⁷
cho đạo hàm một phía +0.243. Với nội suy cubic, giá trị dự báo −0.4245 và đo được −0.4235.
<a href="../axis_ablation_20261114/results/field_stall_diagnosis.json">Số đo</a> ·
<a href="../axis_ablation_20261114/results/field_stall_diagnosis.png">Đồ thị</a>.</p>
<p><b>3. Chỉ thay bộ tối ưu không chữa được hình học.</b> Đã chạy Adam và L-BFGS-B từ đúng cùng sáu nghiệm,
cùng cost và cùng ngân sách tối đa. Có cost giảm mà hình vẫn kéo dài hoặc tự cắt.
<a href="optimizer_comparison.csv">Đối chứng trong từng hàm mục tiêu</a>.</p>
<h2>Cách sửa dùng chung</h2>
<p>Cost theo đơn vị pixel²: D_mask + 1.5 D_point + 0.001 R_shape + λ_camera R_camera + μ B_contact.
Mask dùng cùng trường khoảng cách nhưng nội suy cubic với đạo hàm liên tục trong ảnh.
Cost điểm là trung bình max(||r|| − 1 pixel, 0)²; giữ mọi UV và track, kể cả điểm nằm trong đĩa sai số.
Dung sai 1 pixel là giả định, không phải độ tin cậy được đo từ ảnh.</p>
<p>R_shape gồm đối xứng tâm, giả thuyết gương, C8 quanh trục đề xuất từ ảnh, mômen ngang, độ lõm và độ trơn.
C8 là prior mặc định được giả định, không được suy ra chắc chắn từ ảnh. Trọng số nhỏ không bắt buộc hình đạt đối xứng.</p>
<p><a href="https://github.com/ipc-sim/ipc-toolkit">IPC Toolkit</a> cung cấp barrier và CCD để bước đi tránh xuyên mặt;
Armijo dùng chính hướng đã thu ngắn bởi CCD. μ chạy theo 1 → 0.1 → 0.01, mỗi mức tối đa 600 bước;
đây là điều kiện tránh va chạm, tách khỏi epsilon hình dạng. Chuẩn hóa barrier theo bán kính RMS có đạo hàm đầy đủ.
<a href="https://libigl.github.io/libigl-python-bindings/api/igl_predicates/">libigl</a> kiểm tra giao cắt độc lập, gồm trường hợp đồng phẳng.</p>
<h2>Phạm vi cho paper</h2>
<p>Đây là tối ưu hình học với đặc trưng SIFT và camera chưa biết. Màu/chữ/giọt nước chưa có trong cost RGB.
Mask lấy mẫu và first-hit tại các điểm ghép chưa chứng minh song ánh toàn vùng nhìn thấy.
Ảnh web chưa được xác nhận chụp cùng một vật cứng; phần khuất vẫn phụ thuộc prior, khởi tạo và nghiệm cục bộ.</p>
<p>Hiện thực là Adam có tiền điều kiện, Armijo, cập nhật quay SO(3) và IPC; <b>chưa phải DCA trên đa tạp</b>.
Các thư viện va chạm là công cụ được dùng lại, không phải đóng góp mới. Cần đánh giá trên dữ liệu có 3D chuẩn,
nhiều vật và nhiều khởi tạo trước khi tuyên bố hiệu quả tái dựng.</p>
<p><a href="../axis_ablation_20261114/results/index.html">Đủ sáu cấu hình cost cũ</a> ·
<a href="../guarded_fit_20261114/results/cubic/index.html">Đối chứng chỉ chặn giao cắt</a> ·
<a href="RUN_AND_LIMITS.txt">Cách chạy và giới hạn</a>.</p></html>'''
    (root/'index.html').write_text(content,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();records=load_contacts(root)
    figures(root,records);optimizer_table(root);write_report(root,records)
    print('Saved all-case contact report:',root/'index.html')


if __name__=='__main__':main()
