"""HTML presentation extracted without changing numerical reconstruction."""
def build_report_html(before,after,result):
    rows=''.join(f'<tr><td>{i+1}</td><td>{b["mask_IoU"]*100:.2f}%</td><td>{z["mask_IoU"]*100:.2f}%</td>'
                 f'<td>{z["extra_pixels"]}</td><td>{z["missing_pixels"]}</td></tr>' for i,(b,z) in enumerate(zip(before,after)))
    html='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dựng 3D từ ảnh — pipeline tổng quát</title><style>body{max-width:1200px;margin:36px auto;padding:0 20px;font:16px/1.6 system-ui;color:#263b48}img{max-width:100%}td,th{padding:10px;border-bottom:1px solid #d6dfe3}a{color:#146c8c}.note{background:#fff0c7;padding:18px}</style>
<h1>Kết quả chạy cùng thuật toán trên ảnh đầu vào</h1><p>Khởi tạo từ icosphere trung tính; tối ưu từng đỉnh và camera.
Không dùng phương trình lon, trụ tròn xoay, mẫu ngôi sao, kích thước vật, camera thật hoặc landmark đặt bằng tay.</p>
<p>Các kernel cost hai chiều, first-hit và vòng Adam dùng chung với bản thử trước.
Mask lấy từ nền gần đồng nhất; điểm tương ứng lấy bằng SIFT + lọc hình học.</p>
<div class="note">Đây là lưới tự do với giả thiết topo hình cầu. Phần lưng chưa thấy phụ thuộc prior.
Mô hình được chuẩn hóa chiều cao = 1 đơn vị tương đối để xem, không phải 1 mét.
Không có dữ liệu 3D chuẩn nên không báo sai số 3D theo cm/mm.</div>
<h2>Ảnh đầu vào và chiếu lại</h2><img src="input_reprojection_comparison.png">
<table><tr><th>Ảnh</th><th>IoU khởi tạo</th><th>IoU sau tối ưu</th><th>Pixel thừa</th><th>Pixel thiếu</th></tr>'''+rows+'''</table>
<h2>Mô hình 3D</h2><p><a href="reconstruction.glb">GLB</a> · <a href="reconstruction.obj">OBJ</a> · <a href="reconstruction.ply">PLY</a> · <a href="result.json">Số đo và camera</a></p>
<img src="model_views.png"><h2>Điểm tương ứng từ ảnh</h2><img src="feature_matches.png">
<p>IoU chỉ đánh giá vùng vật thể trong ảnh; không chứng minh mặt sau hoặc camera đã đúng.
Kết quả hiện tại còn sai/lệch độ sâu khi nhìn từ hướng khác; chưa phải mô hình 3D được xác thực.
Hiện chưa tối ưu màu RGB, nhãn in hay giọt nước. Cost lấy mẫu và kiểm tra first-hit tại các điểm ghép chưa chứng minh song ánh trên toàn bộ vùng nhìn thấy.</p>
<p>Thuật toán giả thiết các ảnh có chung một hình học vật thể cứng. Với hai ảnh web này, giả thiết đó và camera thật chưa được xác minh.
Tối ưu hiện dùng Adam với cập nhật SO(3); chưa phải hiện thực DCA hay một kết quả bảo đảm hội tụ/duy nhất.</p></html>'''
    selection=result.get('completion_selection')
    if selection:
        note=(f'<h2>Chọn prior chống kéo dài</h2><p>Ưu tiên log(A³/(36πV²)), không dùng mẫu hình học theo loại vật. '
              f'Trong các độ mạnh đã thử, chọn {selection["selected_strength_pixels"]:g} pixel tương đương '
              f'với sai lệch ảnh không quá {selection["allowed_image_cost_px_squared"]:.3f} px² '
              f'(mức tốt nhất cộng {selection["noise_pixels"]**2:g} px²). '
              'Đây là ngân sách cho cost 2D lấy mẫu có trọng số, không phải giới hạn sai số trên từng pixel. '
              'Prior có thể thiên lệch với vật thật sự dài/mỏng; chưa chứng minh độ sâu hoặc nghiệm duy nhất. '
              'Xem toàn bộ completion_selection trong result.json, kể cả các nghiệm không được chọn.</p>')
        html=html.replace('</html>',note+'</html>')
    axis_selection=result.get('axis_symmetry_selection')
    if axis_selection:
        note=(f'<h2>Đối xứng quanh trục được đề xuất từ ảnh</h2><p>Giả thuyết được chọn: {axis_selection["selected"]}. '
              'Quay 45° (C8) là giả thiết mặc định đã đặt, không phải bậc đối xứng được suy ra từ ảnh. '
              'Trục giữ cố định trong lượt tối ưu; phép phạt so với bề mặt và mômen thể tích, không dùng hoán vị đỉnh. '
              'Ưu tiên độ gọn toàn cục đã tắt. Giả thuyết phải đạt ngân sách khớp ảnh và dung sai first-hit; '
              'nếu không đạt thì giữ mô hình không áp dụng giả thuyết trục. '
              'Xem toàn bộ axis_symmetry_selection trong result.json, gồm cả giả thuyết bị loại.</p>')
        html=html.replace('</html>',note+'</html>')
        if axis_selection.get('symmetry_is_soft_only'):
            epsilon=axis_selection['symmetry_epsilon']
            note=(f'<p><b>Prior mềm: epsilon = {epsilon:g} sau chuẩn hóa theo pixel.</b> '
                  'Không có điều kiện cứng buộc sâu bằng ngang hoặc mô hình phải đạt C8. '
                  'So sánh hai hình bằng cùng một objective; chỉ nhận cập nhật được objective yếu này hỗ trợ. '
                  'Nếu sai lệch đối xứng còn lớn, không được gọi mô hình là đã đạt đối xứng C8.</p>')
            html=html.replace('</html>',note+'</html>')
    return html
