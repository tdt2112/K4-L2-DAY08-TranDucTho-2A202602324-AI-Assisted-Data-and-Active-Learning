# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét ảnh gần trùng hoặc trường hợp model không dự đoán được box:
Nếu chỉ được rà soát 5 ảnh do giới hạn ngân sách, tôi ưu tiên chọn 5 frame có độ bất định và mật độ thông tin cao nhất:
1. `frame_0182.jpg` (hạng 1, điểm 0.9591, t = 72.8s): Đứng đầu toàn tập pool, có độ bất định U = 0.9182 và số box mơ hồ A = 1.0 (18 box không chắc chắn).
2. `frame_0369.jpg` (hạng 2, điểm 0.9324, t = 147.6s): Giao thông rất đông với 43 box, 16 box mơ hồ, độ bất định U = 0.9315 rất cao.
3. `frame_0380.jpg` (hạng 3, điểm 0.9170, t = 152.0s): Cách frame_0369 4.4 giây, chứa 40 box và 15 box mơ hồ.
4. `frame_0326.jpg` (hạng 4, điểm 0.9155, t = 130.4s): Nhiều vùng sáng phức tạp rọi mặt đường, có 39 box với U = 0.9310.
5. `frame_0331.jpg` (hạng 5, điểm 0.9154, t = 132.4s): Cách frame_0326 đúng 2.0s, có tới 47 box và A = 1.0.
Xét về ảnh gần trùng: Mặc dù `frame_0187.jpg` xếp hạng 10 với điểm rất cao (0.8995), tôi không chọn nó cùng `frame_0182.jpg` nếu chỉ có 5 suất vì chúng cách nhau đúng 2.0s, bối cảnh xe gần như lặp lại; nên ưu tiên rải sang các mốc thời gian khác (như giây thứ 130 - 150) để tăng tính đa dạng cho mô hình.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- `frame_0182.jpg` (hạng 1, điểm 0.9591): Có A = 1.0 (18 box mơ hồ). Trên contact sheet, xe ở lề đường bên trái và các xe màu tối chạy xa có nhiều khung viền thể hiện độ tin cậy thấp (0.15 ≤ conf < 0.5).
- `frame_0099.jpg` (hạng 8, điểm 0.9063, t = 39.6s): Chỉ số U = 0.9460. Ảnh có 29 box với 14 box phân vân. Model gặp nhiều khó khăn ở các xe bị mép ảnh cắt mất góc dưới bên phải và cụm đèn xe ở chân cầu.
- `frame_0107.jpg` (hạng 14, điểm 0.8876, t = 42.8s): Cách frame_0099 hơn 3.2s. Ảnh có 33 box với 15 box mơ hồ, đặc biệt là nhiều vệt đèn pha rọi sáng mặt đường khiến mô hình dễ nhầm lẫn.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- `frame_0372.jpg` xếp hạng 6 với điểm số 0.9101 (cao hơn nhiều frame được chọn như frame_0312, frame_0099, frame_0187,...). Tuy nhiên, frame này bị loại (`selected = False`) do thời điểm t = 148.8s chỉ cách `frame_0369.jpg` (t = 147.6s) vỏn vẹn 1.2 giây (nhỏ hơn ngưỡng thời gian tối thiểu `MIN_GAP_S = 2.0s`). Camera cố định nên sau 1.2s bối cảnh xe gần như không thay đổi, gán nhãn cả hai sẽ gây lãng phí chi phí nhân công mà không đem lại giá trị thông tin mới.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
Điểm số cao trong Active Learning chỉ phản ánh rằng mô hình đang phân vân hoặc chưa chắc chắn về các dự đoán trên ảnh đó. Điều này chưa thể chứng minh hay đảm bảo rằng việc gán nhãn các ảnh điểm cao chắc chắn sẽ cải thiện độ chính xác (AP50) của mô hình. Nếu các ảnh được chọn chứa quá nhiều ca nhiễu cực đoan hoặc nhãn sửa có sự sai lệch với tập nhãn kiểm thử (test set do máy tạo), mô hình thậm chí có thể không tăng điểm hoặc giảm nhẹ.
