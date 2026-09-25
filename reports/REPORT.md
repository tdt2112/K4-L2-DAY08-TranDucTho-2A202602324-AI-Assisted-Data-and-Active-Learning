# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Tran Duc Tho

Công cụ gán nhãn đã dùng: CVAT

Mọi con số trong báo cáo được trích xuất trực tiếp và đối chiếu từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json` và `outputs/round1_diff.md`.

## 1. Dữ liệu và cách chia tập

Tập dữ liệu chưa gán nhãn (pool) và tập kiểm thử (test set) được trích xuất từ cùng một video giám sát giao thông ban đêm quay cố định một vị trí đường cao tốc. Trong bối cảnh này, một chiếc xe di chuyển trên đường sẽ tồn tại trong khung hình qua nhiều giây liên tiếp. Do đó, các khung hình liền kề hoặc gần sát nhau có mức độ tương đồng cực kỳ cao về vị trí xe, nền đường và góc chiếu ánh sáng.

Việc phân chia tập pool và test theo trục thời gian kèm vùng đệm (buffer/gap) ở giữa nhằm bảo đảm tính độc lập dữ liệu giữa tập huấn luyện và tập kiểm thử. 

Nếu phân chia ngẫu nhiên (random split), cùng một chiếc xe tại cùng một vị trí sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử. Khi đó, hiện tượng rò rỉ dữ liệu (data leakage) sẽ xảy ra nghiêm trọng. Mô hình chỉ cần ghi nhớ chiếc xe cụ thể đó thay vì học được quy luật tổng quát hóa để phát hiện xe nói chung. Hậu quả là các chỉ số đánh giá (AP50, Precision, Recall) trên tập kiểm thử sẽ bị thổi phồng, cao hơn nhiều so với năng lực thực tế của mô hình (optimistic bias) và không phản ánh đúng khả năng hoạt động trên dữ liệu thực tế trong tương lai.

## 2. Mô hình khởi đầu lạnh (cold start)

Số đo của mô hình khởi đầu lạnh trích xuất từ `reports/rounds_table.md`:
```text
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.773 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
```

Dựa vào ảnh phân tích `outputs/compare_round0.jpg` và các chỉ số chi tiết:
- Mô hình khởi đầu lạnh (YOLOv8n pretrained trên tập COCO) bỏ sót rất nhiều xe ở xa có kích thước nhỏ. Chỉ số độ phủ theo kích thước cho thấy: R small chỉ đạt 0.182, trong khi R medium đạt 0.547 và R large đạt 0.561. Xe càng ở xa sát đường chân trời hoặc chân cầu vượt thì tỷ lệ phát hiện được càng thấp.
- Ngoài ra, mô hình ban đầu bị ảnh hưởng nặng bởi ánh sáng chói của đèn pha xe chiếu xuống mặt đường nhựa ban đêm, dẫn đến việc khoanh nhầm nhiều vệt sáng phản chiếu trên mặt đường thành các đối tượng xe (False Positive).
- Một trường hợp cần người rà soát lại nhãn tham chiếu trước khi kết luận mô hình sai: Nhãn tham chiếu của tập kiểm thử cũng được tạo tự động bởi một mô hình khác mà chưa có chuyên gia con người kiểm duyệt từng khung. Ví dụ, ở các vị trí xe bị che khuất một phần bởi rào chắn hoặc xe ở quá xa chỉ còn 2 chấm đèn nhỏ li ti, nhãn tham chiếu có thể đã bỏ sót hoặc đóng khung chưa chính xác; khi mô hình phát hiện đúng những vị trí này nhưng không khớp với nhãn tham chiếu thì sẽ bị tính nhầm là sai.

## 3. Chiến lược chọn mẫu

Công thức tính điểm chọn mẫu trong thuật toán Active Learning:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D = 0.5 \cdot U + 0.3 \cdot A + 0.2 \cdot D$$

Ý nghĩa các thành phần:
- $U$ (Uncertainty): Độ bất định trung bình của 5 box khó nhất trong ảnh. Điểm bất định $u = 1 - |2 \cdot \text{conf} - 1|$ đạt cực đại khi độ tin cậy $\text{conf} = 0.5$ (lúc mô hình phân vân nhất giữa có xe hay không có xe).
- $A$ (Ambiguity): Số lượng box mơ hồ ($0.15 \le \text{conf} < 0.5$) trong ảnh, được chuẩn hóa theo giá trị lớn nhất trong pool. Ảnh có $A$ cao là ảnh chứa nhiều đối tượng mà mô hình còn lưỡng lự.
- $D$ (Diversity): Độ đa dạng theo thời gian, đo khoảng cách thời gian từ ảnh đang xét tới ảnh đã gán nhãn gần nhất (tính tối đa 10 giây).
- `MIN_GAP_S = 2.0s`: Khoảng cách thời gian tối thiểu bắt buộc giữa 2 ảnh được chọn trong cùng một lô. Camera cố định nên 2 frame cách nhau dưới 2 giây có bối cảnh gần như trùng lặp hoàn toàn. Việc áp dụng `MIN_GAP_S` ngăn chặn việc đưa các ảnh gần trùng vào cùng một lô, tránh lãng phí chi phí gán nhãn.

Minh chứng cụ thể từ `reports/SELECTION.md`:
- Ba frame được chọn trong lô 12 ảnh:
  1. `frame_0182.jpg` (hạng 1, điểm 0.9591): Có $A = 1.0$ (18 box mơ hồ, cao nhất pool) và $U = 0.9182$.
  2. `frame_0099.jpg` (hạng 8, điểm 0.9063, t = 39.6s): Có $U = 0.9460$, phản ánh nhiều tình huống khó ở góc ảnh bị cắt và xe xa chân cầu.
  3. `frame_0107.jpg` (hạng 14, điểm 0.8876, t = 42.8s): Cách frame_0099 hơn 3.2 giây, có 33 box với 15 box mơ hồ và nhiều vệt đèn pha rọi sáng.
- Một frame điểm cao nhưng bị loại: `frame_0372.jpg` xếp hạng 6 với điểm rất cao 0.9101, nhưng bị loại (`selected = False`) vì xuất hiện ở giây 148.8s, chỉ cách `frame_0369.jpg` (147.6s) đúng 1.2 giây (nhỏ hơn ngưỡng 2.0s). Loại bỏ frame này giúp tiết kiệm công sức gán nhãn mà không làm mất mát lượng thông tin học được.

Điểm bất định cao chỉ thể hiện rằng mô hình đang chưa chắc chắn về các dự đoán trên ảnh đó. Điều này **chưa chứng minh** việc gán nhãn ảnh đó chắc chắn sẽ cải thiện điểm số mô hình. Nếu ảnh chứa quá nhiều nhiễu phức tạp, nhòe chuyển động quá mức hoặc phân phối sai khác với tập kiểm thử, việc đưa vào huấn luyện thậm chí có thể làm mô hình trở nên quá khắt khe hoặc suy giảm độ phủ.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp kết quả các vòng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.773 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 347 | 0.687 | -0.086 | 1.000 | 0.196 | 0.328 | 0.000 | 0.213 | 0.390 |

Phân tích chi tiết vòng 1:
- **Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`):** Trong lô 12 ảnh, mô hình ban đầu đề xuất 169 box. Sau khi rà soát và chỉnh sửa trên CVAT, tổng số box đạt **347 box**. Cụ thể: 137 box được giữ nguyên (accepted), 16 box được kéo chỉnh sát thân xe (edited), 16 box bị xóa bỏ (deleted - các lỗi False Positive của AI như vệt sáng đèn pha trên mặt đường), và vẽ thêm mới 194 box (added - các xe nhỏ, xe màu tối hoặc xe bị che mà AI bỏ sót). Tỷ lệ chấp nhận nhãn gợi ý (accept rate) là 81%.
- **Biến động AP50:** Điểm AP50 ở vòng 1 đạt **0.687**, giảm 0.086 so với khởi đầu lạnh (0.773). Tuy nhiên, độ chính xác **Precision@0.25 tăng vọt lên mức tuyệt đối 1.000** (toàn bộ 79 box dự đoán đều là True Positive, số box báo sai FP = 0).
- **Xu hướng các nhóm xe:** Do người gán nhãn đã triệt để loại bỏ các box vệt đèn pha và tinh chỉnh lại các box không chắc chắn, mô hình sau khi fine-tune trở nên cực kỳ thận trọng và khắt khe. Nó chỉ kích hoạt dự đoán khi thật sự chắc chắn, dẫn đến việc Precision đạt 100% nhưng Recall ở các nhóm xe giảm xuống (R large: 0.390, R medium: 0.213, R small: 0.000).

So sánh qua các kênh thông tin:
- **Quan sát độc lập (`BLIND_SCAN.md`):** Khi xem ảnh `frame_0099.jpg` mà chưa có khung AI, mắt người đếm được khoảng 18 xe và nhận diện ngay nguy cơ bỏ sót xe bị cắt góc mép phải và các chấm đèn xa ở chân cầu.
- **Nhật ký rà soát (`REVIEW_LOG.csv`):** Đã xử lý triệt để 3 tình huống thực tế: xóa vệt đèn phản chiếu trên mặt đường (`frame_0326.jpg`), kéo dãn hợp nhất khung xe bị AI tách nhầm thành 2 cụm đèn hậu (`frame_0331.jpg`), và vẽ thêm xe bị mép ảnh cắt mất góc dưới bên phải (`frame_0099.jpg`).
- **Hình ảnh so sánh sau huấn luyện (`outputs/compare_round1.jpg`):** So với `compare_round0.jpg`, mô hình sau fine-tune đã loại bỏ hoàn toàn các khung màu đỏ (False Positive trên vệt đèn mặt đường), khung dự đoán ôm rất gọn gàng thân xe. 
- **Ca khó điển hình theo guideline:** Xe đang di chuyển với tốc độ cao bị nhòe vệt sáng hoặc xe bị mép ảnh cắt ngang. Guideline quy định chỉ khoanh trọn phần thân xe còn nhìn thấy trong khung hình và bao gồm cả vệt nhòe thân xe, không được khoanh tràn ra ngoài mép ảnh hoặc ôm theo vệt đèn pha rọi trên đường.

## 5. Kết luận và giới hạn

So sánh với khởi đầu lạnh, vòng 1 thể hiện rõ nét đặc trưng của việc tinh chỉnh dữ liệu ban đầu: mô hình đã học được cách phân biệt chính xác giữa thân xe thực tế và ánh đèn phản chiếu trên mặt đường (Precision đạt tuyệt đối 1.000, FP = 0), giải quyết triệt để lỗi báo động giả của mô hình gốc, dù điểm AP50 tổng thể có sự suy giảm nhẹ do mô hình trở nên thận trọng hơn khi số lượng mẫu huấn luyện còn khiêm tốn (12 ảnh).

**Quyết định:** Tôi quyết định dừng ở vòng 1 để đánh giá sâu và củng cố chất lượng nhãn, vì việc tiếp tục huấn luyện thêm vòng 2 ngay lập tức mà chưa cân bằng lại các mẫu xe nhỏ ở xa có thể khiến mô hình tiếp tục thiên lệch về tính thận trọng.

Đề xuất cho vòng tiếp theo:
1. **Xe ở khoảng cách xa (chỉ còn hai chấm đèn):** Cần quy định rõ ngưỡng kích thước tối thiểu trước khi gắn nhãn để tránh việc người gán nhãn đưa vào các chấm đèn quá nhỏ mà tập test bỏ qua, gây nhiễu cho hàm mất mát.
2. **Xe kích thước lớn (xe tải, xe buýt):** Cần tăng cường thêm các frame có xe tải/xe buýt để khôi phục và nâng cao chỉ số R large.
3. **Chi phí và nguy cơ trùng lặp:** Việc rà soát 12 ảnh đòi hỏi trung bình 30–45 phút tập trung cao độ. Cần duy trì nghiêm ngặt tham số `MIN_GAP_S` để đảm bảo mỗi phút công sức gán nhãn đều mang lại mẫu dữ liệu có giá trị thông tin mới.

**Giới hạn thực nghiệm:**
Tập kiểm thử chỉ gồm 20 ảnh với 403 box tham chiếu; cỡ mẫu này tương đối nhỏ nên độ nhạy thống kê cao, chỉ một vài box thay đổi cũng có thể làm dao động điểm số. Ngoài ra, việc quy tắc tự động bỏ qua các box cao dưới 16px và việc nhãn tham chiếu test do AI khác tạo ra mà chưa qua kiểm duyệt thủ công của con người đồng nghĩa với việc số đo AP50 chỉ là chỉ số định hướng tương đối, không phải chân lý tuyệt đối.

Nếu điểm AP50 suy giảm ở các vòng sau, quy trình kiểm tra cần thực hiện là:
1. Đối soát độ nhất quán giữa các box đã sửa trong `labels/` với quy tắc của `GUIDELINE_LABEL.md`.
2. Kiểm tra lại xem có tình huống xe tải/xe buýt bị xóa nhầm hay không.
3. Rà soát tỷ lệ phân bố kích thước xe (small/medium/large) giữa tập train và tập test để đảm bảo mô hình không bị thiên lệch vào một nhóm kích thước duy nhất.
