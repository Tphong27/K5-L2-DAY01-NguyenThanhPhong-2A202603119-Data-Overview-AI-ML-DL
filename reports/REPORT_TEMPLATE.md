# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.

Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

### 1.1. Chép record hạng 1

- `class_id` = 468
- `class_name` = cab
- `rank` = 1
- `score` = 0.510915
- `taxonomy_name` = ImageNet-1K

### 1.2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?

Về cả bức ảnh. Record hạng 1 chỉ có class_id, class_name, rank và score. Nó không có bbox_xyxy hay polygon_xy, nên không gắn với vị trí nào trong ảnh. Ảnh traffic có hàng chục xe buýt, ô tô và người đi bộ, nhưng record chỉ đưa ra một nhãn cab cho toàn khung hình.

### 1.3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Đi kèm taxonomy và checkpoint, không phải model tự nghĩ ra. Mỗi record ghi taxonomy_name: ImageNet-1K và model_file: yolo11n-cls.pt. Model chỉ chọn trong danh sách lớp cố định mà nó được huấn luyện cùng (ImageNet-1K), rồi trả về chỉ số class_id. Tên lớp được tra từ danh sách đó (names[class_id] ở cell [3]).

### 1.4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Ba trường này có ba vai trò khác nhau:

| Trường | Vai trò |
|---|---|
| `class_id` | Mã số để máy xử lý, không mơ hồ |
| `class_name` | Tên để người đọc hiểu |
| `taxonomy_name` | Cho biết dùng bảng tra nào |

Số class_id chỉ có nghĩa khi biết taxonomy. Trong dữ liệu, class_id=5 ở detection là bus (taxonomy COCO-80), còn ở phân loại class_id=468 là cab (ImageNet-1K). ImageNet-1K cũng có lớp mang id 5, nhưng là một lớp khác. Nếu chỉ ghi tên, người đọc sau này cũng khó biết đang dùng danh sách nào và phiên bản nào.

### 1.5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?

Guideline cần quy định các điểm sau:

- **Tiêu chí chọn:** gán theo vật thể chiếm nhiều diện tích nhất, vật ở trung tâm, hay chủ đề chung của cả cảnh.
- **Cách xử lý khi hai lớp ngang nhau:** ví dụ ảnh traffic có xe buýt, ô tô, xe máy và người đi bộ. Cần một thứ tự ưu tiên để hai người gán nhãn chọn giống nhau.
- **Lớp nào dùng cho trường hợp nào:** cab, minibus và police_van dễ lẫn nhau, nên cần ví dụ minh họa.
- **Trường hợp không có lớp phù hợp:** nếu danh sách không có lớp đúng nghĩa thì annotator làm gì (chọn lớp gần nhất, hay đưa lên escalation).

### 1.6. Vì sao không được lấy `score` của model làm ground truth?

score là mức độ tin của chính model vào dự đoán của nó. Nó không do người gán nhãn xác nhận theo guideline, nên không chứng minh dự đoán đúng.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.

### 2.1. Chép một record

- `class_name` = person
- `score` = 0.912625
- `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
- `bbox_width` = 113.58
- `bbox_height` = 279.68

### 2.2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

| Thành phần | Giá trị |
|---|---|
| Góc trên-trái | x = 385.33, y = 69.24 |
| Góc dưới-phải | x = 498.92, y = 348.92 |
| Rộng | 113.58 |
| Cao | 279.68 |

### 2.3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

| Ngưỡng | Số prediction |
|---|---:|
| 0.20 | 17 |
| 0.60 | 6 |

### 2.4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?

Khi hạ ngưỡng, ví dụ từ 0.35 xuống 0.20, reviewer phải kiểm tra thêm 6 hộp mới, trong đó có 3 hộp spoon. Họ cần xem từng hộp có thật là vật đó không, và các dụng cụ treo trên tường có bị nhận nhầm thành thìa không.

### 2.5. Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Vẽ hộp sát biên của phần nhìn thấy được của vật thể, bao hết mọi phần nhìn thấy nhưng không lấy thêm nền; với vật bị che một phần thì chỉ vẽ phần nhìn thấy.

### 2.6. Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Tôi chọn person 0.61 ở mép trái, box [0.12, 263.18, 61.55, 310.78]. Hộp này chạm sát mép trái ảnh (x_min = 0.12), và nhìn ảnh chỉ thấy một phần cánh tay đang nằm trên bàn bột. Việc vật bị cắt ở mép ảnh nói chung có thể viết thành Guideline vif gặp ở nhiều ảnh. Tuy nhiên, một cánh tay chưa thể khẳng định đây có là "person" hay không nên tôi sẽ lên hỏi Lab Coach.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.

### 3.1. Chép một record

- `instance_id` = kitchen-001
- `class_name` = person
- `score` = 0.899318
- số điểm polygon = `polygon_point_count` = 348
- 3–5 điểm đầu của `polygon_xy` = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

### 3.2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

polygon cho biết hình dạng thật và diện tích thật của vật

### 3.3. `instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?

- instance_id chỉ đánh số thứ tự các vật thể trong bộ output của bài lab, để phân biệt được từng vật.
- Class ID nói “vật này thuộc loại nào”, còn instance ID nói “đây là cá thể số mấy”.Nó cũng không phải tracking ID. Tracking ID dùng để theo dõi cùng một vật qua nhiều khung hình của video, còn ở đây mỗi ảnh được xử lý riêng, không có liên kết giữa các ảnh

### 3.4. Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.

Vẽ biên mask sát mép thật của vật thể, bao hết phần nhìn thấy nhưng không lấy nền xung quanh; khi hai vật chồng lên nhau, vật ở phía trước giữ phần chồng, và mỗi điểm ảnh chỉ thuộc về một instance.

### 3.5. Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- Chọn cụm bàn (dining table) và các vật đặt trên bàn / vật chồng lên bàn
- Trên ảnh PNG, mask bàn (màu xanh) phủ cả mặt bàn lẫn bột trên đó, trong khi các bowl (hồng) nằm trên mặt bàn. Phần chồng nhau này đặt ra hai câu hỏi: mặt bàn có bao gồm vật đặt trên nó hay không, và vùng giáp giữa lò và bàn thuộc về vật nào.
- chọn guideline vì:

  **1.** Đây là tình huống lặp lại ở mọi ảnh có vật đặt trên bề mặt hoặc vật chồng nhau.

  **2.** Có thể giải quyết bằng một quy tắc rõ ràng, như quy tắc ở 3.4 (“vật ở phía trước giữ phần chồng, mỗi điểm ảnh thuộc một instance”).

  **3.** Không cần Lab Coach quyết định từng ảnh.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn cho **cả ảnh**: `class_id` + `class_name` + `taxonomy_name` (vd. 468, cab, ImageNet-1K). Không có tọa độ. | Ảnh `traffic` có nhiều xe buýt, ô tô, người nhưng hạng 1 là `cab` (score 0.51). Ảnh `kitchen` hạng 1 là `gong` (0.42), hạng 2 là `dining_table`. Chưa rõ nhãn nào hợp lý. | Chọn **một** lớp theo tiêu chí của guideline (vd. vật chính nhất). Nếu danh sách không có lớp khớp thì đưa lên hỏi Lab Coach. | Nhãn có đúng guideline không, hai annotator có chọn giống nhau không, có thuộc đúng taxonomy không. |
| Phát hiện vật thể | Mỗi vật thể một bản ghi: class + hộp `bbox_xyxy` = [x_min, y_min, x_max, y_max] (pixel). | `person 0.61` ở mép trái, hộp [0.12, 263.18, 61.55, 310.78], có vẻ chỉ thấy một phần cánh tay. Hạ ngưỡng từ 0.35 xuống 0.20 thêm 6 hộp (3 `spoon`, `potted plant`, `dining table`, `bottle`). | Vẽ hộp sát phần nhìn thấy của từng vật, đúng class. Ca khó như "cánh tay có phải person không" thì escalation. | Từng hộp: đúng class không, có khít không, có bỏ sót hoặc trùng lặp không, vật bị cắt mép có xử lý nhất quán không. |
| Instance segmentation | Mỗi vật thể một bản ghi: class + polygon `polygon_xy` (các điểm pixel của biên) + `instance_id` riêng. | `dining table` (kitchen-005): mask chỉ chiếm khoảng 55% hộp, hộp bàn chồng lên hộp lò (kitchen-007) và chứa các `bowl`. `spoon` mảnh: polygon chỉ khoảng 24% và 37% diện tích hộp (ước tính từ JSON). | Vẽ biên sát mép thật; khi chồng nhau thì vật phía trước giữ phần chồng, mỗi điểm ảnh thuộc một instance. | Biên có bám vật không, hai mask có chồng nhau không, vật mảnh có bị vẽ lệch không, số instance có khớp số vật nhìn thấy không. |

---

## 5. An toàn dữ liệu

### 5.1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Chỉ dùng ba ảnh COCO công khai đã được notebook cố định và kiểm tra checksum, không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai, và chỉ để họ tên, MSSV trong tên repository chứ không ghi vào báo cáo hay output.

### 5.2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Tôi sẽ dừng lại, không tải lên hay xử lý tiếp, và báo ngay cho Lab Coach (cũng như GV nếu dữ liệu đã bị đưa lên GitHub); không tự quyết định cách xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
