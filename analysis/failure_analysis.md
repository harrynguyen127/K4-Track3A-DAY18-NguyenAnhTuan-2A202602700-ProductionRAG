# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Anh Tuấn

**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8417 | 0.7298 | -0.1119 |
| Answer Relevancy | 0.7614 | 0.7507 | -0.0107 |
| Context Precision | 0.9250 | 0.9542 | +0.0292 |
| Context Recall | 0.9250 | 0.8000 | -0.1250 |

Δ = Production − Naive Baseline, trên thang điểm 0–1; mỗi báo cáo đánh giá 20 câu hỏi. Context Precision tăng nhưng Context Recall và Faithfulness giảm: context được chọn có thể liên quan nhưng chưa đủ bằng chứng để trả lời đầy đủ. Đây là nhận định ở mức tổng hợp, chưa xác định được context của từng câu.

### Phạm vi dữ liệu và cách xếp hạng

- Nguồn điểm: `reports/ragas_report.json` và `reports/naive_baseline_report.json`.
- `ragas_report.json` chỉ lưu `aggregate`, `num_questions` và 10 mục `failures`; không lưu `per_question`, câu trả lời thực tế hay retrieved contexts. Vì vậy, chưa thể xem đủ 4 metric cho từng câu trong 20 câu hỏi hoặc trích nguyên văn output của lần chạy.
- Theo `failure_analysis()` trong `src/m4_eval.py`, `failures[].score` là **trung bình của 4 metric**, được sắp tăng dần. `worst_metric` là tên metric thấp nhất, nhưng điểm riêng của metric đó không được lưu. Không được diễn giải `score` thành điểm của `worst_metric`.
- `diagnosis` được gán theo tên metric bằng bảng quy tắc; nhãn “LLM hallucinating” là tín hiệu để điều tra, chưa chứng minh nguyên nhân gốc hoặc nội dung sai cụ thể.
- Expected được đối chiếu với `test_set.json` và tài liệu trong `data/`. Các giả thuyết bên dưới dựa trên mã nguồn hiện tại; cần log answer/context của lần đánh giá để xác nhận. Không chạy lại pipeline hoặc RAGAS trong lần phân tích này.

### 10 câu có điểm trung bình thấp nhất được lưu trong báo cáo

| Hạng | Câu hỏi | Điểm trung bình | Worst metric |
|------|---------|-----------------|--------------|
| 1 | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.3333 | faithfulness |
| 2 | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.3750 | faithfulness |
| 3 | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.4517 | faithfulness |
| 4 | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.7419 | faithfulness |
| 5 | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.7500 | answer_relevancy |
| 6 | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.7894 | faithfulness |
| 7 | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.8156 | faithfulness |
| 8 | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu? | 0.8255 | faithfulness |
| 9 | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.8428 | context_recall |
| 10 | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | 0.8503 | context_recall |

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban (Director) phê duyệt vì 30 triệu thuộc khoảng 5–50 triệu VNĐ. Phải có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất; đính kèm ít nhất 3 báo giá vì đơn hàng trên 10 triệu. Nguồn: `data/mua_sam.md`, các mục “Thẩm quyền phê duyệt”, “Quy trình đề xuất” và “Lưu ý đặc biệt”. Câu hỏi không nêu tình huống khẩn cấp nên không áp dụng ngoại lệ bỏ qua 3 báo giá.
- **Got:** Báo cáo không lưu answer/context; không thể xác định hệ thống đã nêu sai người phê duyệt hay thiếu điều kiện nào. Chẩn đoán tự động: “LLM hallucinating”.
- **Worst metric:** `faithfulness`; **điểm trung bình 4 metric: 0.3333**. Điểm faithfulness riêng không có trong báo cáo.
- **Error Tree:** Output có cảnh báo faithfulness → Context đủ cả ba yêu cầu? Chưa biết; cần xem các chunk đã chọn → Query rõ số tiền và thiết bị CNTT; pipeline dùng nguyên query, không có bước rewrite → Nếu context thiếu, sửa M1/M3 và mở rộng parent; nếu đủ mà output vẫn sai, sửa generation prompt.
- **Root cause:** Giả thuyết chính là thiếu bằng chứng khi tổng hợp nhiều điều kiện. Kiểm tra chia chunk cục bộ cho thấy quy tắc Director, yêu cầu 3 báo giá và xác nhận CNTT nằm ở các child khác nhau. `build_pipeline()` chỉ index children; `run_query()` lấy top-3 sau rerank (`RERANK_TOP_K = 3`) và không lấy lại parent. Chưa có bằng chứng rằng cả ba child cần thiết đều được chọn trong lần chạy. Prompt hiện tại chỉ yêu cầu dựa trên context, chưa yêu cầu kiểm tra từng ý và dẫn nguồn.
- **Suggested fix:** Retrieve child rồi mở rộng parent theo khóa kết hợp `source` và `parent_id`; giữ đủ ba phần của chính sách mua sắm và loại context trùng. Yêu cầu output nêu lần lượt người phê duyệt, xác nhận CNTT, báo giá, kèm bằng chứng; thiếu ý nào thì báo thiếu dữ liệu cho ý đó. Đặt temperature thấp để giảm biến động. Xác nhận sửa bằng answer/context được lưu và kiểm tra đủ ba điều kiện trên câu này.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024 trong bộ dữ liệu: 15 ngày cơ bản + 9 ÷ 3 = 3 ngày thâm niên, tổng **18 ngày phép năm**; Senior (P3–P4) có lương gross **20–35 triệu VNĐ/tháng**. Nguồn: `data/nghi_phep_nam_v2024.md` và `data/bang_luong_2024.md`.
- **Got:** Không có answer/context được lưu; chưa biết output sai số ngày, khung lương hay dùng chính sách cũ. Chẩn đoán tự động: “LLM hallucinating”.
- **Worst metric:** `faithfulness`; **điểm trung bình 4 metric: 0.3750**. Điểm faithfulness riêng không có trong báo cáo.
- **Error Tree:** Output có cảnh báo faithfulness → Context chứa phép năm v2024, quy tắc thâm niên và dòng lương Senior? Chưa biết → Query rõ nhưng gồm hai nhu cầu; không có rewrite/tách truy vấn → Thiếu nguồn hoặc lẫn phiên bản: sửa retrieval; đủ nguồn mà tính/suy luận sai: sửa generation.
- **Root cause:** Giả thuyết là top-3 child không bao phủ hai tài liệu hoặc trộn chính sách phép năm 2023 và 2024. `load_documents()` nạp cả hai phiên bản; metadata ban đầu chỉ có `source`, chưa trích xuất trạng thái thay thế/ngày hiệu lực thành trường riêng. Luồng tìm kiếm hiện tại không lọc phiên bản. Đây là rủi ro thấy trong mã nguồn, chưa xác nhận phiên bản nào đã vào context của lần đánh giá.
- **Suggested fix:** Tách nhu cầu thành truy vấn phép năm/thâm niên và truy vấn lương Senior, hợp nhất bằng chứng từ hai nguồn. Lưu version/effective_date/status và ưu tiên chính sách v2024 thay thế v2023 theo nội dung tài liệu. Yêu cầu hiển thị phép tính `15 + floor(9/3) = 18`, khung lương gross và nguồn cho từng ý. Kiểm tra lại cả số ngày, mức lương và phiên bản được sử dụng.

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** **Tổng Giám đốc (CEO)** phê duyệt vì 55.000.000 VNĐ > 50.000.000 VNĐ. Nguồn: `data/mua_sam.md`, bảng “Thẩm quyền phê duyệt”.
- **Got:** Không có answer/context được lưu; không thể kết luận hệ thống đã trả lời Director hay chức danh khác. Chẩn đoán tự động: “LLM hallucinating”.
- **Worst metric:** `faithfulness`; **điểm trung bình 4 metric: 0.4517**. Điểm faithfulness riêng không có trong báo cáo.
- **Error Tree:** Output có cảnh báo faithfulness → Context có nguyên cặp điều kiện >50 triệu và CEO? Chưa biết từ lần chạy; kiểm tra chunk cục bộ thấy cặp này bị tách → Query rõ, không cần rewrite để hiểu số tiền → Ưu tiên sửa M1/parent expansion; nếu context đủ thì kiểm tra áp dụng ngưỡng trong generation.
- **Root cause:** Đã xác nhận một lỗi cấu trúc khi chạy riêng hàm chunking hiện tại trên tài liệu nguồn: child 1 kết thúc bằng `| Trên **50.000.000 VNĐ** |`, còn child 2 bắt đầu bằng `Tổng Giám đốc (CEO) |`. Giới hạn 256 ký tự cắt bảng và làm mất liên kết điều kiện–người phê duyệt trong một child. Pipeline chưa mở rộng parent nên có nguy cơ chỉ đưa một nửa quy tắc vào context. Chưa có log để chứng minh lỗi này gây ra output của lần đánh giá.
- **Suggested fix:** Dùng `chunk_structure_aware()` cho bảng mua sắm hoặc giữ nguyên header và từng dòng bảng khi chia; mở rộng parent sau retrieval để khôi phục đầy đủ bảng. Prompt yêu cầu so sánh `55 triệu > 50 triệu` rồi lấy chức danh từ đúng dòng, không tự bổ sung cấp phê duyệt. Khi xác nhận sửa, kiểm tra context giữ nguyên dòng >50 triệu → CEO và output nêu CEO.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** **Giám đốc điều hành (CEO)** phê duyệt vì 20 ngày thuộc khoảng 16–30 ngày. Lưu ý theo đáp án chuẩn: nghỉ trên 14 ngày không lương phải tự đóng phần bảo hiểm của mình. Nguồn: `data/nghi_phep_khong_luong.md`, các mục “Quy trình phê duyệt” và “Ảnh hưởng đến phúc lợi”.
- **Got:** Không có answer/context được lưu; chưa xác định sai cấp phê duyệt, bỏ sót lưu ý hay bổ sung thông tin không có căn cứ. Chẩn đoán tự động: “LLM hallucinating”.
- **Worst metric:** `faithfulness`; **điểm trung bình 4 metric: 0.7419**. Điểm faithfulness riêng không có trong báo cáo.
- **Error Tree:** Output có cảnh báo faithfulness → Context có chính sách nghỉ không lương, đoạn 16–30 ngày và lưu ý bảo hiểm? Chưa biết → Query rõ; cần giữ “không lương” và “20 ngày” → Nếu thiếu/lẫn nguồn thì sửa retrieval; nếu đủ thì sửa áp dụng khoảng và giới hạn phát biểu.
- **Root cause:** Giả thuyết là nhầm loại nghỉ/cấp phê duyệt hoặc thêm phát biểu không được hỗ trợ. Kiểm tra chunk cục bộ cho thấy ba mức phê duyệt vẫn nằm nguyên trong child 2; lưu ý bảo hiểm nằm ở child 3. Vì vậy chưa có căn cứ quy lỗi cắt dòng phê duyệt cho câu này. Cần xác nhận child 2/3 có được chọn và liệu context có lẫn phép năm hoặc quy định nghỉ của nhân viên thử việc.
- **Suggested fix:** Ưu tiên đúng tài liệu nghỉ không lương, đưa đoạn phê duyệt và phúc lợi vào context. Yêu cầu kết luận CEO từ điều kiện `16 ≤ 20 ≤ 30`, đặt lưu ý bảo hiểm sau câu trả lời chính và chỉ nêu nội dung có nguồn. Kiểm tra câu trả lời không thay CEO bằng trưởng phòng/Giám đốc Nhân sự hoặc tự thêm quy trình chưa được tài liệu quy định.

### #5
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** **Không.** Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI; theo tài liệu, được tham gia bảo hiểm xã hội bắt buộc. Gói PVI áp dụng cho nhân viên chính thức. Nguồn: `data/thu_viec.md`, mục “Quyền lợi trong thử việc”; đối chiếu `data/bao_hiem_suc_khoe.md`, mục “Bảo hiểm cho nhân viên”.
- **Got:** Không có answer/context được lưu; chưa biết output trả lời sai Có/Không hay chỉ mô tả hạn mức/quyền lợi PVI mà chưa trả lời điều kiện thử việc. Chẩn đoán tự động: “Answer doesn't match question”.
- **Worst metric:** `answer_relevancy`; **điểm trung bình 4 metric: 0.7500**. Điểm answer_relevancy riêng không có trong báo cáo.
- **Error Tree:** Output có cảnh báo lệch câu hỏi → Context có điều khoản loại trừ nhân viên thử việc? Chưa biết → Query rõ đối tượng “thử việc” và loại bảo hiểm “PVI”, không có rewrite → Thiếu điều khoản: sửa retrieval; đủ điều khoản nhưng trả lời lan man: sửa prompt.
- **Root cause:** Giả thuyết là retrieval ưu tiên mô tả PVI chung thay vì điều kiện đối tượng, hoặc generation không trả lời Có/Không trực tiếp. Chia chunk cục bộ cho thấy câu loại trừ PVI nằm ở child 3 của `thu_viec.md`, tách khỏi tiêu đề tài liệu. Prompt chưa quy định cấu trúc câu trả lời về điều kiện hưởng phúc lợi. Không thể kết luận đã xảy ra nhầm PVI với bảo hiểm xã hội nếu chưa có output thực tế.
- **Suggested fix:** Bảo đảm context chứa câu “chưa được hưởng gói bảo hiểm sức khỏe PVI”, giữ tên tài liệu/đối tượng trong child hoặc mở rộng parent. Prompt yêu cầu mở đầu “Không”, giải thích một câu về thử việc và phân biệt PVI với bảo hiểm xã hội bắt buộc. Kiểm tra câu trả lời bám đúng điều kiện thử việc, không chỉ liệt kê hạn mức/quy trình PVI.

## Case Study (cho presentation)

**Question chọn phân tích:** “Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?” (#3). Chọn vì đã tái hiện được việc cắt dòng bảng bằng hàm chunking cục bộ, giúp minh họa một lỗi kỹ thuật cụ thể.

**Error Tree walkthrough:**

1. Output đúng? → Đáp án chuẩn là CEO. Báo cáo đánh dấu faithfulness thấp nhất, điểm trung bình 0.4517; chưa có output để chỉ ra phát biểu sai cụ thể.
2. Context đúng? → Chưa có retrieved contexts của lần chạy. Tuy nhiên, child 1 giữ điều kiện >50 triệu, child 2 giữ chức danh CEO; `run_query()` chỉ dùng child sau rerank. Cần log để kiểm tra cả hai nửa có vào context hay không.
3. Query rewrite OK? → Pipeline không có bước rewrite. Câu hỏi đã rõ số tiền và nhu cầu phê duyệt; chưa thấy lý do ưu tiên rewrite trước việc bảo toàn bảng.
4. Fix ở bước: → M1 giữ nguyên bảng/dòng; retrieval mở rộng parent bằng `source + parent_id`; generation dẫn đúng quy tắc >50 triệu → CEO. Xác nhận bằng context đầy đủ và answer thực tế trước khi kết luận sửa thành công.

**Nếu có thêm 1 giờ, sẽ optimize:**

- **0–15 phút:** Bổ sung lưu `per_question` ở `save_report()` gồm question, answer, contexts, ground_truth và đủ 4 metric; thêm nguồn/rank trước và sau rerank để xác nhận nhánh Error Tree. Đây là đề xuất, chưa thay đổi mã nguồn trong lần phân tích này.
- **15–35 phút:** Bảo toàn bảng mua sắm và triển khai parent expansion; lưu parent bằng khóa kết hợp nguồn tài liệu và parent ID vì `parent_0` hiện có thể lặp giữa các tài liệu.
- **35–45 phút:** Ràng buộc câu trả lời theo từng ý, nguồn chứng minh và cách xử lý thiếu dữ liệu; câu Có/Không phải trả lời trực tiếp, câu ngưỡng số phải nêu điều kiện áp dụng.
- **45–60 phút:** Khi thực hiện tối ưu, đánh giá lại trên bộ tài liệu lab, ưu tiên 5 câu trên rồi so sánh đủ 20 câu với báo cáo hiện tại. Theo dõi Faithfulness và Context Recall cùng Context Precision, tránh kết luận cải thiện chỉ từ một metric; ghi nhận thêm chi phí/độ dài context.