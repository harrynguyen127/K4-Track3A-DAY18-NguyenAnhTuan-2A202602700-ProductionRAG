# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Anh Tuấn  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 2026-10-04  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Hàm tách văn bản theo ranh giới câu/đoạn, dùng `all-MiniLM-L6-v2` để tính cosine similarity giữa hai câu liên tiếp; similarity < 0.85 thì mở chunk mới. Cách này nhóm các câu gần nhau về ngữ nghĩa thay vì chỉ gom đoạn theo giới hạn 500 ký tự như basic; tăng threshold có xu hướng tạo nhiều chunk hơn. Chưa có số liệu lưu để so sánh số chunk semantic với basic. Pipeline production hiện dùng hierarchical (parent 2048, child 256 ký tự), nên điểm RAGAS hiện tại không phản ánh riêng hiệu quả semantic chunking. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 tìm theo từ khóa sau xử lý tiếng Việt, còn Dense dùng `BAAI/bge-m3` và cosine similarity trong Qdrant để tìm theo ngữ nghĩa. Mỗi nhánh lấy tối đa 20 kết quả; RRF cộng `1/(60 + rank)` với rank bắt đầu từ 1, gộp các kết quả trùng text và giữ tối đa 20 candidate. RRF dùng thứ hạng nên không cần chuẩn hóa điểm BM25 và Dense; chunk xuất hiện ở cả hai nhánh được cộng thêm đóng góp. Chưa có đánh giá tách riêng từng nhánh để kết luận mức cải thiện do fusion. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Model `BAAI/bge-reranker-v2-m3` chấm trực tiếp từng cặp query–chunk, sắp xếp theo `rerank_score` và chọn top-3 từ tối đa 20 candidate của hybrid search. Bước này đánh giá độ liên quan sâu hơn nhưng thêm chi phí suy luận. Giữ 3 child nhỏ mà không mở rộng parent có nguy cơ thiếu bằng chứng cho câu hỏi nhiều điều kiện hoặc nhiều tài liệu. Context Precision của toàn pipeline tăng nhưng Context Recall giảm; chưa thể quy thay đổi này riêng cho reranking. Có hàm `benchmark_reranker()` nhưng chưa có log latency để điền số ms thực đo. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Hàm đánh giá Faithfulness (câu trả lời có căn cứ trong context), Answer Relevancy (đúng trọng tâm câu hỏi), Context Precision (context liên quan được xếp ưu tiên) và Context Recall (context bao phủ thông tin trong đáp án chuẩn). Trên 20 câu hỏi trong `reports/ragas_report.json`, production đạt lần lượt 0.7298 / 0.7507 / 0.9542 / 0.8000; baseline trong `reports/naive_baseline_report.json` đạt 0.8417 / 0.7614 / 0.9250 / 0.9250. Precision tăng 0.0292 nhưng Faithfulness giảm 0.1119 và Recall giảm 0.1250: chưa thể kết luận production tốt hơn tổng thể. Báo cáo không lưu answer/context từng câu, nên cần bổ sung log để xác nhận nguyên nhân. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Enrichment diễn ra trước indexing: prepend một câu mô tả nguồn/chủ đề vào chunk để bổ sung ngữ cảnh cho cả BM25 và Dense. Pipeline dùng chế độ combined; khi API thành công, một lần gọi `gpt-4o-mini` trả về summary, questions, context và metadata cho mỗi chunk. Tuy nhiên, text được index hiện chỉ là context + chunk gốc; summary và HyQA chưa được index riêng. Context được tạo từ tên tài liệu và chunk, không có toàn bộ tài liệu, nên không khôi phục được quy tắc đã bị cắt sang chunk khác. Chưa có đánh giá bật/tắt M5 để đo mức giảm retrieval failure trong lab. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - Chưa có log exception được lưu để trích nguyên văn thông báo lỗi. Khó khăn có bằng chứng trong lab là chất lượng trả lời chưa cải thiện đồng đều: trên 20 câu hỏi, Faithfulness giảm từ 0.8417 xuống 0.7298 và Context Recall giảm từ 0.9250 xuống 0.8000, dù Context Precision tăng từ 0.9250 lên 0.9542. Vì vậy, không thể kết luận pipeline production tốt hơn chỉ dựa vào Precision.
  - Báo cáo gán nhãn `LLM hallucinating` cho câu hỏi “Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?”. Đây là nhãn chẩn đoán tự động theo metric thấp nhất, không phải exception hay bằng chứng xác nhận nội dung sai cụ thể. Điểm 0.4517 của câu này là trung bình 4 metric, không phải điểm Faithfulness riêng.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Qua đối chiếu `analysis/failure_analysis.md`, tài liệu `data/mua_sam.md` và mã nguồn M1, một lỗi cấu trúc đã được ghi nhận: giới hạn child 256 ký tự làm điều kiện `Trên **50.000.000 VNĐ**` và người phê duyệt `Tổng Giám đốc (CEO)` bị tách sang hai child khác nhau. Điều này làm mất liên kết điều kiện–người phê duyệt trong một chunk.
  - Kiểm tra luồng xử lý trong `src/pipeline.py` cho thấy `build_pipeline()` chỉ index children; `run_query()` lấy top-3 sau reranking và chưa mở rộng về parent. Do đó, context có nguy cơ thiếu một nửa quy tắc hoặc thiếu bằng chứng cho câu hỏi nhiều điều kiện. Tuy nhiên, báo cáo không lưu answer và retrieved contexts nên chưa thể khẳng định lỗi cắt bảng là nguyên nhân trực tiếp của kết quả đánh giá này.
  - Hướng debug tiếp theo là lưu question, answer, contexts, ground_truth, nguồn, thứ hạng trước/sau rerank và đủ 4 metric cho từng câu. Nếu context thiếu quy tắc thì sửa chunking/retrieval; nếu context đầy đủ nhưng answer vẫn sai thì kiểm tra generation prompt và cách áp dụng ngưỡng số.
  - Cách khắc phục dự kiến: dùng structure-aware chunking để bảo toàn bảng hoặc từng dòng kèm header; mở rộng parent bằng khóa kết hợp `source + parent_id` và loại context trùng. Prompt cần yêu cầu trả lời theo bằng chứng, dẫn nguồn và báo thiếu thông tin khi context chưa đủ. Các sửa đổi này chưa được triển khai trong lần phân tích; cần kiểm tra context giữ đầy đủ quy tắc >50 triệu → CEO, answer thực tế và đánh giá lại cùng bộ câu hỏi trước khi kết luận sửa thành công.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Tôi cần hiểu rõ hơn sự đánh đổi giữa Context Precision và Context Recall, cách triển khai parent–child retrieval và cách phân biệt lỗi retrieval với lỗi generation. Context liên quan chưa chắc đã bao phủ đủ bằng chứng để trả lời toàn bộ câu hỏi.
  - Tôi sẽ đọc lại các module M1–M4, theo dõi luồng query → candidates → reranking → context → answer và thực hiện các thí nghiệm thay đổi từng thành phần riêng biệt trên cùng bộ câu hỏi. Đồng thời, bổ sung log chi tiết và kiểm tra thủ công các câu điểm thấp để đối chiếu chẩn đoán tự động với bằng chứng thực tế.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: AI Nutrition

#### 1. Hiện trạng
- **Pipeline hiện tại:** Kiến trúc và kết quả benchmark của AI Nutrition chưa được mô tả trong reflection này, nên chưa có cơ sở xác nhận các thành phần đang triển khai. Kế hoạch dự kiến là xây dựng hoặc chuẩn hóa baseline theo luồng: tài liệu dinh dưỡng có nguồn rõ ràng → trích xuất và chuẩn hóa dữ liệu → chunking → embedding và indexing → retrieval → LLM trả lời kèm nguồn. Baseline sẽ được ghi nhận trước khi bổ sung các kỹ thuật production đã học trong lab.
- **Vấn đề / Bottlenecks đang gặp:** Chưa có số liệu để xác định bottleneck thực tế của AI Nutrition. Các rủi ro cần kiểm chứng gồm mất liên kết giữa tên thực phẩm, hàm lượng và đơn vị khi chia bảng; thiếu ngữ cảnh về khẩu phần hoặc đối tượng áp dụng; trộn phiên bản tài liệu; và câu trả lời có thông tin chưa được nguồn hỗ trợ. Cần đo thêm latency và chi phí trước khi chọn cấu hình.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Ưu tiên thử structure-aware chunking cho bảng thành phần dinh dưỡng, giữ liên kết giữa tên thực phẩm, giá trị, đơn vị và khẩu phần tham chiếu. Với tài liệu dài, thử hierarchical chunking: tìm child rồi mở rộng parent để bổ sung ngữ cảnh, trong giới hạn độ dài context. So sánh với basic chunking trên cùng bộ câu hỏi để kiểm tra độ bao phủ bằng chứng.
2. **Search retrieval:** Thử Hybrid Search kết hợp BM25 và Dense bằng RRF, dùng cấu hình trong lab làm điểm xuất phát. BM25 phục vụ tìm tên thực phẩm và thuật ngữ chính xác; Dense phục vụ câu hỏi diễn đạt theo ngữ nghĩa. Đánh giá riêng BM25, Dense và Hybrid trên cùng benchmark để xác định mức đóng góp của fusion, thay vì mặc định Hybrid luôn tốt hơn.
3. **Reranking:** Thử `BAAI/bge-reranker-v2-m3` đang dùng trong lab. So sánh bật/tắt reranking và top-k = 3, 5; kiểm tra bằng chứng cần thiết còn lại sau rerank và sau parent expansion. Chọn cấu hình dựa trên chất lượng cùng latency, không cố định top-3 khi chưa đo khả năng bao phủ câu hỏi nhiều ý.
4. **Evaluation:** Xây dựng benchmark ban đầu 30 câu: 25 câu có đáp án trong nguồn, gồm tra cứu đơn giản, đọc bảng, so sánh và tổng hợp nhiều đoạn; 5 câu không có đáp án để kiểm tra khả năng báo thiếu thông tin. Mỗi câu có đáp án hoặc hành vi mong đợi và nguồn bằng chứng tương ứng. Dùng 4 metric RAGAS cho nhóm câu có đáp án; kiểm tra thủ công số liệu, đơn vị, khẩu phần, trích dẫn và khả năng báo thiếu dữ liệu cho nhóm còn lại. Lưu answer, contexts, nguồn, điểm từng câu, latency và chi phí; giữ cố định bộ câu hỏi và điều kiện đánh giá khi so sánh các cấu hình.
5. **Enrichment:** Ưu tiên metadata về nguồn, mục tài liệu, phiên bản, đối tượng áp dụng, đơn vị và khẩu phần tham chiếu khi có trong tài liệu. Thử contextual prepend bằng tên tài liệu và tiêu đề mục để bổ sung ngữ cảnh cho chunk; kiểm tra mô tả được tạo không thêm thông tin ngoài nguồn trước khi index. Chỉ thử thêm HyQA sau khi đã có baseline và đánh giá riêng bật/tắt từng loại enrichment để xác định lợi ích và chi phí.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa bộ tài liệu và metadata; xây dựng benchmark 30 câu với nguồn bằng chứng; triển khai hoặc ghi nhận baseline và lưu log đầy đủ. Thử structure-aware chunking và parent expansion, kiểm tra thủ công các bảng để bảo đảm giá trị không bị tách khỏi tên thực phẩm, đơn vị và khẩu phần. Đầu ra: bộ dữ liệu đánh giá, báo cáo baseline và danh sách lỗi ban đầu.
- **Tuần 2:** So sánh BM25, Dense và Hybrid; thử reranking và contextual prepend từng bước để xác định tác động riêng. Đánh giá trên cùng benchmark, phân tích các câu điểm thấp và lựa chọn cấu hình theo chất lượng, latency và chi phí. Đầu ra: báo cáo so sánh, cấu hình được chọn và danh sách lỗi còn tồn tại.

**Tiêu chí đánh giá kết quả:** Không kết luận cải thiện chỉ từ Context Precision. Theo dõi đồng thời Faithfulness và Context Recall, kiểm tra tính đúng của số liệu/đơn vị và bằng chứng trích dẫn; nếu chỉ số giảm, cần phân tích các câu bị ảnh hưởng trước khi chọn cấu hình. Mọi nhận định về cải thiện phải dựa trên benchmark và log được lưu, thay vì xem mục tiêu dự kiến là kết quả đã đạt được.
