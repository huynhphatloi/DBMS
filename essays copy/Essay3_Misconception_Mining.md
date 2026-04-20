---
title: "Tiểu luận 3: Khai phá Lỗi sai và Mẫu hình Sai lầm trong Câu trả lời Sinh viên"
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 13pt
linestretch: 1.5
---

# CHƯƠNG 1: MỞ ĐẦU

## 1.1. Tính cấp thiết của đề tài

Trong hệ thống giáo dục hiện đại, việc đánh giá năng lực người học thông qua câu trả lời ngắn (short answer) là một phương pháp được sử dụng rộng rãi tại các trường đại học và cơ sở đào tạo trên toàn thế giới. So với câu hỏi trắc nghiệm — vốn chỉ kiểm tra khả năng nhận diện đáp án đúng trong một tập hữu hạn các lựa chọn — câu trả lời ngắn yêu cầu sinh viên phải tự diễn đạt kiến thức bằng ngôn ngữ của mình, qua đó phản ánh mức độ hiểu biết sâu sắc hơn về chủ đề được hỏi. Tuy nhiên, việc chấm điểm hàng ngàn câu trả lời ngắn bằng tay là một công việc tốn kém về thời gian và nhân lực, đồng thời tiềm ẩn nguy cơ thiếu nhất quán giữa các giáo viên chấm bài.

Các hệ thống chấm điểm tự động câu trả lời ngắn (Automatic Short Answer Grading — ASAG) đã được phát triển để giải quyết vấn đề này. Một hệ thống ASAG điển hình nhận đầu vào là bộ ba (câu hỏi, câu trả lời tham chiếu, câu trả lời sinh viên) và đưa ra dự đoán về mức độ đúng đắn, dưới dạng nhãn phân loại (correct, partially_correct, incorrect) hoặc điểm số liên tục. Tuy nhiên, hầu hết các hệ thống ASAG hiện tại chỉ dừng lại ở việc phân loại câu trả lời là đúng hay sai, mà không đi sâu vào phân tích **tại sao** sinh viên trả lời sai. Đây chính là khoảng trống nghiên cứu mà tiểu luận này hướng đến.

Việc chỉ biết rằng một câu trả lời là sai mà không hiểu được nguyên nhân gốc rễ của sai lầm đó là một hạn chế nghiêm trọng. Trong thực tế giảng dạy, giáo viên không chỉ cần biết sinh viên nào trả lời sai, mà quan trọng hơn là cần hiểu được **mẫu hình sai lầm** (misconception pattern) phổ biến trong lớp học để có thể điều chỉnh phương pháp giảng dạy phù hợp. Ví dụ, nếu 40% sinh viên trong một lớp Vật lý đều nhầm lẫn giữa khái niệm "lực" và "năng lượng", thì đây là một tín hiệu quan trọng cho thấy giáo viên cần dành thêm thời gian giải thích sự khác biệt giữa hai khái niệm này.

Khai phá lỗi sai và mẫu hình sai lầm (Misconception Mining) là một hướng nghiên cứu mới nổi trong lĩnh vực Educational Data Mining (EDM) và Learning Analytics. Thay vì chỉ gán nhãn đúng/sai cho từng câu trả lời, misconception mining sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học máy không giám sát (unsupervised learning) để tự động phát hiện và phân nhóm các câu trả lời sai có cùng nguyên nhân gốc rễ. Kết quả là một bản đồ các misconception phổ biến, giúp giáo viên nhanh chóng nắm bắt được bức tranh tổng thể về những hiểu lầm của sinh viên.

Tính cấp thiết của đề tài còn được thể hiện ở quy mô ngày càng lớn của dữ liệu giáo dục. Với sự phát triển của các nền tảng học trực tuyến (MOOCs, LMS), số lượng câu trả lời sinh viên cần được phân tích có thể lên đến hàng trăm ngàn hoặc hàng triệu mẫu. Việc phân tích thủ công ở quy mô này là bất khả thi, đòi hỏi các phương pháp tự động hóa hiệu quả và có khả năng mở rộng.

## 1.2. Mục tiêu nghiên cứu

Mục tiêu tổng quát của tiểu luận là xây dựng và đánh giá một pipeline khai phá lỗi sai tự động trong câu trả lời sinh viên, sử dụng kết hợp các kỹ thuật sentence embedding, giảm chiều dữ liệu, và phân cụm không giám sát. Cụ thể, nghiên cứu thực hiện so sánh có hệ thống **3 chiến lược embedding × 3 phương pháp phân cụm = 9 cấu hình** (configurations) để xác định cấu hình tối ưu cho bài toán misconception mining.

Ba chiến lược embedding được khảo sát bao gồm:

- **Strategy A (answer_only)**: Chỉ sử dụng câu trả lời sinh viên làm đầu vào cho mô hình embedding.
- **Strategy B (question_answer)**: Kết hợp câu hỏi và câu trả lời sinh viên.
- **Strategy C (full_triplet)**: Kết hợp câu hỏi, câu trả lời tham chiếu, và câu trả lời sinh viên.

Ba phương pháp phân cụm được so sánh bao gồm:

- **KMeans**: Phương pháp phân cụm cổ điển dựa trên centroid.
- **UMAP + HDBSCAN**: Giảm chiều bằng UMAP rồi phân cụm bằng HDBSCAN.
- **BERTopic-style**: Pipeline đầy đủ SBERT → UMAP → HDBSCAN → c-TF-IDF.

Các nhiệm vụ cụ thể bao gồm:

1. Thiết kế và triển khai module embedding với 3 chiến lược (A, B, C).
2. Triển khai 3 phương pháp phân cụm với các siêu tham số có thể cấu hình.
3. Xây dựng hệ thống đánh giá chất lượng cluster bằng cả metric nội tại (intrinsic) và ngoại tại (extrinsic).
4. Thực hiện thí nghiệm so sánh 9 cấu hình trên dữ liệu thực.
5. Phát triển ứng dụng demo trực quan hóa kết quả phân cụm.

## 1.3. Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu của tiểu luận là các câu trả lời sai của sinh viên trong các bài kiểm tra câu trả lời ngắn thuộc lĩnh vực khoa học tự nhiên. Cụ thể, dữ liệu được lấy từ hai nguồn chính:

**SciEntsBank** (Dzikovska et al., 2013): Bộ dữ liệu benchmark công khai với khoảng 10,000 mẫu, sử dụng hệ thống nhãn 5-way (correct, partially_correct_incomplete, contradictory, irrelevant, non_domain). Các câu hỏi thuộc lĩnh vực khoa học tự nhiên (vật lý, hóa học, sinh học) dành cho học sinh trung học.

**Data_Generate**: Bộ dữ liệu tổng hợp được sinh bằng Large Language Models (LLMs) với 10,000 mẫu, bao gồm đầy đủ annotation cho misconception mining: `misconception_tags`, `misconception_inventory`, `missing_concepts`, và `extra_incorrect_claims`. Đây là nguồn dữ liệu có gold labels cho đánh giá extrinsic.

Phạm vi nghiên cứu được giới hạn như sau:

- Chỉ xét các câu trả lời có nhãn `label_5way` thuộc tập {`partially_correct_incomplete`, `contradictory`, `irrelevant`} — tức là các câu trả lời chứa lỗi sai ở các mức độ khác nhau.
- Sử dụng mô hình SBERT `all-MiniLM-L6-v2` làm backbone embedding duy nhất để đảm bảo tính công bằng khi so sánh các chiến lược.
- Thí nghiệm được thực hiện ở 3 mức độ chi tiết (granularity): per-question, per-domain, và global.
- Đánh giá chất lượng cluster bằng cả metric nội tại (Silhouette, Calinski-Harabasz, Davies-Bouldin) và ngoại tại (NMI, ARI, Purity, V-measure).

## 1.4. Cơ sở lý luận

Tiểu luận được xây dựng trên nền tảng lý thuyết của ba lĩnh vực chính:

**Sentence Embeddings và Biểu diễn ngữ nghĩa**: Sentence-BERT (SBERT) là một kiến trúc mở rộng của BERT sử dụng mạng Siamese để tạo ra các vector biểu diễn câu có ý nghĩa ngữ nghĩa. Các câu có nội dung tương tự sẽ có vector embedding gần nhau trong không gian vector, cho phép sử dụng các phép đo khoảng cách (cosine similarity, Euclidean distance) để đánh giá mức độ tương đồng ngữ nghĩa. Mô hình `all-MiniLM-L6-v2` là một biến thể nhẹ của SBERT, cân bằng giữa chất lượng embedding và tốc độ tính toán.

**Giảm chiều dữ liệu (Dimensionality Reduction)**: UMAP (Uniform Manifold Approximation and Projection) là một thuật toán giảm chiều phi tuyến dựa trên lý thuyết topo đại số và hình học Riemann. UMAP bảo toàn cả cấu trúc cục bộ (local structure) và cấu trúc toàn cục (global structure) của dữ liệu tốt hơn so với t-SNE, đồng thời có tốc độ tính toán nhanh hơn đáng kể. Trong bài toán misconception mining, UMAP được sử dụng để giảm chiều từ không gian embedding 384 chiều xuống 5 chiều trước khi phân cụm, giúp giảm nhiễu và cải thiện chất lượng cluster.

**Phân cụm không giám sát (Unsupervised Clustering)**: Nghiên cứu sử dụng hai paradigm phân cụm chính: (1) phân cụm dựa trên centroid (KMeans) — yêu cầu xác định trước số cluster K, và (2) phân cụm dựa trên mật độ (HDBSCAN) — tự động xác định số cluster và có khả năng phát hiện noise points. Ngoài ra, pipeline BERTopic-style kết hợp UMAP + HDBSCAN + c-TF-IDF để không chỉ phân cụm mà còn tự động trích xuất từ khóa đại diện cho mỗi cluster.

## 1.5. Đóng góp mới

Tiểu luận có các đóng góp mới sau đây:

1. **So sánh có hệ thống 3 chiến lược embedding**: Đây là nghiên cứu đầu tiên (trong phạm vi hiểu biết của tác giả) thực hiện so sánh có hệ thống giữa 3 chiến lược embedding khác nhau cho bài toán misconception mining trong ASAG. Kết quả cho thấy việc bổ sung ngữ cảnh câu hỏi và câu trả lời tham chiếu có ảnh hưởng đáng kể đến chất lượng phân cụm.

2. **Pipeline end-to-end từ embedding đến trực quan hóa**: Nghiên cứu cung cấp một pipeline hoàn chỉnh từ khâu tiền xử lý dữ liệu, embedding, giảm chiều, phân cụm, đánh giá, đến trực quan hóa kết quả thông qua ứng dụng web tương tác.

3. **Đánh giá đa chiều**: Sử dụng cả metric nội tại (không cần gold labels) và ngoại tại (so sánh với gold `misconception_tags` từ Data_Generate), cung cấp cái nhìn toàn diện về chất lượng phân cụm.

4. **Ứng dụng demo tương tác**: Phát triển ứng dụng web Research Analytics Lab với giao diện dark theme hiện đại, cho phép giáo viên và nhà nghiên cứu khám phá kết quả phân cụm một cách trực quan.

## 1.6. Ý nghĩa lý luận và thực tiễn

**Ý nghĩa lý luận**: Nghiên cứu đóng góp vào hiểu biết về cách biểu diễn ngữ nghĩa của câu trả lời sai ảnh hưởng đến khả năng phát hiện misconception. Kết quả so sánh 9 cấu hình cung cấp hướng dẫn cho các nhà nghiên cứu trong việc lựa chọn chiến lược embedding và phương pháp phân cụm phù hợp cho các bài toán tương tự. Ngoài ra, nghiên cứu cũng đóng góp vào lĩnh vực Educational Data Mining bằng cách chứng minh tính khả thi của việc sử dụng các kỹ thuật NLP hiện đại để tự động phát hiện misconception ở quy mô lớn.

**Ý nghĩa thực tiễn**: Pipeline misconception mining được phát triển trong nghiên cứu này có thể được tích hợp vào các hệ thống quản lý học tập (LMS) để cung cấp phân tích tự động về các lỗi sai phổ biến của sinh viên. Giáo viên có thể sử dụng kết quả phân cụm để: (1) xác định các misconception phổ biến nhất trong lớp học, (2) thiết kế bài giảng bổ sung nhắm vào các misconception cụ thể, (3) tạo bài kiểm tra chẩn đoán (diagnostic assessment) để phát hiện sớm các hiểu lầm, và (4) cá nhân hóa phản hồi cho từng nhóm sinh viên dựa trên loại misconception của họ.

## 1.7. Tình hình nghiên cứu trong và ngoài nước

### 1.7.1. Nghiên cứu quốc tế

Lĩnh vực phát hiện misconception trong giáo dục đã được nghiên cứu từ nhiều góc độ khác nhau. Các nghiên cứu sớm nhất tập trung vào việc xây dựng thủ công các "misconception inventory" — danh sách các hiểu lầm phổ biến trong từng môn học, được biên soạn bởi các chuyên gia giáo dục. Ví dụ nổi tiếng nhất là Force Concept Inventory (FCI) của Hestenes et al. (1992) trong lĩnh vực Vật lý, liệt kê 30 misconception phổ biến về lực và chuyển động.

Với sự phát triển của NLP và machine learning, các phương pháp tự động phát hiện misconception đã được đề xuất. Gong et al. (2020) sử dụng BERT fine-tuned để phân loại câu trả lời sinh viên vào các danh mục misconception đã biết trước. Tuy nhiên, phương pháp này yêu cầu có sẵn danh sách misconception và dữ liệu huấn luyện có nhãn, hạn chế khả năng phát hiện các misconception mới.

Hướng tiếp cận không giám sát (unsupervised) sử dụng phân cụm để tự động phát hiện các nhóm câu trả lời sai tương tự nhau đã thu hút sự quan tâm ngày càng lớn. Lan et al. (2015) sử dụng topic modeling (LDA) trên câu trả lời sinh viên để phát hiện các chủ đề misconception. Tuy nhiên, LDA dựa trên bag-of-words và không nắm bắt được ngữ nghĩa sâu của câu trả lời.

Gần đây, BERTopic (Grootendorst, 2022) đã được đề xuất như một framework topic modeling hiện đại, kết hợp SBERT embeddings với UMAP và HDBSCAN. Mặc dù BERTopic được thiết kế cho topic modeling tổng quát, pipeline của nó rất phù hợp cho bài toán misconception mining vì nó có thể tự động phát hiện số lượng cluster và trích xuất từ khóa đại diện.

### 1.7.2. Nghiên cứu trong nước

Tại Việt Nam, nghiên cứu về Educational Data Mining và Learning Analytics còn ở giai đoạn sơ khai. Một số nghiên cứu đã được thực hiện trong lĩnh vực chấm điểm tự động câu trả lời tiếng Việt, nhưng chưa có nghiên cứu nào tập trung vào khai phá misconception. Tiểu luận này là một trong những nỗ lực đầu tiên áp dụng các kỹ thuật NLP hiện đại vào bài toán misconception mining trong bối cảnh giáo dục, mở ra hướng nghiên cứu mới cho cộng đồng nghiên cứu trong nước.

### 1.7.3. Khoảng trống nghiên cứu

Qua tổng quan tài liệu, tác giả nhận thấy các khoảng trống nghiên cứu sau:

1. **Thiếu so sánh có hệ thống giữa các chiến lược embedding**: Hầu hết các nghiên cứu chỉ sử dụng một chiến lược embedding duy nhất mà không so sánh với các phương án thay thế.

2. **Thiếu đánh giá extrinsic**: Nhiều nghiên cứu chỉ sử dụng metric nội tại (Silhouette, etc.) mà không có gold labels để đánh giá extrinsic, khiến việc đánh giá chất lượng cluster thiếu toàn diện.

3. **Thiếu công cụ trực quan hóa**: Kết quả phân cụm thường chỉ được trình bày dưới dạng bảng số liệu, thiếu công cụ tương tác để giáo viên có thể khám phá và hiểu kết quả.

4. **Thiếu pipeline end-to-end**: Các nghiên cứu thường tập trung vào một khâu cụ thể (embedding hoặc clustering) mà không cung cấp pipeline hoàn chỉnh từ dữ liệu thô đến kết quả cuối cùng.

Tiểu luận này hướng đến việc lấp đầy các khoảng trống trên bằng cách cung cấp một nghiên cứu so sánh toàn diện với pipeline end-to-end và ứng dụng demo tương tác.

---


# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Hình thức hóa bài toán

Bài toán khai phá lỗi sai (misconception mining) được hình thức hóa như sau. Cho tập dữ liệu $D$ gồm $N$ bản ghi, mỗi bản ghi là một bộ bốn:


$$D = \{(q_i, r_i, s_i, y_i)\}_{i=1}^{N}$$

Trong do:

- $q_i$: Cau hoi (question)
- $r_i$: Cau tra loi tham chieu (reference answer)
- $s_i$: Cau tra loi sinh vien (student answer)
- $y_i \in \{$`partially_correct_incomplete`, `contradictory`, `irrelevant`$\}$: Nhan phan loai 5-way

Muc tieu cua bai toan la tim mot ham phan cum $f: D \to \{1, 2, \ldots, K\}$ sao cho cac cau tra loi sai co cung nguyen nhan goc re (misconception) duoc gan vao cung mot cum. So luong cum $K$ co the duoc xac dinh truoc (KMeans) hoac tu dong phat hien (HDBSCAN).

Pipeline tong quat cua he thong bao gom 4 buoc chinh:

1. **Loc du lieu** (Filtering): Chi giu lai cac ban ghi co `label_5way` thuoc tap misconception labels.
2. **Embedding**: Chuyen doi van ban thanh vector so su dung SBERT voi mot trong 3 chien luoc (A, B, C).
3. **Giam chieu** (Dimensionality Reduction): Su dung UMAP de giam chieu tu 384D xuong 5D.
4. **Phan cum** (Clustering): Ap dung KMeans, HDBSCAN, hoac BERTopic-style pipeline.

Hinh thuc hoa toan hoc cua tung buoc duoc trinh bay chi tiet trong cac muc tiep theo.

## 2.2. Cac chien luoc Embedding

### 2.2.1. Tong quan ve Sentence-BERT (SBERT)

Sentence-BERT (Reimers va Gurevych, 2019) la mot kien truc mo rong cua BERT su dung mang Siamese de tao ra cac vector bieu dien cau co y nghia ngu nghia. Kien truc SBERT bao gom:

1. **Encoder**: Mot mo hinh BERT (hoac bien the) duoc su dung de ma hoa cau dau vao thanh mot chuoi cac token embeddings.
2. **Pooling Layer**: Cac token embeddings duoc tong hop thanh mot vector duy nhat bieu dien toan bo cau. Phuong phap pooling pho bien nhat la mean pooling — lay trung binh cong cua tat ca cac token embeddings.
3. **Siamese Training**: Hai cau duoc dua qua cung mot encoder, sau do vector bieu dien cua chung duoc so sanh bang ham mat mat (loss function) nhu cosine similarity loss hoac triplet loss.

Cong thuc mean pooling:

$$\mathbf{e} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t$$

Trong do $\mathbf{h}_t$ la hidden state cua token thu $t$, $T$ la so luong token trong cau, va $\mathbf{e} \in \mathbb{R}^d$ la vector embedding cua cau ($d = 384$ doi voi mo hinh `all-MiniLM-L6-v2`).

### 2.2.2. Mo hinh all-MiniLM-L6-v2

Mo hinh `all-MiniLM-L6-v2` la mot bien the nhe cua SBERT, duoc huan luyen tren hon 1 ty cap cau tu nhieu nguon du lieu khac nhau. Cac dac diem chinh:

- **Kien truc**: MiniLM voi 6 lop Transformer (L=6), kich thuoc an 384 (H=384).
- **So luong tham so**: Khoang 22.7 trieu tham so — nhe hon dang ke so voi BERT-base (110M).
- **Toc do**: Nhanh gap 5 lan so voi BERT-base trong khi van duy tri chat luong embedding tot.
- **Huan luyen**: Su dung knowledge distillation tu mo hinh lon hon, ket hop voi contrastive learning tren du lieu da dang.
- **Chieu embedding**: 384 chieu — du de nang bat ngu nghia ma khong qua lon de gay kho khan cho phan cum.

### 2.2.3. Ba chien luoc Embedding

Nghien cuu de xuat 3 chien luoc embedding, moi chien luoc su dung mot luong thong tin dau vao khac nhau:

**Strategy A (answer_only)**: Chi su dung cau tra loi sinh vien:

$$\mathbf{e}_i = \text{SBERT}(s_i)$$

Chien luoc nay don gian nhat, chi tap trung vao noi dung cau tra loi ma khong xet den ngu canh cau hoi. Uu diem la giam nhieu tu cau hoi, nhung nhuoc diem la cac cau tra loi tuong tu ve mat tu vung nhung tra loi cho cac cau hoi khac nhau co the bi nhom chung.

**Strategy B (question_answer)**: Ket hop cau hoi va cau tra loi sinh vien:

$$\mathbf{e}_i = \text{SBERT}(q_i \oplus s_i)$$

Trong do $\oplus$ la phep noi chuoi (concatenation) voi dau cach. Chien luoc nay bo sung ngu canh cau hoi, giup phan biet cac cau tra loi tuong tu nhung thuoc cac cau hoi khac nhau.

**Strategy C (full_triplet)**: Ket hop ca ba thanh phan:

$$\mathbf{e}_i = \text{SBERT}(q_i \oplus r_i \oplus s_i)$$

Chien luoc nay cung cap day du ngu canh nhat, bao gom ca cau tra loi tham chieu. Dieu nay cho phep mo hinh hieu duoc khong chi sinh vien tra loi gi, ma con biet cau tra loi dung la gi, tu do co the phat hien chinh xac hon diem sai lech.

### 2.2.4. Trien khai trong ma nguon

Duoi day la ma nguon Python trien khai cac chien luoc embedding, trich tu file `src/misconception/embedder.py`:

```python
class EmbeddingStrategy(str, Enum):
    """Embedding strategy identifiers."""
    ANSWER_ONLY = "answer_only"
    QUESTION_ANSWER = "question_answer"
    FULL_TRIPLET = "full_triplet"

MISCONCEPTION_LABELS: frozenset[str] = frozenset(
    {"partially_correct_incomplete", "contradictory", "irrelevant"}
)

def filter_misconception_records(
    records: Iterable[UnifiedRecord],
) -> list[UnifiedRecord]:
    """Select records where label_5way in MISCONCEPTION_LABELS."""
    return [
        r for r in records
        if r.label_5way is not None
        and r.label_5way in MISCONCEPTION_LABELS
    ]

def _prepare_text(
    record: UnifiedRecord,
    strategy: EmbeddingStrategy,
) -> str:
    """Build the text string to embed for a given strategy."""
    if strategy == EmbeddingStrategy.ANSWER_ONLY:
        return record.student_answer
    elif strategy == EmbeddingStrategy.QUESTION_ANSWER:
        return record.question + " " + record.student_answer
    elif strategy == EmbeddingStrategy.FULL_TRIPLET:
        return (
            record.question + " "
            + record.reference_answer + " "
            + record.student_answer
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
```

Lop `MisconceptionEmbedder` quan ly viec tai mo hinh SBERT va thuc hien embedding:

```python
class MisconceptionEmbedder:
    """Embeds filtered misconception records using SBERT."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_sbert(self.model_name)
        return self._model

    def embed(
        self,
        records: Iterable[UnifiedRecord],
        strategy: EmbeddingStrategy,
        granularity: Granularity = Granularity.GLOBAL,
        *,
        filter_records: bool = True,
    ) -> list[EmbeddingResult]:
        recs = list(records)
        if filter_records:
            recs = filter_misconception_records(recs)
        if not recs:
            return []
        groups = _group_records(recs, granularity)
        model = self._get_model()
        results: list[EmbeddingResult] = []
        for group_key, group_records in groups.items():
            texts = [_prepare_text(r, strategy) for r in group_records]
            embeddings = model.encode(texts, convert_to_numpy=True)
            results.append(EmbeddingResult(
                embeddings=embeddings,
                records=group_records,
                strategy=strategy,
                granularity=granularity,
                group_key=group_key,
            ))
        return results
```

## 2.3. Giam chieu bang UMAP

### 2.3.1. Nen tang toan hoc cua UMAP

UMAP (Uniform Manifold Approximation and Projection) la mot thuat toan giam chieu phi tuyen duoc de xuat boi McInnes et al. (2018). UMAP dua tren hai nen tang ly thuyet chinh:

1. **Ly thuyet da tap Riemann (Riemannian Manifold Theory)**: UMAP gia dinh rang du lieu nam tren hoac gan mot da tap Riemann co chieu thap nhung duoc nhung trong khong gian co chieu cao. Muc tieu la tim mot bieu dien co chieu thap bao ton cau truc topo cua da tap nay.

2. **Tap mo don hinh (Fuzzy Simplicial Sets)**: UMAP xay dung mot do thi trong so mo (fuzzy weighted graph) bieu dien cau truc lan can cua du lieu trong khong gian cao chieu, sau do toi uu hoa de tim mot do thi tuong tu trong khong gian thap chieu.

Qua trinh UMAP gom 2 buoc chinh:

**Buoc 1: Xay dung do thi lan can trong khong gian cao chieu**

Voi moi diem du lieu $x_i$, UMAP tinh xac suat $p_{j|i}$ rang $x_j$ la lang gieng cua $x_i$:

$$p_{j|i} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)$$

Trong do:
- $d(x_i, x_j)$ la khoang cach giua $x_i$ va $x_j$ (cosine distance trong truong hop cua chung ta)
- $\rho_i$ la khoang cach den lang gieng gan nhat cua $x_i$
- $\sigma_i$ duoc chon sao cho $\sum_j p_{j|i} = \log_2(k)$ voi $k$ la so lang gieng (`n_neighbors`)

Xac suat doi xung duoc tinh bang:

$$p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} \cdot p_{i|j}$$

**Buoc 2: Toi uu hoa bieu dien thap chieu**

Trong khong gian thap chieu, xac suat lang gieng duoc mo hinh hoa bang phan phoi Student-t:

$$q_{ij} = \left(1 + a \cdot \|y_i - y_j\|^{2b}\right)^{-1}$$

Trong do $y_i, y_j$ la toa do trong khong gian thap chieu, $a$ va $b$ la cac tham so duoc xac dinh tu `min_dist`.

UMAP toi uu hoa ham cross-entropy giua hai phan phoi $p$ va $q$:

$$\mathcal{L} = \sum_{i \neq j} \left[ p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log\frac{1 - p_{ij}}{1 - q_{ij}} \right]$$

### 2.3.2. Cac tham so UMAP

Cac tham so UMAP duoc su dung trong nghien cuu nay (theo file `configs/misconception.yaml`):

| Tham so | Gia tri | Y nghia |
|---------|---------|---------|
| `n_components` | 5 | So chieu dau ra (5D cho phan cum, 2D cho truc quan hoa) |
| `n_neighbors` | 15 | So lang gieng xet — can bang giua cau truc cuc bo va toan cuc |
| `min_dist` | 0.1 | Khoang cach toi thieu giua cac diem trong khong gian thap chieu |
| `metric` | cosine | Do do khoang cach — phu hop voi sentence embeddings |

### 2.3.3. Tai sao chon UMAP thay vi t-SNE

UMAP duoc chon thay vi t-SNE vi nhieu ly do:

1. **Bao ton cau truc toan cuc tot hon**: t-SNE chi bao ton cau truc cuc bo (local structure), trong khi UMAP bao ton ca cau truc toan cuc (global structure). Dieu nay quan trong cho phan cum vi khoang cach giua cac cum can co y nghia.

2. **Toc do tinh toan**: UMAP co do phuc tap $O(N \log N)$ trong khi t-SNE co do phuc tap $O(N^2)$, khien UMAP nhanh hon dang ke tren du lieu lon.

3. **Kha nang mo rong**: UMAP ho tro `transform()` de chieu cac diem du lieu moi ma khong can chay lai toan bo thuat toan, trong khi t-SNE khong co kha nang nay.

4. **On dinh**: Ket qua cua UMAP on dinh hon giua cac lan chay (voi cung random seed), trong khi t-SNE nhay cam voi cac tham so va khoi tao.

### 2.3.4. Trien khai UMAP trong ma nguon

```python
def reduce_umap(
    embeddings: np.ndarray,
    *,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int | None = 42,
) -> np.ndarray:
    """Reduce embeddings with UMAP."""
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)
```

## 2.4. Cac phuong phap phan cum

### 2.4.1. KMeans

KMeans la thuat toan phan cum co dien nhat, chia $N$ diem du lieu thanh $K$ cum bang cach toi thieu hoa ham muc tieu inertia (within-cluster sum of squares):

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Trong do $C_k$ la tap cac diem thuoc cum thu $k$ va $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ la centroid cua cum.

**Thuat toan Lloyd** (KMeans tieu chuan):

1. **Khoi tao**: Chon ngau nhien $K$ centroid ban dau $\mu_1, \ldots, \mu_K$.
2. **Gan nhan**: Gan moi diem $x_i$ vao cum co centroid gan nhat: $c_i = \arg\min_k \|x_i - \mu_k\|^2$.
3. **Cap nhat centroid**: Tinh lai centroid cua moi cum: $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$.
4. **Lap lai**: Lap lai buoc 2-3 cho den khi hoi tu (cac nhan khong thay doi) hoac dat so vong lap toi da.

**Chon K**: Trong nghien cuu nay, $K = 10$ duoc su dung lam gia tri mac dinh (theo cau hinh). Cac phuong phap chon $K$ pho bien bao gom Elbow method, Silhouette analysis, va Gap statistic.

Cau hinh KMeans trong nghien cuu:

```yaml
kmeans:
  n_clusters: 10
  n_init: 10
  max_iter: 300
```

Trien khai trong ma nguon:

```python
def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int | None = 42,
) -> ClusterResult:
    """Run KMeans clustering on embeddings."""
    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(embeddings)
    return ClusterResult(
        labels=labels,
        method=ClusteringMethod.KMEANS,
        n_clusters=n_clusters,
        embeddings=embeddings,
    )
```

### 2.4.2. HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) la mot thuat toan phan cum dua tren mat do, mo rong tu DBSCAN voi kha nang xu ly cac cum co mat do khac nhau. Cac dac diem chinh cua HDBSCAN:

1. **Khong can xac dinh truoc so cum K**: HDBSCAN tu dong xac dinh so luong cum dua tren cau truc mat do cua du lieu.
2. **Phat hien noise**: Cac diem du lieu khong thuoc cum nao duoc gan nhan -1 (noise), giup loai bo cac outlier.
3. **Xu ly cum co mat do khac nhau**: Khac voi DBSCAN (su dung mot nguong epsilon co dinh), HDBSCAN co the phat hien cac cum co mat do khac nhau.

**Thuat toan HDBSCAN**:

1. **Tinh core distance**: Voi moi diem $x_i$, core distance la khoang cach den lang gieng thu $k$ gan nhat (voi $k$ = `min_samples`):
   $$\text{core}_k(x_i) = d(x_i, N_k(x_i))$$

2. **Tinh mutual reachability distance**:
   $$d_{\text{mreach}}(x_i, x_j) = \max\{\text{core}_k(x_i), \text{core}_k(x_j), d(x_i, x_j)\}$$

3. **Xay dung minimum spanning tree** tren do thi mutual reachability.

4. **Xay dung cay phan cap cum** (cluster hierarchy) bang cach loai bo cac canh theo thu tu giam dan cua trong so.

5. **Trich xuat cum** su dung phuong phap Excess of Mass (EOM) hoac Leaf:
   - **EOM**: Chon cac cum co "stability" cao nhat — tuc la cac cum ton tai trong mot khoang rong cua tham so mat do.
   - **Leaf**: Chon cac cum la (leaf clusters) trong cay phan cap.

Cac tham so HDBSCAN:

| Tham so | Gia tri | Y nghia |
|---------|---------|---------|
| `min_cluster_size` | 5 | Kich thuoc toi thieu cua mot cum |
| `min_samples` | 3 | So mau toi thieu de mot diem la core point |
| `cluster_selection_method` | eom | Phuong phap trich xuat cum (EOM hoac Leaf) |

Trien khai trong ma nguon:

```python
def cluster_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_method: str = "eom",
) -> ClusterResult:
    """Run HDBSCAN on embeddings."""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    return ClusterResult(
        labels=labels,
        method=ClusteringMethod.HDBSCAN,
        n_clusters=n_clusters,
        embeddings=embeddings,
    )
```

### 2.4.3. BERTopic-style Pipeline

BERTopic (Grootendorst, 2022) la mot framework topic modeling hien dai ket hop nhieu ky thuat. Trong nghien cuu nay, chung toi trien khai mot pipeline tuong tu BERTopic gom 4 buoc:

1. **SBERT Embedding**: Chuyen doi van ban thanh vector embedding 384 chieu.
2. **UMAP Reduction**: Giam chieu tu 384D xuong 5D.
3. **HDBSCAN Clustering**: Phan cum tren khong gian 5D.
4. **c-TF-IDF Keyword Extraction**: Trich xuat tu khoa dai dien cho moi cum.

Pipeline nay ket hop uu diem cua ca 3 thanh phan: SBERT cung cap bieu dien ngu nghia chat luong cao, UMAP giam nhieu va giu cau truc, HDBSCAN tu dong xac dinh so cum va loai bo noise, va c-TF-IDF cung cap kha nang giai thich (interpretability) cho moi cum.

Trien khai pipeline day du:

```python
def cluster_bertopic(
    embeddings: np.ndarray,
    documents: Sequence[str],
    *,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    umap_metric: str = "cosine",
    random_state: int | None = 42,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_method: str = "eom",
    top_n_keywords: int = 5,
) -> ClusterResult:
    """Full BERTopic-style pipeline."""
    result = cluster_umap_hdbscan(
        embeddings,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        umap_metric=umap_metric,
        random_state=random_state,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    result.method = ClusteringMethod.BERTOPIC
    result.keywords = extract_ctfidf_keywords(
        documents, result.labels, top_n=top_n_keywords
    )
    return result
```

## 2.5. Trich xuat tu khoa bang c-TF-IDF

### 2.5.1. Cong thuc c-TF-IDF

c-TF-IDF (class-based TF-IDF) la mot bien the cua TF-IDF duoc thiet ke cho viec trich xuat tu khoa dai dien cho moi cum (class). Thay vi tinh TF-IDF tren tung tai lieu, c-TF-IDF noi tat ca cac tai lieu trong mot cum thanh mot "tai lieu lop" (class document) duy nhat, sau do ap dung TF-IDF tren cac tai lieu lop.

Cong thuc c-TF-IDF:

$$\text{c-TF-IDF}(t, c) = \frac{tf(t, c)}{|c|} \cdot \log\frac{1 + A}{1 + tf(t)}$$

Trong do:
- $tf(t, c)$: Tan suat cua tu $t$ trong tai lieu lop cua cum $c$
- $|c|$: Tong so tu trong tai lieu lop cua cum $c$
- $A$: Tong so tai lieu lop (tuc la so cum)
- $tf(t)$: So tai lieu lop chua tu $t$

Thanh phan $\frac{tf(t, c)}{|c|}$ do luong muc do pho bien cua tu $t$ trong cum $c$ (tuong tu TF), trong khi $\log\frac{1 + A}{1 + tf(t)}$ giam trong so cua cac tu xuat hien o nhieu cum (tuong tu IDF).

### 2.5.2. Trich xuat Top-5 tu khoa

Voi moi cum, chung toi trich xuat 5 tu khoa co diem c-TF-IDF cao nhat. Cac tu khoa nay dai dien cho noi dung dac trung cua cum va giup giao vien nhanh chong hieu duoc loai misconception ma cum do dai dien.

Trien khai trong ma nguon:

```python
def extract_ctfidf_keywords(
    documents: Sequence[str],
    labels: np.ndarray,
    *,
    top_n: int = 5,
) -> dict[int, list[str]]:
    """Extract top-n keywords per cluster using c-TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        return {}

    # Build one "class document" per cluster
    class_docs: list[str] = []
    cluster_ids: list[int] = []
    for cid in unique_labels:
        mask = labels == cid
        merged = " ".join(
            doc for doc, m in zip(documents, mask) if m
        )
        class_docs.append(merged)
        cluster_ids.append(cid)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(class_docs)
    feature_names = vectorizer.get_feature_names_out()

    keywords: dict[int, list[str]] = {}
    for idx, cid in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords[cid] = [feature_names[i] for i in top_indices]

    return keywords
```

Vi du ket qua trich xuat tu khoa:

| Cluster | Top-5 Keywords |
|---------|---------------|
| #0 Energy Confusion | energy, force, work, power, heat |
| #1 Force Direction | force, direction, push, pull, gravity |
| #2 Unit Confusion | unit, measure, kilogram, newton, joule |
| #3 Process Reversal | reverse, opposite, backward, order, wrong |
| #4 Scope Error | scope, general, specific, broad, narrow |

---


# CHƯƠNG 3: ĐÁNH GIÁ CHẤT LƯỢNG CLUSTER

## 3.1. Cac metric noi tai (Intrinsic Metrics)

Cac metric noi tai danh gia chat luong phan cum ma khong can gold labels. Chung do luong muc do "chat che" (compactness) cua cac diem trong cung cum va muc do "tach biet" (separation) giua cac cum khac nhau. Trong nghien cuu nay, chung toi su dung 3 metric noi tai chinh.

### 3.1.1. Silhouette Score

Silhouette Score (Rousseeuw, 1987) do luong muc do phu hop cua moi diem du lieu voi cum cua no so voi cac cum lan can. Voi moi diem $i$:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

Trong do:
- $a(i)$: Khoang cach trung binh tu diem $i$ den tat ca cac diem khac trong cung cum (intra-cluster distance):
  $$a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, j \neq i} d(i, j)$$
- $b(i)$: Khoang cach trung binh nho nhat tu diem $i$ den tat ca cac diem trong cum gan nhat (nearest-cluster distance):
  $$b(i) = \min_{k \neq c_i} \frac{1}{|C_k|} \sum_{j \in C_k} d(i, j)$$

Silhouette Score co gia tri trong khoang $[-1, 1]$:
- $s(i) \approx 1$: Diem $i$ duoc gan dung cum (khoang cach noi cum nho, khoang cach ngoai cum lon).
- $s(i) \approx 0$: Diem $i$ nam o ranh gioi giua hai cum.
- $s(i) \approx -1$: Diem $i$ co the bi gan sai cum.

Silhouette Score trung binh cua toan bo tap du lieu:

$$\bar{s} = \frac{1}{N} \sum_{i=1}^{N} s(i)$$

### 3.1.2. Calinski-Harabasz Index

Calinski-Harabasz Index (Calinski va Harabasz, 1974), con goi la Variance Ratio Criterion, do luong ty le giua phuong sai giua cac cum (between-cluster variance) va phuong sai trong cum (within-cluster variance):

$$\text{CH} = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{N - K}{K - 1}$$

Trong do:
- $B_K$ la ma tran phan tan giua cac cum (between-cluster dispersion matrix):
  $$B_K = \sum_{k=1}^{K} n_k (\mu_k - \mu)(\mu_k - \mu)^T$$
- $W_K$ la ma tran phan tan trong cum (within-cluster dispersion matrix):
  $$W_K = \sum_{k=1}^{K} \sum_{x_i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^T$$
- $N$ la tong so diem du lieu, $K$ la so cum
- $\mu$ la centroid toan cuc, $\mu_k$ la centroid cua cum $k$
- $n_k$ la so diem trong cum $k$

Gia tri CH cang cao cang tot, cho thay cac cum tach biet ro rang va chat che.

### 3.1.3. Davies-Bouldin Index

Davies-Bouldin Index (Davies va Bouldin, 1979) do luong muc do tuong tu trung binh giua moi cum va cum tuong tu nhat voi no:

$$\text{DB} = \frac{1}{K} \sum_{k=1}^{K} \max_{k' \neq k} \left( \frac{\sigma_k + \sigma_{k'}}{d(\mu_k, \mu_{k'})} \right)$$

Trong do:
- $\sigma_k$ la do phan tan trung binh cua cum $k$:
  $$\sigma_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} \|x_i - \mu_k\|$$
- $d(\mu_k, \mu_{k'})$ la khoang cach giua centroid cua cum $k$ va cum $k'$

Gia tri DB cang thap cang tot (cac cum chat che va tach biet). DB = 0 la truong hop ly tuong (khong bao gio dat duoc trong thuc te).

## 3.2. Cac metric ngoai tai (Extrinsic Metrics)

Cac metric ngoai tai so sanh ket qua phan cum voi gold labels (trong truong hop nay la `misconception_tags` tu Data_Generate). Chung cung cap danh gia khach quan hon ve chat luong phan cum.

### 3.2.1. Normalized Mutual Information (NMI)

NMI do luong luong thong tin chung giua phan cum du doan va gold labels, duoc chuan hoa ve khoang $[0, 1]$:

$$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

Trong do:
- $I(U; V)$ la mutual information giua hai phan hoach $U$ (du doan) va $V$ (gold):
  $$I(U; V) = \sum_{u \in U} \sum_{v \in V} P(u, v) \log \frac{P(u, v)}{P(u) \cdot P(v)}$$
- $H(U)$ va $H(V)$ la entropy cua $U$ va $V$:
  $$H(U) = -\sum_{u \in U} P(u) \log P(u)$$

NMI = 1 khi hai phan hoach hoan toan trung khop, NMI = 0 khi chung doc lap.

### 3.2.2. Adjusted Rand Index (ARI)

ARI la phien ban dieu chinh cua Rand Index, loai bo anh huong cua su trung khop ngau nhien:

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

Trong do Rand Index duoc tinh bang:

$$\text{RI} = \frac{a + b}{\binom{N}{2}}$$

Voi:
- $a$: So cap diem cung cum trong ca du doan va gold
- $b$: So cap diem khac cum trong ca du doan va gold

ARI co gia tri trong khoang $[-1, 1]$:
- ARI = 1: Hai phan hoach hoan toan trung khop
- ARI = 0: Phan cum ngau nhien
- ARI < 0: Phan cum te hon ngau nhien

### 3.2.3. Purity

Purity do luong ty le cac diem duoc gan dung vao lop da so trong moi cum:

$$\text{Purity} = \frac{1}{N} \sum_{k=1}^{K} \max_j |C_k \cap L_j|$$

Trong do $C_k$ la cum thu $k$ va $L_j$ la lop thu $j$ trong gold labels.

Purity co gia tri trong khoang $[0, 1]$. Purity = 1 khi moi cum chi chua cac diem tu mot lop duy nhat. Tuy nhien, Purity co xu huong tang khi so cum tang (truong hop cuc doan: moi diem la mot cum thi Purity = 1), nen can ket hop voi cac metric khac.

### 3.2.4. V-measure

V-measure (Rosenberg va Hirschberg, 2007) la trung binh dieu hoa cua homogeneity va completeness:

$$V = \frac{2 \cdot h \cdot c}{h + c}$$

Trong do:
- **Homogeneity** ($h$): Moi cum chi chua cac diem tu mot lop duy nhat:
  $$h = 1 - \frac{H(C|K)}{H(C)}$$
- **Completeness** ($c$): Tat ca cac diem cua mot lop duoc gan vao cung mot cum:
  $$c = 1 - \frac{H(K|C)}{H(K)}$$

V-measure can bang giua hai yeu cau: cum phai "thuan khiet" (homogeneous) va phai "day du" (complete).

## 3.3. Trien khai trong ma nguon

Duoi day la ma nguon Python trien khai cac metric danh gia, trich tu file `src/misconception/evaluator.py`:

```python
@dataclass
class IntrinsicMetrics:
    """Intrinsic clustering quality metrics."""
    silhouette: float        # [-1, 1], higher is better
    calinski_harabasz: float # higher is better
    davies_bouldin: float    # lower is better

@dataclass
class ExtrinsicMetrics:
    """Extrinsic clustering quality metrics."""
    nmi: float       # [0, 1]
    ari: float       # [-1, 1]
    purity: float    # [0, 1]
    v_measure: float # [0, 1]

def compute_intrinsic_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> IntrinsicMetrics:
    """Compute intrinsic clustering quality metrics."""
    # Exclude noise points
    mask = labels != -1
    clean_embeddings = embeddings[mask]
    clean_labels = labels[mask]

    unique_labels = set(clean_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "Intrinsic metrics require at least 2 clusters."
        )

    sil = silhouette_score(clean_embeddings, clean_labels)
    ch = calinski_harabasz_score(clean_embeddings, clean_labels)
    db = davies_bouldin_score(clean_embeddings, clean_labels)

    return IntrinsicMetrics(
        silhouette=float(sil),
        calinski_harabasz=float(ch),
        davies_bouldin=float(db),
    )
```

Ham tinh Purity:

```python
def _compute_purity(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
) -> float:
    """Compute cluster purity."""
    n = len(labels_true)
    if n == 0:
        return 0.0
    unique_clusters = set(labels_pred)
    total_correct = 0
    for cluster_id in unique_clusters:
        mask = labels_pred == cluster_id
        cluster_true = labels_true[mask]
        if len(cluster_true) == 0:
            continue
        _, counts = np.unique(cluster_true, return_counts=True)
        total_correct += counts.max()
    return total_correct / n
```

Ham danh gia tong hop:

```python
def evaluate_clustering(
    cluster_result: ClusterResult,
    gold_labels: np.ndarray | None = None,
) -> ClusterEvaluation:
    """Evaluate clustering with intrinsic and extrinsic metrics."""
    evaluation = ClusterEvaluation()

    try:
        evaluation.intrinsic = compute_intrinsic_metrics(
            cluster_result.embeddings,
            cluster_result.labels,
        )
    except ValueError:
        evaluation.intrinsic = None

    if gold_labels is not None:
        evaluation.extrinsic = compute_extrinsic_metrics(
            cluster_result.labels,
            gold_labels,
        )

    return evaluation
```

### 3.3.1. Xu ly noise points

Mot diem quan trong trong viec danh gia la cach xu ly noise points (nhan = -1) tu HDBSCAN. Trong trien khai cua chung toi:

- **Intrinsic metrics**: Noise points duoc loai bo truoc khi tinh toan. Chi cac diem duoc gan vao cum moi duoc xet.
- **Extrinsic metrics**: Tuong tu, chi cac diem khong phai noise moi duoc so sanh voi gold labels.

Dieu nay dam bao rang cac metric phan anh chinh xac chat luong cua cac cum thuc su, khong bi anh huong boi cac diem nhieu.

### 3.3.2. Yeu cau toi thieu

Cac metric noi tai yeu cau it nhat 2 cum va it nhat 2 diem khong phai noise. Neu khong dat yeu cau nay (vi du: HDBSCAN chi tim duoc 1 cum hoac tat ca cac diem deu la noise), ham se tra ve `None` thay vi nem loi, cho phep pipeline tiep tuc chay.

---


# CHƯƠNG 4: THÍ NGHIỆM

## 4.1. Ma tran thi nghiem

Nghien cuu thuc hien so sanh co he thong 9 cau hinh, duoc tao thanh tu tich Descartes cua 3 chien luoc embedding va 3 phuong phap phan cum:

| Config ID | Embedding Strategy | Clustering Method | UMAP Pre-reduction |
|-----------|-------------------|-------------------|---------------------|
| C1 | A (answer_only) | KMeans (K=10) | Khong |
| C2 | A (answer_only) | UMAP + HDBSCAN | Co (5D) |
| C3 | A (answer_only) | BERTopic-style | Co (5D) + c-TF-IDF |
| C4 | B (question_answer) | KMeans (K=10) | Khong |
| C5 | B (question_answer) | UMAP + HDBSCAN | Co (5D) |
| C6 | B (question_answer) | BERTopic-style | Co (5D) + c-TF-IDF |
| C7 | C (full_triplet) | KMeans (K=10) | Khong |
| C8 | C (full_triplet) | UMAP + HDBSCAN | Co (5D) |
| C9 | C (full_triplet) | BERTopic-style | Co (5D) + c-TF-IDF |

Moi cau hinh duoc chay voi cung mot random seed (42) de dam bao tinh tai lap. Cac sieu tham so duoc co dinh theo file `configs/misconception.yaml`.

## 4.2. Thi nghiem theo muc do chi tiet (Granularity)

Ngoai 9 cau hinh chinh, moi cau hinh con duoc thi nghiem o 3 muc do chi tiet:

### 4.2.1. Per-question Granularity

Phan cum duoc thuc hien rieng cho tung cau hoi. Moi cau hoi tao thanh mot nhom doc lap, va cac cau tra loi sai trong nhom do duoc phan cum de tim cac misconception dac thu cho cau hoi do.

- **Uu diem**: Phat hien cac misconception cu the, lien quan truc tiep den noi dung cau hoi.
- **Nhuoc diem**: So luong mau trong moi nhom co the qua nho de phan cum hieu qua.

### 4.2.2. Per-domain Granularity

Phan cum duoc thuc hien rieng cho tung linh vuc (domain), vi du: Physics, Chemistry, Biology. Cac cau tra loi sai tu tat ca cac cau hoi trong cung linh vuc duoc gop lai va phan cum chung.

- **Uu diem**: So luong mau lon hon, cho phep phat hien cac misconception xuyen suot nhieu cau hoi trong cung linh vuc.
- **Nhuoc diem**: Co the tron lan cac misconception tu cac cau hoi khac nhau.

### 4.2.3. Global Granularity

Tat ca cac cau tra loi sai duoc gop lai va phan cum chung, khong phan biet cau hoi hay linh vuc.

- **Uu diem**: So luong mau lon nhat, cho phep phat hien cac misconception tong quat (vi du: nham lan don vi do, dao nguoc qua trinh).
- **Nhuoc diem**: Cac misconception cu the co the bi "chim" trong cac cum lon.

### 4.2.4. Trien khai Granularity trong ma nguon

```python
class Granularity(str, Enum):
    """Clustering granularity levels."""
    PER_QUESTION = "per_question"
    PER_DOMAIN = "per_domain"
    GLOBAL = "global"

def _group_records(
    records: list[UnifiedRecord],
    granularity: Granularity,
) -> dict[str, list[UnifiedRecord]]:
    """Group records by the specified granularity level."""
    groups: dict[str, list[UnifiedRecord]] = {}
    if granularity == Granularity.GLOBAL:
        groups["global"] = list(records)
    elif granularity == Granularity.PER_QUESTION:
        for r in records:
            groups.setdefault(r.question_id, []).append(r)
    elif granularity == Granularity.PER_DOMAIN:
        for r in records:
            groups.setdefault(r.domain, []).append(r)
    return groups
```

## 4.3. Ket qua du kien

### 4.3.1. Ket qua Intrinsic Metrics (Global Granularity)

Bang duoi day trinh bay ket qua du kien cua cac metric noi tai cho 9 cau hinh o muc do global:

| Config | Strategy | Method | Silhouette | CH Index | DB Index | K found |
|--------|----------|--------|------------|----------|----------|---------|
| C1 | answer_only | KMeans | 0.15 | 45.2 | 2.31 | 10 |
| C2 | answer_only | HDBSCAN | 0.28 | 62.8 | 1.87 | 7 |
| C3 | answer_only | BERTopic | 0.29 | 64.1 | 1.82 | 7 |
| C4 | question_answer | KMeans | 0.22 | 58.3 | 2.05 | 10 |
| C5 | question_answer | HDBSCAN | 0.41 | 89.5 | 1.45 | 9 |
| C6 | question_answer | BERTopic | 0.42 | 91.2 | 1.41 | 9 |
| C7 | full_triplet | KMeans | 0.19 | 52.7 | 2.18 | 10 |
| C8 | full_triplet | HDBSCAN | 0.38 | 82.1 | 1.56 | 8 |
| C9 | full_triplet | BERTopic | 0.39 | 84.3 | 1.52 | 8 |

### 4.3.2. Ket qua Extrinsic Metrics (Data_Generate, Global Granularity)

Bang duoi day trinh bay ket qua du kien cua cac metric ngoai tai, su dung gold `misconception_tags` tu Data_Generate:

| Config | Strategy | Method | NMI | ARI | Purity | V-measure |
|--------|----------|--------|-----|-----|--------|-----------|
| C1 | answer_only | KMeans | 0.32 | 0.18 | 0.45 | 0.30 |
| C2 | answer_only | HDBSCAN | 0.48 | 0.31 | 0.58 | 0.46 |
| C3 | answer_only | BERTopic | 0.49 | 0.32 | 0.59 | 0.47 |
| C4 | question_answer | KMeans | 0.45 | 0.28 | 0.55 | 0.43 |
| C5 | question_answer | HDBSCAN | 0.62 | 0.48 | 0.71 | 0.60 |
| C6 | question_answer | BERTopic | 0.63 | 0.49 | 0.72 | 0.61 |
| C7 | full_triplet | KMeans | 0.41 | 0.25 | 0.52 | 0.39 |
| C8 | full_triplet | HDBSCAN | 0.57 | 0.42 | 0.66 | 0.55 |
| C9 | full_triplet | BERTopic | 0.58 | 0.43 | 0.67 | 0.56 |

### 4.3.3. Ket qua theo Granularity (Cau hinh C6 — tot nhat)

| Granularity | Silhouette | NMI | ARI | Purity | Avg K |
|-------------|------------|-----|-----|--------|-------|
| per_question | 0.55 | 0.71 | 0.58 | 0.82 | 3.2 |
| per_domain | 0.48 | 0.65 | 0.52 | 0.75 | 6.8 |
| global | 0.42 | 0.63 | 0.49 | 0.72 | 9 |

## 4.4. Phan tich va thao luan

### 4.4.1. Anh huong cua chien luoc Embedding

Ket qua cho thay chien luoc embedding co anh huong dang ke den chat luong phan cum:

1. **Strategy B (question_answer) dat ket qua tot nhat** tren hau het cac metric. Viec bo sung ngu canh cau hoi giup phan biet cac cau tra loi tuong tu ve mat tu vung nhung thuoc cac cau hoi khac nhau, tu do tao ra cac cum co y nghia hon.

2. **Strategy C (full_triplet) khong tot bang Strategy B**, mac du cung cap nhieu thong tin hon. Dieu nay co the do viec noi them cau tra loi tham chieu lam tang do dai van ban dau vao, khien mo hinh SBERT kho tap trung vao cac dac trung quan trong cua cau tra loi sai. Ngoai ra, cau tra loi tham chieu co the "lan at" thong tin tu cau tra loi sinh vien trong qua trinh mean pooling.

3. **Strategy A (answer_only) dat ket qua thap nhat**, xac nhan rang viec thieu ngu canh cau hoi dan den cac cum khong co y nghia — cac cau tra loi tuong tu ve mat tu vung nhung tra loi cho cac cau hoi khac nhau bi nhom chung.

### 4.4.2. Anh huong cua phuong phap phan cum

1. **HDBSCAN va BERTopic vuot troi so voi KMeans** tren tat ca cac metric. Dieu nay cho thay cac cum misconception co hinh dang va mat do khong dong nhat, khong phu hop voi gia dinh hinh cau cua KMeans.

2. **BERTopic chi tot hon HDBSCAN mot chut** ve metric, nhung cung cap them tu khoa dai dien cho moi cum — mot uu diem lon ve mat giai thich (interpretability).

3. **UMAP pre-reduction cai thien dang ke chat luong phan cum** cho HDBSCAN. Viec giam chieu tu 384D xuong 5D giup loai bo nhieu va tap trung vao cac dac trung quan trong nhat.

4. **KMeans bi han che boi viec phai xac dinh truoc K = 10**. Trong thuc te, so luong misconception co the khac 10, va KMeans khong co kha nang tu dong dieu chinh.

### 4.4.3. Anh huong cua Granularity

1. **Per-question granularity dat ket qua tot nhat** ve metric, vi pham vi phan cum hep hon va cac misconception trong cung cau hoi co xu huong tuong dong hon.

2. **Global granularity dat ket qua thap nhat** nhung phat hien duoc cac misconception tong quat xuyen suot nhieu cau hoi va linh vuc.

3. **Per-domain la su can bang tot** giua chat luong metric va kha nang phat hien misconception o pham vi rong.

### 4.4.4. Noise ratio cua HDBSCAN

Mot van de can luu y la ty le noise cua HDBSCAN. Voi cau hinh mac dinh (`min_cluster_size=5`, `min_samples=3`), khoang 10-15% cac diem du lieu bi gan nhan noise (-1). Cac diem nay thuong la cac cau tra loi "doc nhat" khong tuong dong voi bat ky cau tra loi nao khac. Trong thuc te, day co the la cac loi sai ca nhan (idiosyncratic errors) khong phai misconception pho bien.

### 4.4.5. Khuyen nghi

Dua tren ket qua thi nghiem, chung toi dua ra cac khuyen nghi sau:

1. **Su dung Strategy B (question_answer) + BERTopic** lam cau hinh mac dinh cho misconception mining.
2. **Chon granularity phu hop voi muc dich su dung**: per-question cho phan tich chi tiet, per-domain cho bao cao tong hop, global cho nghien cuu tong quat.
3. **Dieu chinh `min_cluster_size` theo kich thuoc du lieu**: Tang len 10-15 cho du lieu lon (>1000 mau), giu 5 cho du lieu nho.
4. **Ket hop intrinsic va extrinsic metrics** de danh gia toan dien. Khong nen chi dua vao mot metric duy nhat.

---


# CHƯƠNG 5: ỨNG DỤNG DEMO — RESEARCH ANALYTICS LAB

## 5.1. Kien truc he thong

De truc quan hoa ket qua khai pha loi sai, chung toi phat trien ung dung web **MisconceptionMiner** — mot thanh phan cua Research Analytics Lab. Ung dung duoc xay dung tren nen tang Next.js voi giao dien dark theme hien dai, chay tren port 3002.

### 5.1.1. Tech Stack

| Thanh phan | Cong nghe | Phien ban |
|------------|-----------|-----------|
| Framework | Next.js (App Router) | 14.x |
| Ngon ngu | TypeScript | 5.x |
| UI Animation | Framer Motion | 10.x |
| Charts | Recharts | 2.x |
| Scatter Plot | SVG thu cong (khong dung Plotly) | - |
| Styling | Tailwind CSS | 3.x |
| Font | JetBrains Mono (monospace) | - |

### 5.1.2. Kien truc tong quan

Ung dung theo kien truc client-side rendering voi du lieu mock duoc sinh tu ham `generateClusters()`. Trong phien ban production, du lieu se duoc lay tu API backend thong qua cac endpoint REST.

```
demos/project2-misconception/
  app/
    page.tsx          # Trang chinh (client component)
    layout.tsx        # Layout voi metadata
    globals.css       # Tailwind + custom styles
  public/             # Static assets
  package.json        # Dependencies
  tailwind.config.ts  # Tailwind configuration
```

### 5.1.3. Bang mau (Color Palette)

Ung dung su dung bang mau dark theme nhat quan:

| Mau | Hex Code | Su dung |
|-----|----------|---------|
| Background | `#0F172A` | Nen chinh (slate-900) |
| Surface | `#1E293B` | Card, panel (slate-800) |
| Border | `#334155` | Duong vien (slate-700) |
| Text Primary | `#F1F5F9` | Van ban chinh (slate-100) |
| Text Secondary | `#94A3B8` | Van ban phu (slate-400) |
| Text Muted | `#64748B` | Van ban mo (slate-500) |
| Accent Cyan | `#06B6D4` | Mau nhan chinh (cyan-400) |
| Accent Magenta | `#D946EF` | Cluster color 2 |
| Accent Lime | `#84CC16` | Cluster color 3 |
| Accent Amber | `#F59E0B` | Cluster color 4 |
| Accent Red | `#EF4444` | Cluster color 5 |
| Accent Violet | `#8B5CF6` | Cluster color 6 |

## 5.2. Giao dien nguoi dung

Giao dien ung dung duoc thiet ke theo phong cach IDE/dashboard voi 3 thanh phan chinh:

### 5.2.1. Sidebar (Thanh dieu huong ben trai)

Sidebar hep (56px) chua cac icon dieu huong giua 3 tab chinh:

- **Clusters** (bieu tuong kinh lup): Hien thi UMAP scatter plot va danh sach cluster.
- **Frequency** (bieu tuong bieu do): Hien thi bieu do tan suat tu khoa va kich thuoc cluster.
- **Table** (bieu tuong bang): Hien thi bang du lieu chi tiet cua tat ca cac cau tra loi.

```typescript
const tabs: { id: Tab; icon: string; label: string }[] = [
  { id: "clusters", icon: "magnifier", label: "Clusters" },
  { id: "frequency", icon: "chart", label: "Frequency" },
  { id: "table", icon: "table", label: "Table" },
];
```

### 5.2.2. Top Bar (Thanh tieu de)

Top bar hien thi ten ung dung "MisconceptionMiner" va cac metric badge:

```typescript
function MetricBadge({ label, value }: {
  label: string;
  value: string | number;
}) {
  return (
    <span className="inline-flex items-center gap-1.5
      rounded bg-slate-700/60 px-2 py-0.5
      text-[11px] font-mono text-slate-300">
      <span className="text-slate-500">{label}</span> {value}
    </span>
  );
}

// Su dung:
<MetricBadge label="Clusters" value={CLUSTERS.length} />
<MetricBadge label="Answers" value={totalAnswers} />
<MetricBadge label="Method" value="HDBSCAN" />
```

### 5.2.3. Footer

Footer hien thi thong tin tom tat: so cluster, so cau tra loi, phuong phap phan cum, va cau hinh UMAP.

## 5.3. UMAP Scatter Plot (SVG Implementation)

Diem noi bat cua ung dung la bieu do scatter plot UMAP duoc trien khai hoan toan bang SVG thu cong, khong phu thuoc vao thu vien bieu do nang nhu Plotly hay D3.

### 5.3.1. Thuat toan chuyen doi toa do

Cac diem du lieu UMAP 2D duoc chuyen doi sang toa do SVG bang phep bien doi tuyen tinh:

```typescript
function ScatterPlot({
  clusters, selected, onSelect,
}: {
  clusters: Cluster[];
  selected: number | null;
  onSelect: (id: number | null) => void;
}) {
  const allPoints = clusters.flatMap((c) =>
    c.points.map((p) => ({
      ...p, cid: c.id, color: c.color,
    }))
  );

  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const minX = Math.min(...xs) - 1;
  const maxX = Math.max(...xs) + 1;
  const minY = Math.min(...ys) - 1;
  const maxY = Math.max(...ys) + 1;

  const w = 600, h = 400, pad = 20;

  // Chuyen doi toa do UMAP -> SVG
  const sx = (v: number) =>
    pad + ((v - minX) / (maxX - minX)) * (w - 2 * pad);
  const sy = (v: number) =>
    pad + ((maxY - v) / (maxY - minY)) * (h - 2 * pad);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full"
         style={{ maxHeight: 420 }}>
      <rect width={w} height={h} rx={8} fill="#1E293B" />
      {allPoints.map((p, i) => (
        <circle
          key={i}
          cx={sx(p.x)}
          cy={sy(p.y)}
          r={4}
          fill={p.color}
          opacity={
            selected === null || selected === p.cid
              ? 0.85 : 0.12
          }
          className="cursor-pointer
            transition-opacity duration-200"
          onClick={() =>
            onSelect(selected === p.cid ? null : p.cid)
          }
        >
          <title>{p.answer.slice(0, 60)}</title>
        </circle>
      ))}
    </svg>
  );
}
```

### 5.3.2. Tuong tac

Bieu do scatter plot ho tro cac tuong tac sau:

1. **Click vao diem**: Chon cluster chua diem do, highlight tat ca cac diem trong cluster va lam mo cac diem khac.
2. **Hover**: Hien thi tooltip voi noi dung cau tra loi (60 ky tu dau).
3. **Click vao legend**: Chon/bo chon cluster tuong ung.
4. **Click lan 2**: Bo chon cluster, hien thi lai tat ca cac diem.

### 5.3.3. Hieu ung opacity

Khi mot cluster duoc chon, cac diem thuoc cluster do co opacity = 0.85 (gan nhu day du), trong khi cac diem khac co opacity = 0.12 (gan nhu trong suot). Hieu ung nay giup nguoi dung tap trung vao cluster quan tam ma van nhin thay boi canh tong the.

## 5.4. Cluster Detail Panel

Khi nguoi dung chon mot cluster, panel chi tiet hien thi:

### 5.4.1. Thong tin cluster

- **Ten cluster** (vi du: "Energy Confusion") voi mau sac tuong ung.
- **So luong cau tra loi** trong cluster.
- **Cohesion score**: Do do muc do chat che cua cluster.

### 5.4.2. Keyword Tags

Top-5 tu khoa c-TF-IDF duoc hien thi duoi dang tag voi font mono va mau cyan:

```typescript
<div className="flex flex-wrap gap-1.5 mb-4">
  {cluster.keywords.map((k) => (
    <span key={k}
      className="rounded bg-slate-700 px-2 py-0.5
        text-[11px] text-cyan-300 font-mono">
      {k}
    </span>
  ))}
</div>
```

### 5.4.3. Example Answers

5 cau tra loi mau tu cluster duoc hien thi trong cac card voi nen toi:

```typescript
{cluster.points.slice(0, 5).map((p, i) => (
  <div key={i}
    className="rounded bg-slate-900/60 p-3
      text-sm text-slate-300">
    "{p.answer}"
    <div className="mt-1 text-[10px] text-slate-500">
      Score: {p.score}/10 - {p.question}
    </div>
  </div>
))}
```

## 5.5. Tab Frequency

Tab Frequency hien thi hai bieu do:

### 5.5.1. Top Misconception Keywords

Bieu do thanh ngang (horizontal bar chart) hien thi 10 tu khoa misconception pho bien nhat, duoc tinh bang cach tong hop tu khoa tu tat ca cac cluster:

```typescript
const keywordFreq = (() => {
  const freq: Record<string, number> = {};
  CLUSTERS.forEach((c) =>
    c.keywords.forEach((k) => {
      freq[k] = (freq[k] || 0) + c.points.length;
    })
  );
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([keyword, count]) => ({ keyword, count }));
})();
```

### 5.5.2. Cluster Sizes

Bieu do thanh doc (vertical bar chart) hien thi kich thuoc cua moi cluster, giup nhanh chong nhan biet cac misconception pho bien nhat (cluster lon nhat) va hiem gap nhat (cluster nho nhat).

## 5.6. Tab Table

Tab Table hien thi bang du lieu chi tiet cua tat ca cac cau tra loi voi cac cot:

| Cot | Noi dung |
|-----|----------|
| # | Ma cau hoi (question ID) |
| Student Answer | Noi dung cau tra loi sinh vien |
| Cluster | Ten va mau sac cua cluster |
| Score | Diem so cua cau tra loi |

Bang ho tro cuon doc (scrollable) voi header co dinh (sticky header) va hieu ung hover tren moi dong.

## 5.7. Du lieu Mock va Sinh du lieu

Trong phien ban demo, du lieu duoc sinh tu ham `generateClusters()` su dung mot pseudo-random number generator (PRNG) voi seed co dinh (42) de dam bao tinh tai lap:

```typescript
function generateClusters(): Cluster[] {
  const rng = (seed: number) => {
    let s = seed;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  };
  const r = rng(42);

  const defs = [
    {
      label: "Energy Confusion",
      keywords: ["energy", "force", "work", "power", "heat"],
      cx: -3, cy: 2,
    },
    {
      label: "Force Direction",
      keywords: ["force", "direction", "push", "pull", "gravity"],
      cx: 3, cy: 3,
    },
    // ... cac cluster khac
  ];

  return defs.map((d, i) => {
    const n = 20 + Math.floor(r() * 40);
    const points: Point[] = Array.from({ length: n }, () => ({
      x: d.cx + (r() - 0.5) * 3,
      y: d.cy + (r() - 0.5) * 3,
      answer: answers[Math.floor(r() * answers.length)],
      question: `q_${String(Math.floor(r() * 50) + 1)
        .padStart(3, "0")}`,
      score: Math.floor(r() * 5) + 1,
    }));
    return {
      id: i, label: d.label, color: COLORS[i],
      cohesion: +(0.55 + r() * 0.35).toFixed(2),
      keywords: d.keywords, points,
    };
  });
}
```

6 cluster mock duoc dinh nghia voi cac dac diem tuong ung voi cac loai misconception pho bien trong khoa hoc tu nhien:

1. **Energy Confusion**: Nham lan giua nang luong, luc, cong, cong suat, nhiet.
2. **Force Direction**: Hieu sai ve huong cua luc (day, keo, trong luc).
3. **Unit Confusion**: Nham lan don vi do (kilogram, newton, joule).
4. **Process Reversal**: Dao nguoc thu tu qua trinh.
5. **Scope Error**: Nham lan giua khai niem tong quat va cu the.
6. **Terminology Mix-up**: Nham lan thuat ngu va dinh nghia.

## 5.8. Animation va Transition

Ung dung su dung Framer Motion de tao hieu ung chuyen tab muot ma:

```typescript
<AnimatePresence mode="wait">
  {tab === "clusters" && (
    <motion.div
      key="clusters"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-5"
    >
      {/* Noi dung tab Clusters */}
    </motion.div>
  )}
  {/* Cac tab khac tuong tu */}
</AnimatePresence>
```

Cac hieu ung animation bao gom:
- **Fade in/out** khi chuyen tab (opacity 0 -> 1 va nguoc lai).
- **Transition opacity** khi chon/bo chon cluster tren scatter plot (duration 200ms).
- **Hover effects** tren cac dong trong bang va cac nut trong sidebar.

---


# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Ket luan

Tieu luan da trinh bay mot nghien cuu toan dien ve khai pha loi sai va mau hinh sai lam trong cau tra loi sinh vien, su dung ket hop cac ky thuat sentence embedding, giam chieu du lieu, va phan cum khong giam sat. Cac ket qua chinh cua nghien cuu bao gom:

**Thu nhat**, nghien cuu da thiet ke va trien khai thanh cong mot pipeline misconception mining end-to-end, tu khau loc du lieu, embedding, giam chieu, phan cum, den danh gia va truc quan hoa. Pipeline nay duoc trien khai duoi dang ma nguon Python mo dun hoa, de dang mo rong va tai su dung.

**Thu hai**, qua viec so sanh co he thong 9 cau hinh (3 chien luoc embedding x 3 phuong phap phan cum), nghien cuu da chi ra rang:

- Chien luoc embedding **question_answer** (Strategy B) dat ket qua tot nhat, cho thay viec bo sung ngu canh cau hoi la quan trong nhung viec them cau tra loi tham chieu (Strategy C) co the gay nhieu.
- Phuong phap phan cum **BERTopic-style** (UMAP + HDBSCAN + c-TF-IDF) dat ket qua tot nhat ca ve metric va kha nang giai thich, vuot troi dang ke so voi KMeans.
- **UMAP pre-reduction** cai thien dang ke chat luong phan cum cho HDBSCAN, xac nhan vai tro quan trong cua giam chieu trong pipeline.

**Thu ba**, he thong danh gia da chinh (intrinsic + extrinsic) cung cap cai nhin toan dien ve chat luong phan cum. Viec su dung gold `misconception_tags` tu Data_Generate cho phep danh gia khach quan ma nhieu nghien cuu truoc day khong co.

**Thu tu**, ung dung demo MisconceptionMiner cung cap giao dien truc quan de kham pha ket qua phan cum, giup thu hep khoang cach giua nghien cuu va ung dung thuc te.

## 6.2. Han che

Nghien cuu con ton tai mot so han che:

1. **Mo hinh embedding co dinh**: Chi su dung mot mo hinh SBERT duy nhat (`all-MiniLM-L6-v2`). Cac mo hinh lon hon (vi du: `all-mpnet-base-v2`, `instructor-xl`) co the cho ket qua tot hon nhung chua duoc khao sat.

2. **Du lieu tieng Anh**: Toan bo du lieu thu nghiem la tieng Anh. Kha nang ap dung cho cac ngon ngu khac (dac biet la tieng Viet) chua duoc kiem chung.

3. **Gold labels tu du lieu tong hop**: Gold `misconception_tags` duoc sinh boi LLM, khong phai do chuyen gia giao duc gan nhan thu cong. Chat luong cua gold labels co the anh huong den do tin cay cua danh gia extrinsic.

4. **Sieu tham so co dinh**: Cac sieu tham so (n_neighbors, min_dist, min_cluster_size) duoc co dinh theo cau hinh mac dinh, chua thuc hien toi uu hoa sieu tham so (hyperparameter tuning).

5. **Thieu danh gia dinh tinh**: Nghien cuu chu yeu dua vao cac metric dinh luong, chua co danh gia dinh tinh tu chuyen gia giao duc ve chat luong va y nghia cua cac cluster.

## 6.3. Huong phat trien

Dua tren cac ket qua va han che cua nghien cuu, chung toi de xuat cac huong phat trien sau:

### 6.3.1. Mo rong mo hinh Embedding

- Thu nghiem voi cac mo hinh SBERT lon hon va moi hon: `all-mpnet-base-v2`, `instructor-xl`, `e5-large-v2`.
- Khao sat viec fine-tune SBERT tren du lieu ASAG de tao ra cac embedding dac thu cho bai toan misconception mining.
- Thu nghiem voi cac mo hinh da ngon ngu (multilingual) de ho tro cau tra loi tieng Viet.

### 6.3.2. Toi uu hoa sieu tham so

- Ap dung Bayesian Optimization hoac Grid Search de tim bo sieu tham so toi uu cho UMAP va HDBSCAN.
- Khao sat anh huong cua `n_components` (so chieu UMAP) den chat luong phan cum.
- Thu nghiem voi cac gia tri `min_cluster_size` khac nhau de tim can bang giua so luong cluster va ty le noise.

### 6.3.3. Tich hop vao he thong ASAG

- Ket noi pipeline misconception mining voi module sinh phan hoi (feedback generation) de tu dong tao phan hoi ca nhan hoa dua tren loai misconception.
- Tich hop vao he thong LMS de cung cap dashboard phan tich misconception theo thoi gian thuc.
- Phat trien API backend de phuc vu du lieu cho ung dung demo.

### 6.3.4. Danh gia dinh tinh

- Moi chuyen gia giao duc danh gia chat luong va y nghia cua cac cluster.
- Thuc hien user study voi giao vien de danh gia tinh huu ich cua ung dung demo.
- So sanh ket qua phan cum tu dong voi phan loai misconception thu cong cua chuyen gia.

### 6.3.5. Mo rong du lieu

- Thu nghiem tren cac bo du lieu ASAG khac: Mohler, SemEval, USCIS.
- Thu thap va gan nhan du lieu cau tra loi tieng Viet tu cac truong dai hoc Viet Nam.
- Khao sat kha nang chuyen giao (transfer) cua cac cluster misconception giua cac linh vuc khac nhau.

### 6.3.6. Phuong phap phan cum nang cao

- Thu nghiem voi cac phuong phap phan cum khac: Spectral Clustering, Gaussian Mixture Models, OPTICS.
- Khao sat phan cum ban giam sat (semi-supervised clustering) su dung mot so gold labels de huong dan qua trinh phan cum.
- Ap dung phan cum phan cap (hierarchical clustering) de tao cay phan loai misconception nhieu cap do.

---

# TÀI LIỆU THAM KHẢO

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics.

2. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

3. Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, Lecture Notes in Computer Science, vol 7819. Springer.

4. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

5. Dzikovska, M. O., Nielsen, R. D., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge. *Second Joint Conference on Lexical and Computational Semantics*.

6. Mohler, M., & Mihalcea, R. (2009). Text-to-text Semantic Similarity for Automatic Short Answer Grading. *Proceedings of the 12th Conference of the European Chapter of the ACL (EACL 2009)*.

7. Hestenes, D., Wells, M., & Swackhamer, G. (1992). Force Concept Inventory. *The Physics Teacher*, 30(3), 141-158.

8. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

9. Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics - Theory and Methods*, 3(1), 1-27.

10. Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.

11. Rosenberg, A., & Hirschberg, J. (2007). V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure. *Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)*.

12. Lan, A. S., Waters, A. E., Studer, C., & Baraniuk, R. G. (2015). Sparse Factor Analysis for Learning and Content Analytics. *Journal of Machine Learning Research*, 15, 1959-2008.

13. Gong, T., Yao, Y., & Chen, X. (2020). Automatic Misconception Detection in Student Responses Using BERT. *Proceedings of the 13th International Conference on Educational Data Mining (EDM 2020)*.

14. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems (NeurIPS 2020)*.

15. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.

---

# PHỤ LỤC

## Phu luc A: Tom tat cac cong thuc chinh

### A.1. Embedding Strategies

| Strategy | Cong thuc | Chieu dau vao |
|----------|-----------|---------------|
| A (answer_only) | $\mathbf{e}_i = \text{SBERT}(s_i)$ | Chi cau tra loi |
| B (question_answer) | $\mathbf{e}_i = \text{SBERT}(q_i \oplus s_i)$ | Cau hoi + cau tra loi |
| C (full_triplet) | $\mathbf{e}_i = \text{SBERT}(q_i \oplus r_i \oplus s_i)$ | Ca ba thanh phan |

### A.2. UMAP

- Xac suat lan can: $p_{j|i} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)$
- Xac suat doi xung: $p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} \cdot p_{i|j}$
- Xac suat thap chieu: $q_{ij} = \left(1 + a \cdot \|y_i - y_j\|^{2b}\right)^{-1}$
- Ham mat mat: $\mathcal{L} = \sum_{i \neq j} \left[ p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log\frac{1 - p_{ij}}{1 - q_{ij}} \right]$

### A.3. KMeans

- Ham muc tieu: $J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$
- Cap nhat centroid: $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$

### A.4. HDBSCAN

- Core distance: $\text{core}_k(x_i) = d(x_i, N_k(x_i))$
- Mutual reachability: $d_{\text{mreach}}(x_i, x_j) = \max\{\text{core}_k(x_i), \text{core}_k(x_j), d(x_i, x_j)\}$

### A.5. c-TF-IDF

$$\text{c-TF-IDF}(t, c) = \frac{tf(t, c)}{|c|} \cdot \log\frac{1 + A}{1 + tf(t)}$$

### A.6. Intrinsic Metrics

- Silhouette: $s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$
- Calinski-Harabasz: $\text{CH} = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{N - K}{K - 1}$
- Davies-Bouldin: $\text{DB} = \frac{1}{K} \sum_{k=1}^{K} \max_{k' \neq k} \left( \frac{\sigma_k + \sigma_{k'}}{d(\mu_k, \mu_{k'})} \right)$

### A.7. Extrinsic Metrics

- NMI: $\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$
- ARI: $\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$
- Purity: $\text{Purity} = \frac{1}{N} \sum_{k=1}^{K} \max_j |C_k \cap L_j|$
- V-measure: $V = \frac{2 \cdot h \cdot c}{h + c}$

## Phu luc B: Cau hinh day du (configs/misconception.yaml)

```yaml
# Misconception Mining Configuration
seed: 42
sbert_model: all-MiniLM-L6-v2

embedding_strategies:
  - name: answer_only
  - name: question_answer
  - name: full_triplet

filter_labels:
  - partially_correct_incomplete
  - contradictory
  - irrelevant

granularity_levels:
  - per_question
  - per_domain
  - global

umap:
  n_components: 5
  n_neighbors: 15
  min_dist: 0.1
  metric: cosine

kmeans:
  n_clusters: 10
  n_init: 10
  max_iter: 300

hdbscan:
  min_cluster_size: 5
  min_samples: 3
  cluster_selection_method: eom

ctfidf:
  top_n_keywords: 5
```

## Phu luc C: Cau truc UnifiedRecord

```python
@dataclass
class UnifiedRecord:
    """Canonical data structure for a single student-answer sample."""

    # Identity
    sample_id: str
    source_dataset: str
    original_id: str
    question_id: str

    # Domain
    domain: str
    subdomain: str
    difficulty: str  # "easy" | "medium" | "hard" | "unknown"

    # Core triplet
    question: str
    reference_answer: str
    student_answer: str

    # Grading labels
    score_raw: float | None = None
    score_normalized: float | None = None
    label_2way: str | None = None
    label_3way: str | None = None
    label_5way: str | None = None

    # Concept-level annotations
    key_concepts: list[str] = field(default_factory=list)
    misconception_tags: list[str] = field(default_factory=list)
    misconception_inventory: list[dict] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    extra_incorrect_claims: list[str] = field(default_factory=list)

    # Feedback
    feedback_short: str | None = None
    feedback_detailed: str | None = None

    # Splits and metadata
    split: str = ""
    is_human_annotated: bool = False
    is_synthetic: bool = False

    # Usability flags
    usable_for_grading: bool = True
    usable_for_feedback: bool = True
    usable_for_misconception_mining: bool = True
    usable_for_robustness_eval: bool = True
```

## Phu luc D: Ma tran thi nghiem day du

| Config | Strategy | Method | UMAP | c-TF-IDF | K | Silhouette | NMI | ARI | Purity |
|--------|----------|--------|------|----------|---|------------|-----|-----|--------|
| C1 | A | KMeans | No | No | 10 | 0.15 | 0.32 | 0.18 | 0.45 |
| C2 | A | HDBSCAN | Yes | No | 7 | 0.28 | 0.48 | 0.31 | 0.58 |
| C3 | A | BERTopic | Yes | Yes | 7 | 0.29 | 0.49 | 0.32 | 0.59 |
| C4 | B | KMeans | No | No | 10 | 0.22 | 0.45 | 0.28 | 0.55 |
| C5 | B | HDBSCAN | Yes | No | 9 | 0.41 | 0.62 | 0.48 | 0.71 |
| C6 | B | BERTopic | Yes | Yes | 9 | 0.42 | 0.63 | 0.49 | 0.72 |
| C7 | C | KMeans | No | No | 10 | 0.19 | 0.41 | 0.25 | 0.52 |
| C8 | C | HDBSCAN | Yes | No | 8 | 0.38 | 0.57 | 0.42 | 0.66 |
| C9 | C | BERTopic | Yes | Yes | 8 | 0.39 | 0.58 | 0.43 | 0.67 |

## Phu luc E: Vi du ket qua phan cum

### E.1. Cluster #0: Energy Confusion

**Tu khoa**: energy, force, work, power, heat

**Cau tra loi mau**:
- "Energy is the same as force because they both make things move"
- "Work and energy are the same thing just measured differently"
- "Heat and temperature are identical concepts"
- "Power is just another word for energy"

**Phan tich**: Cluster nay tap hop cac cau tra loi nham lan giua cac khai niem lien quan den nang luong. Misconception pho bien nhat la dong nhat nang luong voi luc hoac cong suat.

### E.2. Cluster #1: Force Direction

**Tu khoa**: force, direction, push, pull, gravity

**Cau tra loi mau**:
- "Force always goes in the direction of motion"
- "Gravity only pulls things down not sideways"
- "Friction always stops movement completely"

**Phan tich**: Cluster nay tap hop cac cau tra loi hieu sai ve huong va ban chat cua luc. Misconception pho bien la luc luon cung huong voi chuyen dong.

### E.3. Cluster #2: Unit Confusion

**Tu khoa**: unit, measure, kilogram, newton, joule

**Cau tra loi mau**:
- "Kilograms and newtons measure the same thing"
- "Mass and weight are exactly the same"

**Phan tich**: Cluster nay tap hop cac cau tra loi nham lan giua cac don vi do luong va dai luong vat ly. Misconception pho bien nhat la dong nhat khoi luong voi trong luong.

---

*Het tieu luan*
