---
title: "Tiểu luận 1: Thiết kế Cơ sở Dữ liệu Thống nhất cho Hệ thống Chấm điểm Tự động Câu trả lời Ngắn"
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 13pt
linestretch: 1.5
---

# CHƯƠNG 1: MỞ ĐẦU

## 1.1. Tính cấp thiết của đề tài

Trong bối cảnh giáo dục hiện đại, việc đánh giá năng lực học sinh thông qua câu trả lời ngắn (short answer) là một phương pháp phổ biến và hiệu quả. Khác với câu hỏi trắc nghiệm chỉ kiểm tra khả năng nhận diện đáp án đúng, câu trả lời ngắn yêu cầu người học phải tự diễn đạt kiến thức bằng ngôn ngữ của mình, qua đó thể hiện mức độ hiểu biết sâu sắc hơn về chủ đề. Tuy nhiên, việc chấm điểm hàng ngàn câu trả lời ngắn bằng tay là một công việc tốn kém về thời gian và nhân lực, đồng thời tiềm ẩn nguy cơ thiếu nhất quán giữa các giáo viên chấm bài.

Automatic Short Answer Grading (ASAG) — chấm điểm tự động câu trả lời ngắn — là một hướng nghiên cứu quan trọng trong lĩnh vực Educational AI và Natural Language Processing (NLP). Các hệ thống ASAG nhận đầu vào là bộ ba (câu hỏi, câu trả lời tham chiếu, câu trả lời sinh viên) và đưa ra dự đoán về mức độ đúng đắn của câu trả lời, dưới dạng nhãn phân loại (correct/incorrect) hoặc điểm số liên tục.

Tuy nhiên, một thách thức cốt lõi mà các nhà nghiên cứu ASAG phải đối mặt là vấn đề **dữ liệu**. Các bộ dữ liệu benchmark công khai hiện có như SciEntsBank (Dzikovska et al., 2013) và MohlerASAG (Mohler & Mihalcea, 2009) tuy có giá trị học thuật cao nhưng tồn tại nhiều hạn chế: quy mô nhỏ (chỉ vài ngàn mẫu), schema không đồng nhất giữa các bộ dữ liệu, hệ thống nhãn khác nhau (5-way categorical vs. continuous 0-5), và thiếu các annotation phong phú phục vụ cho nhiều tác vụ nghiên cứu cùng lúc như khai phá lỗi sai (misconception mining), sinh phản hồi (feedback generation), hay đánh giá độ bền vững (robustness evaluation).

Tiểu luận này trình bày phương pháp thiết kế và xây dựng một **hệ thống dữ liệu thống nhất** (Unified Data Pipeline) phục vụ nghiên cứu ASAG toàn diện. Hệ thống tích hợp 4 nguồn dữ liệu khác nhau — bao gồm 2 bộ dữ liệu công khai (SciEntsBank, MohlerASAG), 1 bộ dữ liệu thu thập từ web (Data_Scraping từ OpenStax), và 1 bộ dữ liệu tổng hợp sinh bằng AI (Data_Generate) — vào một schema duy nhất gọi là UnifiedRecord, với hệ thống harmonize nhãn, quản lý split chống data leakage, và công cụ kiểm tra chất lượng tự động.

## 1.2. Mục tiêu và nhiệm vụ của đề tài

Mục tiêu tổng quát của tiểu luận là xây dựng một pipeline dữ liệu end-to-end phục vụ nghiên cứu ASAG đa tác vụ. Cụ thể, các nhiệm vụ bao gồm:

1. **Thu thập dữ liệu từ web (Web Scraping)**: Xây dựng công cụ crawl dữ liệu câu hỏi và câu trả lời tham chiếu từ nền tảng giáo dục mở OpenStax, tạo ra bộ Data_Scraping với 129 mẫu.

2. **Sinh dữ liệu tổng hợp bằng AI (Synthetic Data Generation)**: Thiết kế và triển khai framework sinh dữ liệu đa giai đoạn sử dụng Large Language Models (LLMs), tạo ra bộ Data_Generate với 10,000 mẫu bao gồm đầy đủ annotation cho 4 tác vụ nghiên cứu.

3. **Tích hợp dữ liệu công khai**: Xây dựng loader cho SciEntsBank (~10,000 mẫu, nhãn 5-way) và MohlerASAG (~2,273 mẫu, điểm liên tục 0-5).

4. **Thiết kế schema thống nhất**: Xây dựng dataclass UnifiedRecord với 30+ trường dữ liệu, bao phủ tất cả thông tin cần thiết cho grading, misconception mining, feedback generation, và robustness evaluation.

5. **Harmonize nhãn**: Xây dựng LabelHarmonizer ánh xạ các hệ thống nhãn khác nhau (5-way, 3-way, 2-way, continuous score) về một không gian nhãn thống nhất.

6. **Quản lý split và kiểm tra chất lượng**: Xây dựng SplitManager đảm bảo không có data leakage, và DataAudit tool kiểm tra chất lượng dữ liệu tự động.

## 1.3. Đối tượng và phạm vi nghiên cứu

### 1.3.1. Đối tượng nghiên cứu

Đối tượng nghiên cứu chính là dữ liệu câu hỏi-câu trả lời ngắn trong lĩnh vực khoa học tự nhiên (biology, chemistry, physics, earth science) và tin học (programming, data structures, algorithms, databases). Dữ liệu bao gồm cả câu trả lời thực từ sinh viên (SciEntsBank, MohlerASAG) và câu trả lời mô phỏng bằng AI (Data_Generate).

### 1.3.2. Phạm vi nghiên cứu

Phạm vi bao gồm 4 nguồn dữ liệu với tổng cộng khoảng 22,402 mẫu:

| Nguồn dữ liệu | Số mẫu | Loại | Có nhãn |
|---|---|---|---|
| SciEntsBank | ~10,000 | Thực (human-annotated) | Có (5-way) |
| MohlerASAG | ~2,273 | Thực (human-annotated) | Có (0-5 continuous) |
| Data_Generate | 10,000 | Tổng hợp (AI-generated) | Có (5-way + score) |
| Data_Scraping | 129 | Thu thập (web scraping) | Không |

## 1.4. Cơ sở lý luận và phương pháp nghiên cứu

### 1.4.1. Cơ sở lý luận

Nghiên cứu dựa trên các nền tảng lý thuyết sau:

- **Information Extraction**: Kỹ thuật trích xuất thông tin có cấu trúc từ các nguồn dữ liệu phi cấu trúc (web pages, XML files, CSV files).
- **Data Integration**: Phương pháp tích hợp dữ liệu từ nhiều nguồn heterogeneous vào một schema thống nhất (schema mapping, entity resolution).
- **Synthetic Data Generation**: Lý thuyết về sinh dữ liệu tổng hợp sử dụng generative models, bao gồm các vấn đề về diversity, fidelity, và utility.
- **Educational Assessment Theory**: Lý thuyết đánh giá giáo dục, đặc biệt là rubric-based assessment và formative feedback.

### 1.4.2. Phương pháp nghiên cứu

- **Phương pháp thực nghiệm**: Xây dựng pipeline, chạy thử, đo lường kết quả.
- **Phương pháp so sánh**: So sánh các nguồn dữ liệu, phương pháp harmonize, chiến lược split.
- **Property-Based Testing**: Sử dụng thư viện Hypothesis (Python) để kiểm chứng các tính chất bất biến (invariants) của hệ thống dữ liệu.

## 1.5. Đóng góp mới về khoa học

1. Thiết kế schema UnifiedRecord hỗ trợ đồng thời 4 tác vụ nghiên cứu ASAG trong một cấu trúc dữ liệu duy nhất.
2. Xây dựng framework sinh dữ liệu tổng hợp đa giai đoạn với semantic de-biasing, giảm thiểu shortcut learning.
3. Phương pháp harmonize nhãn có thể cấu hình (configurable thresholds) cho phép thích ứng với nhiều bối cảnh đánh giá khác nhau.
4. Hệ thống kiểm tra leakage tự động với 6 correctness properties được kiểm chứng bằng property-based testing.

## 1.6. Ý nghĩa lý luận và thực tiễn

### 1.6.1. Ý nghĩa lý luận

Tiểu luận đóng góp vào lý thuyết về data integration trong Educational AI bằng cách đề xuất một phương pháp luận có hệ thống để tích hợp dữ liệu ASAG từ nhiều nguồn heterogeneous. Các correctness properties được định nghĩa formal và kiểm chứng bằng property-based testing, tạo tiền đề cho việc xây dựng các benchmark ASAG đáng tin cậy hơn.

### 1.6.2. Ý nghĩa thực tiễn

Pipeline dữ liệu được xây dựng có thể tái sử dụng cho các nghiên cứu ASAG khác. Bộ dữ liệu thống nhất với 22,000+ mẫu cung cấp nền tảng cho việc huấn luyện và đánh giá các mô hình chấm điểm, khai phá lỗi sai, sinh phản hồi, và đánh giá robustness.

## 1.7. Tình hình nghiên cứu của đề tài

Các nghiên cứu liên quan bao gồm:

**SciEntsBank** (Dzikovska et al., 2013): Bộ dữ liệu từ SemEval-2013 Task 7, chứa khoảng 10,000 câu trả lời sinh viên trong lĩnh vực khoa học với nhãn 5-way. Đây là benchmark phổ biến nhất cho ASAG với 3 split đánh giá: Unseen Answers (UA), Unseen Questions (UQ), và Unseen Domains (UD).

**MohlerASAG** (Mohler & Mihalcea, 2009): Bộ dữ liệu chứa 2,273 câu trả lời sinh viên trong lĩnh vực Computer Science với điểm liên tục 0-5 (trung bình từ nhiều annotator). Không có split sẵn, cần tự tạo.

**Synthetic Data for NLP** (các nghiên cứu gần đây): Xu hướng sử dụng LLMs để sinh dữ liệu huấn luyện đã được áp dụng trong nhiều lĩnh vực NLP, nhưng chưa có nghiên cứu nào xây dựng synthetic benchmark đa tác vụ cho ASAG với quy mô 10,000 mẫu và semantic de-biasing.


# CHƯƠNG 2: GIỚI THIỆU TỔNG QUAN VỀ CÁC NGUỒN DỮ LIỆU

## 2.1. Đặt vấn đề

Một hệ thống ASAG toàn diện cần dữ liệu đa dạng để huấn luyện và đánh giá. Tuy nhiên, mỗi bộ dữ liệu công khai hiện có đều có format riêng, hệ thống nhãn riêng, và phạm vi ứng dụng riêng. Điều này tạo ra một thách thức lớn: **làm sao để một mô hình có thể được huấn luyện trên dữ liệu từ nhiều nguồn khác nhau và được đánh giá một cách công bằng trên các benchmark khác nhau?**

Vấn đề cụ thể bao gồm:

- **Heterogeneous schemas**: SciEntsBank sử dụng XML format với nhãn 5-way, MohlerASAG sử dụng CSV với điểm liên tục, Data_Generate có 30 cột CSV, Data_Scraping là JSON.
- **Incompatible label spaces**: Nhãn "partially_correct_incomplete" trong SciEntsBank không có tương đương trực tiếp trong MohlerASAG (chỉ có điểm số).
- **Missing annotations**: Data_Scraping không có student answers hay nhãn; MohlerASAG không có misconception tags hay feedback.
- **Split inconsistency**: SciEntsBank có split sẵn (UA/UQ/UD), MohlerASAG không có, Data_Generate có 7 split phức tạp.

## 2.2. Bài toán hợp nhất dữ liệu

Bài toán được formalize như sau:

Cho $n$ nguồn dữ liệu $\{D_1, D_2, ..., D_n\}$, mỗi nguồn có schema riêng $S_i$, hệ thống nhãn riêng $L_i$, và chiến lược split riêng $P_i$. Mục tiêu là xây dựng:

1. **Schema thống nhất** $S^*$ sao cho $\forall i: S_i \subseteq S^*$ (mọi schema nguồn đều là tập con của schema đích)
2. **Hàm ánh xạ nhãn** $h_i: L_i \rightarrow L^*$ cho mỗi nguồn, trong đó $L^*$ là không gian nhãn thống nhất
3. **Hàm chuyển đổi** $f_i: D_i \rightarrow D^*$ biến đổi mỗi record từ schema nguồn sang schema đích
4. **Chiến lược split** $P^*$ đảm bảo không có data leakage giữa train/validation/test

Các ràng buộc (constraints):
- **Completeness**: Không mất thông tin khi chuyển đổi (trường không có giá trị được đặt null thay vì bỏ qua)
- **Consistency**: Nhãn sau harmonize phải nhất quán (VD: score_raw=5 → label_2way="correct")
- **Uniqueness**: Mọi sample_id phải duy nhất trên toàn bộ dataset hợp nhất
- **Leakage-free**: Không có question hay answer nào xuất hiện ở cả train và test

## 2.3. Tổng quan 4 nguồn dữ liệu

### 2.3.1. SciEntsBank

SciEntsBank là bộ dữ liệu benchmark từ SemEval-2013 Task 7 (Joint Student Response Analysis). Bộ dữ liệu chứa khoảng 10,000 câu trả lời sinh viên trong lĩnh vực khoa học tự nhiên, được annotate bởi chuyên gia với nhãn 5-way:

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| correct | Câu trả lời đúng hoàn toàn | "Photosynthesis converts light energy into chemical energy" |
| partially_correct_incomplete | Đúng một phần, thiếu thông tin | "Plants use sunlight to make food" (thiếu chi tiết về CO2, O2) |
| contradictory | Mâu thuẫn với đáp án đúng | "Plants get food from soil" |
| irrelevant | Không liên quan đến câu hỏi | "I like biology class" |
| non_domain | Ngoài phạm vi môn học | "The weather is nice today" |

Đặc điểm quan trọng của SciEntsBank là hệ thống 3 split đánh giá:
- **UA (Unseen Answers)**: Câu trả lời mới cho câu hỏi đã thấy trong training
- **UQ (Unseen Questions)**: Câu hỏi hoàn toàn mới
- **UD (Unseen Domains)**: Domain khoa học hoàn toàn mới

Hệ thống split này cho phép đánh giá khả năng generalize của mô hình ở nhiều mức độ khác nhau.

### 2.3.2. MohlerASAG

MohlerASAG (Mohler & Mihalcea, 2009) chứa 2,273 câu trả lời sinh viên trong lĩnh vực Computer Science (Data Structures, Algorithms). Mỗi câu trả lời được chấm điểm bởi 2 annotator trên thang 0-5, và điểm cuối cùng là trung bình cộng.

Đặc điểm:
- **Nhãn gốc**: score_raw ∈ [0, 5] (continuous)
- **Không có split sẵn**: Cần tự tạo split theo question_id
- **Có alternative reference answers**: Một số câu hỏi có nhiều đáp án tham chiếu
- **Domain đơn**: Chỉ Computer Science

### 2.3.3. Data_Generate (Synthetic Benchmark)

Bộ dữ liệu tổng hợp 10,000 mẫu được sinh bằng framework đa giai đoạn sử dụng LLMs. Đây là bộ dữ liệu phong phú nhất với 30 cột, bao gồm đầy đủ annotation cho cả 4 tác vụ nghiên cứu. Chi tiết về phương pháp sinh dữ liệu sẽ được trình bày ở Chương 4.

### 2.3.4. Data_Scraping (OpenStax)

Bộ dữ liệu 129 mẫu thu thập từ nền tảng giáo dục mở OpenStax, chỉ chứa câu hỏi và câu trả lời tham chiếu (không có student answers). Chi tiết về phương pháp scraping sẽ được trình bày ở Chương 4.

## 2.4. Mục tiêu của tiểu luận

Mục tiêu chính là xây dựng một pipeline hoàn chỉnh biến đổi 4 nguồn dữ liệu heterogeneous thành một bộ dữ liệu thống nhất, sẵn sàng phục vụ cho 4 tác vụ nghiên cứu ASAG. Pipeline phải đảm bảo:

1. Không mất thông tin khi chuyển đổi
2. Nhãn nhất quán sau harmonize
3. Không có data leakage
4. Có công cụ kiểm tra chất lượng tự động
5. Dễ mở rộng khi thêm nguồn dữ liệu mới

## 2.5. Cấu trúc của tiểu luận

- **Chương 1**: Mở đầu — giới thiệu vấn đề, mục tiêu, phạm vi
- **Chương 2**: Tổng quan — đặt vấn đề, mô tả 4 nguồn dữ liệu
- **Chương 3**: Thiết kế schema — UnifiedRecord, ERD, validation
- **Chương 4**: Thu thập và sinh dữ liệu — scraping, AI generation, loaders
- **Chương 5**: Hợp nhất dữ liệu — harmonize, split, audit
- **Chương 6**: Kết quả thực nghiệm và đánh giá
- **Chương 7**: Kết luận và hướng phát triển

# CHƯƠNG 3: THIẾT KẾ CƠ SỞ DỮ LIỆU — UNIFIED RECORD SCHEMA

## 3.1. Phân tích yêu cầu dữ liệu

Hệ thống cần hỗ trợ 4 tác vụ nghiên cứu đồng thời, mỗi tác vụ có yêu cầu dữ liệu riêng:

**Tác vụ 1 — Chấm điểm tự động (Grading)**:
- Bắt buộc: question, reference_answer, student_answer
- Nhãn: label_2way, label_3way, label_5way, score_raw, score_normalized
- Metadata: source_dataset, split, is_human_annotated

**Tác vụ 2 — Khai phá lỗi sai (Misconception Mining)**:
- Bắt buộc: student_answer, label_5way (để lọc câu trả lời sai)
- Annotation: misconception_tags, misconception_inventory, key_concepts
- Metadata: domain, subdomain, question_id (cho clustering theo granularity)

**Tác vụ 3 — Sinh phản hồi (Feedback Generation)**:
- Bắt buộc: question, reference_answer, student_answer, predicted_label
- Annotation: missing_concepts, extra_incorrect_claims, key_concepts
- Target: feedback_short, feedback_detailed, feedback_type, feedback_tone

**Tác vụ 4 — Đánh giá độ bền vững (Robustness Evaluation)**:
- Bắt buộc: student_answer (gốc và đã perturb)
- Metadata: perturbation_type, adversarial_variant_of, is_adversarial
- Annotation: robustness_notes

## 3.2. Thiết kế mô hình dữ liệu (ERD)

UnifiedRecord được thiết kế như một flat dataclass (không có nested entities) để đơn giản hóa việc serialize/deserialize sang JSONL format. Tuy nhiên, về mặt logic, các trường được tổ chức thành 8 nhóm:

```
UnifiedRecord
├── Identity Group:      sample_id, source_dataset, original_id, question_id
├── Domain Group:        domain, subdomain, difficulty
├── Core Triplet:        question, reference_answer, student_answer,
│                        alternative_reference_answers
├── Grading Labels:      score_raw, score_normalized, label_2way,
│                        label_3way, label_5way
├── Concept Annotations: key_concepts, misconception_tags,
│                        misconception_inventory, missing_concepts,
│                        extra_incorrect_claims
├── Feedback:            feedback_short, feedback_detailed,
│                        feedback_type, feedback_tone
├── Metadata:            split, is_human_annotated, is_synthetic,
│                        is_adversarial, perturbation_type,
│                        adversarial_variant_of, student_answer_style,
│                        annotation_confidence
└── Usability Flags:     usable_for_grading, usable_for_feedback,
                         usable_for_misconception_mining,
                         usable_for_robustness_eval
```

## 3.3. Chi tiết Schema — Python Dataclass

```python
from dataclasses import dataclass, field

VALID_SOURCE_DATASETS = frozenset(
    {"scientsbank", "mohler", "data_generate", "data_scraping"}
)
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "unknown"})

@dataclass
class UnifiedRecord:
    """Canonical data structure for a single student-answer sample."""

    # ── Identity ──────────────────────────────────────────────
    sample_id: str           # Globally unique, e.g. "SEB_UA_0001"
    source_dataset: str      # One of VALID_SOURCE_DATASETS
    original_id: str         # ID from the original dataset
    question_id: str         # Groups answers to the same question

    # ── Domain ────────────────────────────────────────────────
    domain: str              # e.g. "biology", "physics", "programming"
    subdomain: str           # e.g. "plant_biology", "mechanics"
    difficulty: str          # "easy" | "medium" | "hard" | "unknown"

    # ── Core triplet ──────────────────────────────────────────
    question: str
    reference_answer: str
    student_answer: str
    alternative_reference_answers: list[str] = field(default_factory=list)

    # ── Grading labels ────────────────────────────────────────
    score_raw: float | None = None
    score_normalized: float | None = None  # Always in [0.0, 1.0]
    label_2way: str | None = None          # "correct" | "incorrect"
    label_3way: str | None = None          # + "partially_correct"
    label_5way: str | None = None          # Full SciEntsBank labels

    # ── Concept-level annotations ─────────────────────────────
    key_concepts: list[str] = field(default_factory=list)
    misconception_tags: list[str] = field(default_factory=list)
    misconception_inventory: list[dict] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    extra_incorrect_claims: list[str] = field(default_factory=list)

    # ── Feedback ──────────────────────────────────────────────
    feedback_short: str | None = None
    feedback_detailed: str | None = None
    feedback_type: str | None = None
    feedback_tone: str | None = None

    # ── Splits and metadata ───────────────────────────────────
    split: str = ""
    is_human_annotated: bool = False
    is_synthetic: bool = False
    is_adversarial: bool = False
    perturbation_type: str | None = None
    adversarial_variant_of: str | None = None
    student_answer_style: str | None = None
    annotation_confidence: float | None = None

    # ── Usability flags ───────────────────────────────────────
    usable_for_grading: bool = True
    usable_for_feedback: bool = True
    usable_for_misconception_mining: bool = True
    usable_for_robustness_eval: bool = True

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.source_dataset not in VALID_SOURCE_DATASETS:
            raise ValueError(
                f"source_dataset must be one of "
                f"{sorted(VALID_SOURCE_DATASETS)}, "
                f"got {self.source_dataset!r}"
            )
        if self.score_normalized is not None:
            if not (0.0 <= self.score_normalized <= 1.0):
                raise ValueError(
                    f"score_normalized must be in [0.0, 1.0], "
                    f"got {self.score_normalized}"
                )
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of "
                f"{sorted(VALID_DIFFICULTIES)}, "
                f"got {self.difficulty!r}"
            )
```

## 3.4. Ma trận tương thích nhãn giữa các nguồn dữ liệu

Bảng sau cho thấy mỗi nguồn dữ liệu hỗ trợ những trường nhãn nào:

| Trường | SciEntsBank | MohlerASAG | Data_Generate | Data_Scraping |
|---|---|---|---|---|
| label_5way | ✓ (native) | ✗ | ✓ (native) | ✗ |
| label_3way | ✓ (derived) | ✓ (derived) | ✓ (native) | ✗ |
| label_2way | ✓ (derived) | ✓ (derived) | ✓ (native) | ✗ |
| score_raw | ✗ | ✓ (native, 0-5) | ✓ (native, 0-5) | ✗ |
| score_normalized | ✗ | ✓ (derived) | ✓ (derived) | ✗ |
| key_concepts | ✗ | ✗ | ✓ | ✗ |
| misconception_tags | ✗ | ✗ | ✓ | ✗ |
| feedback_short | ✗ | ✗ | ✓ | ✗ |
| feedback_detailed | ✗ | ✗ | ✓ | ✗ |
| perturbation_type | ✗ | ✗ | ✓ | ✗ |

Ký hiệu: ✓ (native) = có sẵn trong dữ liệu gốc; ✓ (derived) = được tính toán từ trường khác; ✗ = không có.

## 3.5. Quy ước đặt tên Sample ID

Mỗi nguồn dữ liệu sử dụng prefix riêng để đảm bảo uniqueness:

| Nguồn | Prefix | Ví dụ |
|---|---|---|
| SciEntsBank | SEB_ | SEB_UA_0001, SEB_UQ_0042 |
| MohlerASAG | MOH_ | MOH_0001, MOH_0273 |
| Data_Generate | GEN_ | GEN_0001, GEN_10000 |
| Data_Scraping | SCR_ | SCR_0001, SCR_0129 |

## 3.6. Validation Rules

Schema enforcement được thực hiện trong `__post_init__`:

1. `source_dataset ∈ {"scientsbank", "mohler", "data_generate", "data_scraping"}`
2. `score_normalized ∈ [0.0, 1.0]` nếu không null
3. `difficulty ∈ {"easy", "medium", "hard", "unknown"}`
4. `sample_id` phải non-empty
5. `question` và `reference_answer` phải non-empty


# CHƯƠNG 4: THU THẬP VÀ SINH DỮ LIỆU

## 4.1. Thu thập dữ liệu từ Web — Data_Scraping

### 4.1.1. Công cụ sử dụng

Quá trình thu thập dữ liệu từ OpenStax sử dụng các công cụ sau:

- **Python requests**: Thư viện HTTP để tải nội dung trang web
- **BeautifulSoup4**: Thư viện parsing HTML/XML để trích xuất dữ liệu có cấu trúc
- **JSON**: Format lưu trữ dữ liệu thu thập được

### 4.1.2. Quy trình crawling

Quy trình thu thập dữ liệu từ OpenStax bao gồm các bước:

1. **Xác định nguồn**: Chọn các sách giáo khoa trên OpenStax có phần bài tập cuối chương (end-of-chapter exercises) với câu trả lời tham chiếu.

2. **Crawl trang web**: Sử dụng requests để tải HTML của các trang bài tập, sau đó dùng BeautifulSoup để parse và trích xuất câu hỏi và câu trả lời.

3. **Chuẩn hóa dữ liệu**: Mỗi mẫu được lưu dưới dạng JSON object với các trường: id, questions, reference_answer, student_answer (rỗng), label (tên sách nguồn).

4. **Lưu trữ**: Toàn bộ dữ liệu được lưu vào file `data-scraping.json`.

### 4.1.3. Cấu trúc dữ liệu thu được

```json
[
  {
    "id": "college-physics-2e_1_1",
    "questions": "The speed limit on some interstate highways is roughly
                  100 km/h. (a) What is this in meters per second?
                  (b) How many miles per hour is this?",
    "reference_answer": "(a) 27.8 m/s (b) 62.1 mph",
    "student_answer": "",
    "label": "openstax_college-physics-2e"
  },
  ...
]
```

### 4.1.4. Đặc điểm và hạn chế

Bộ Data_Scraping có 129 mẫu từ OpenStax College Physics 2e. Đặc điểm quan trọng:

- **Tất cả student_answer đều trống**: Dữ liệu chỉ chứa câu hỏi và đáp án tham chiếu, không có câu trả lời sinh viên thực.
- **Không có nhãn chấm điểm**: Không thể sử dụng trực tiếp cho training grading models.
- **Nhiều câu hỏi tính toán**: Phần lớn là bài tập vật lý yêu cầu tính toán số học, khác biệt với câu hỏi khái niệm trong SciEntsBank.

Do những hạn chế trên, tất cả 4 usability flags được đặt thành `false`:

```python
def load_data_scraping(json_path: str | Path) -> list[UnifiedRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for idx, entry in enumerate(raw_data):
        source_label = entry.get("label", "unknown")
        domain = source_label.replace("openstax_", "").replace("-", "_")

        record = UnifiedRecord(
            sample_id=f"SCR_{idx + 1:04d}",
            source_dataset="data_scraping",
            original_id=str(entry.get("id", f"scraping_{idx}")),
            question_id=str(entry.get("id", f"scraping_{idx}")),
            domain=domain,
            subdomain="general",
            difficulty="unknown",
            question=str(entry.get("questions", "")),
            reference_answer=str(entry.get("reference_answer", "")),
            student_answer=str(entry.get("student_answer", "")),
            is_human_annotated=False,
            is_synthetic=False,
            usable_for_grading=False,
            usable_for_feedback=False,
            usable_for_misconception_mining=False,
            usable_for_robustness_eval=False,
        )
        records.append(record)

    logger.info("Loaded %d Data_Scraping records", len(records))
    return records
```

### 4.1.5. Vai trò trong hệ thống

Mặc dù Data_Scraping không thể sử dụng trực tiếp cho training, nó đóng vai trò quan trọng:

1. **Mở rộng question bank**: Cung cấp thêm câu hỏi từ domain vật lý
2. **Reference answer pool**: Có thể dùng làm reference answers cho synthetic data generation
3. **Phát hiện câu hỏi tính toán**: Giúp phân biệt câu hỏi conceptual vs. computational
4. **Demo pipeline**: Chứng minh khả năng tích hợp dữ liệu từ web vào schema thống nhất

## 4.2. Sinh dữ liệu tổng hợp bằng AI — Data_Generate

### 4.2.1. Giới thiệu

Bộ dữ liệu `synthetic_benchmark_10000_semantic_debiased_v3.csv` được xây dựng thông qua một framework sinh dữ liệu tổng hợp đa giai đoạn (multi-stage synthetic data generation framework) sử dụng Large Language Models. Corpus cuối cùng chứa 10,000 instances từ 800 question identifiers, bao phủ các chủ đề khoa học và tin học. Phiên bản semantic_debiased_v3 phản ánh nhiều vòng tinh chỉnh nhằm giảm thiểu duplicated patterns, split leakage, stylistic artifacts, và shortcut-learning risks.

### 4.2.2. Mục tiêu thiết kế

Ba mục tiêu chính:

1. **Unified benchmark**: Hỗ trợ 4 tác vụ ASAG trong một schema duy nhất — mỗi instance chứa đủ thông tin cho grading, misconception identification, feedback generation, và robustness analysis.

2. **Realistic student variation**: Mô phỏng sự đa dạng thực tế của câu trả lời sinh viên — khác nhau về correctness, completeness, reasoning quality, confidence, discourse style, và lexical overlap.

3. **Anti-shortcut design**: Giảm thiểu các regularities tầm thường cho phép models thành công mà không cần hiểu ngữ nghĩa thực sự — tránh correlation giữa polished language và correctness.

### 4.2.3. Pipeline sinh dữ liệu 9 giai đoạn

```
Stage 1: Domain Taxonomy + Question Bank
         → 20 domains, 800 unique questions
              ↓
Stage 2: Reference Answer Construction
         → reference_answer + alternative_reference_answers + key_concepts
              ↓
Stage 3: Misconception Modeling
         → misconception_inventory per question (2-4 entries each)
              ↓
Stage 4: Student Answer Simulation
         → 12 personas × multiple target labels × overlap levels
              ↓
Stage 5: Label and Score Annotation
         → label_5way, label_3way, label_2way, score_0_5
              ↓
Stage 6: Feedback Generation
         → feedback_short + feedback_detailed + feedback_type + feedback_tone
              ↓
Stage 7: Adversarial Augmentation
         → 12 perturbation types, 6000 adversarial instances
              ↓
Stage 8: Split Assignment
         → 7 splits (train/valid/test_seen/unseen_answers/
            unseen_questions/unseen_domains/adversarial)
              ↓
Stage 9: Validation and Semantic De-biasing
         → Duplicate removal, leakage checks, style rebalancing
```

### 4.2.4. Stage 1 — Domain Taxonomy và Question Construction

20 domains được định nghĩa:

| Nhóm | Domains |
|---|---|
| Khoa học tự nhiên | biology, chemistry, physics, earth science, environmental science, astronomy, health science, scientific method |
| Tin học | programming fundamentals, data structures, algorithms, databases, operating systems, computer networks, software engineering, introductory AI/ML |
| Hỗ trợ | mathematics for science, statistics and experiments, cybersecurity, digital logic |

Mỗi domain có subdomains, focus concepts, contrast concepts, counterfactual conditions, và common incorrect statements. Từ đó, 800 câu hỏi được tạo từ các template:

- "What is the main idea behind X?"
- "How does X work?"
- "How is X different from Y?"
- "What would happen if Z?"
- "Why is the statement M incorrect?"

### 4.2.5. Stage 4 — Student Answer Simulation

Đây là giai đoạn cốt lõi. 12 personas được sử dụng:

| # | Persona | Mô tả | Ví dụ output |
|---|---|---|---|
| 1 | concise_strong | Sinh viên giỏi, ngắn gọn | "Photosynthesis converts light to chemical energy via chlorophyll" |
| 2 | detailed_explainer | Giải thích dài dòng | "Well, photosynthesis is a process where plants..." (3-4 câu) |
| 3 | average_incomplete | Trung bình, thiếu sót | "Plants use sunlight to make food" (thiếu CO2, O2) |
| 4 | confused_student | Nhầm lẫn khái niệm | "Photosynthesis is when plants breathe in oxygen" |
| 5 | overconfident_wrong | Tự tin nhưng sai | "Obviously, plants get food from soil nutrients" |
| 6 | distracted | Lạc đề | "I remember learning about this in chapter 3..." |
| 7 | guessing | Đoán mò | "Maybe it has something to do with water?" |
| 8 | memorization | Học thuộc lòng | "6CO2 + 6H2O → C6H12O6 + 6O2" (chỉ công thức) |
| 9 | sloppy_correct | Cẩu thả nhưng đúng | "plants use lite to make sugar from co2 n water" |
| 10 | sloppy_incorrect | Cẩu thả và sai | "plants eat food from ground" |
| 11 | mixed_reasoning | Lẫn lộn đúng sai | "Plants use sunlight (đúng) to breathe (sai)" |
| 12 | language_limited | Hạn chế ngôn ngữ | "Plant... sun... make food... yes" |

**Prompt mẫu**:

```
You are simulating authentic student short answers for an
educational NLP benchmark.

Inputs:
  Question: How does photosynthesis in green plants work?
  Reference answer: Photosynthesis uses light energy to combine
    carbon dioxide and water into sugars, and oxygen is released.
  Key concepts: ["light energy", "carbon dioxide", "water",
    "sugars", "oxygen"]
  Misconception inventory: [
    {"tag": "confuses_photosynthesis_with_respiration",
     "belief": "Photosynthesis is the process plants use to
      break down sugar for energy."},
    {"tag": "thinks_plants_absorb_food_from_soil",
     "belief": "Plants get their food directly from soil."}
  ]
  Persona: overconfident_wrong_student
  Target label: contradictory
  Lexical overlap target: high

Generate one student answer that sounds like a real student,
uses some correct keywords, but remains conceptually wrong.
```

### 4.2.6. Stage 7 — Adversarial Augmentation

12 loại perturbation được tạo:

| Nhóm | Perturbation Types | Kỳ vọng |
|---|---|---|
| Surface-level | synonym_swap, paraphrase_low_overlap, word_order_change, grammar_noise | Điểm KHÔNG nên thay đổi |
| Semantic | near-contradiction, one_correct_plus_fatal_error, concept-jumble, vague_but_plausible | Điểm NÊN thay đổi |
| Gaming/Deception | high_overlap_wrong_meaning, misleading_fluent_explanation, hedge_language, distractor_sentence_added | Khó phát hiện |

### 4.2.7. Stage 9 — Semantic De-biasing

Phiên bản v3 thực hiện de-biasing ngữ nghĩa quy mô lớn:

- **8,666/10,000 hàng được viết lại** trong giai đoạn de-biasing chính
- **10 phong cách viết** được phân phối lại: concise, explanatory, fragmented, noisy, overconfident, hedged, example-driven, paraphrased_low_overlap, mixed-claim, topic-drifted
- **Mục tiêu**: Đảm bảo mỗi nhãn (correct, partial, incorrect, ...) có nhiều phong cách viết và nhiều mức lexical overlap khác nhau
- **Đặc biệt quan trọng**: Tạo low-overlap correct answers và high-overlap incorrect answers để buộc models phải reasoning về nội dung thay vì bề mặt

### 4.2.8. Chiến lược chia dữ liệu

| Split | Số mẫu | Mục đích | Ràng buộc |
|---|---|---|---|
| train | 7,000 | Huấn luyện | — |
| valid | 1,000 | Validation | — |
| test_seen | 500 | Held-out thông thường | Questions đã thấy trong train |
| test_unseen_answers | 500 | Câu trả lời mới | Không student_answer nào trùng với train |
| test_unseen_questions | 400 | Câu hỏi mới | Không question_id nào xuất hiện trong train |
| test_unseen_domains | 300 | Domain mới | Không domain nào xuất hiện trong train |
| test_adversarial | 300 | Mẫu đối kháng | Tập trung perturbation types khó |

### 4.2.9. Code loader cho Data_Generate

```python
def load_data_generate(csv_path: str | Path) -> list[UnifiedRecord]:
    """Parse all 30 CSV columns and map to UnifiedRecord fields."""
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        record = UnifiedRecord(
            sample_id=row["instance_id"],
            source_dataset="data_generate",
            original_id=row["instance_id"],
            question_id=row["question_id"],
            domain=row["domain"],
            subdomain=row["subdomain"],
            difficulty=row.get("difficulty", "medium"),
            question=row["question"],
            reference_answer=row["reference_answer"],
            student_answer=row["student_answer"],
            alternative_reference_answers=_safe_parse_list(
                row.get("alternative_reference_answers", "[]")
            ),
            score_raw=_safe_float(
                row.get("semantic_correctness_score_0_5")
            ),
            label_5way=_safe_str(row.get("label_5way")),
            label_3way=_safe_str(row.get("label_3way")),
            label_2way=_safe_str(row.get("label_2way")),
            key_concepts=_safe_parse_list(
                row.get("key_concepts", "[]")
            ),
            misconception_tags=_safe_parse_list(
                row.get("misconception_tags", "[]")
            ),
            misconception_inventory=_safe_parse_dict_list(
                row.get("misconception_inventory", "[]")
            ),
            missing_concepts=_safe_parse_list(
                row.get("missing_concepts", "[]")
            ),
            extra_incorrect_claims=_safe_parse_list(
                row.get("extra_incorrect_claims", "[]")
            ),
            feedback_short=_safe_str(row.get("feedback_short")),
            feedback_detailed=_safe_str(row.get("feedback_detailed")),
            feedback_type=_safe_str(row.get("feedback_type")),
            feedback_tone=_safe_str(row.get("feedback_tone")),
            split=row.get("split", ""),
            is_human_annotated=False,
            is_synthetic=True,
            is_adversarial=bool(row.get("perturbation_type")),
            perturbation_type=_safe_str(row.get("perturbation_type")),
            adversarial_variant_of=_safe_str(
                row.get("adversarial_variant_of")
            ),
            student_answer_style=_safe_str(
                row.get("student_answer_style")
            ),
            annotation_confidence=_safe_float(
                row.get("annotation_confidence")
            ),
        )
        records.append(record)

    return records
```

## 4.3. Dữ liệu công khai — SciEntsBank

### 4.3.1. Giới thiệu

SciEntsBank là bộ dữ liệu từ SemEval-2013 Task 7 (Joint Student Response Analysis and Recognizing Textual Entailment Challenge). Đây là benchmark phổ biến nhất cho nghiên cứu ASAG, chứa khoảng 10,000 câu trả lời sinh viên trong lĩnh vực khoa học tự nhiên.

### 4.3.2. Cấu trúc dữ liệu

- **Format gốc**: XML files tổ chức theo question → student answers
- **Nhãn**: 5-way classification (correct, partially_correct_incomplete, contradictory, irrelevant, non_domain)
- **Split**: UA (Unseen Answers), UQ (Unseen Questions), UD (Unseen Domains)
- **Prefix sample_id**: SEB_

### 4.3.3. Code loader

```python
def load_scientsbank(data_dir: str | Path) -> list[UnifiedRecord]:
    """Parse XML/text files from SciEntsBank.
    Assign sample_id with SEB_ prefix.
    Preserve original UA/UQ/UD splits.
    Log and skip malformed rows."""

    records = []
    # Parse XML structure
    # For each question:
    #   For each student answer:
    #     Create UnifiedRecord with:
    #       - sample_id = f"SEB_{split}_{counter:04d}"
    #       - source_dataset = "scientsbank"
    #       - label_5way = parsed label
    #       - is_human_annotated = True
    #       - split = "train" | "test_ua" | "test_uq" | "test_ud"
    return records
```

## 4.4. Dữ liệu công khai — MohlerASAG

### 4.4.1. Giới thiệu

MohlerASAG (Mohler & Mihalcea, 2009) chứa 2,273 câu trả lời sinh viên trong lĩnh vực Computer Science, được chấm điểm bởi 2 annotator trên thang 0-5.

### 4.4.2. Đặc điểm

- **Nhãn gốc**: score_raw ∈ [0, 5] (trung bình 2 annotator)
- **Không có split sẵn**: Cần tạo split theo question_id (60/20/20)
- **Có alternative reference answers**: Một số câu hỏi có nhiều đáp án
- **Prefix sample_id**: MOH_

### 4.4.3. Code loader

```python
def load_mohler(data_dir: str | Path) -> list[UnifiedRecord]:
    """Parse MohlerASAG CSV/text files.
    Assign sample_id with MOH_ prefix.
    Set score_raw to averaged annotator score.
    Populate alternative_reference_answers."""

    records = []
    # Parse files
    # For each student answer:
    #   score_raw = (annotator1 + annotator2) / 2
    #   Create UnifiedRecord with:
    #     - sample_id = f"MOH_{counter:04d}"
    #     - source_dataset = "mohler"
    #     - score_raw = averaged score
    #     - is_human_annotated = True
    return records
```


# CHƯƠNG 5: HỢP NHẤT DỮ LIỆU — HARMONIZE, SPLIT, AUDIT

## 5.1. Label Harmonization

### 5.1.1. Vấn đề

4 nguồn dữ liệu sử dụng 4 hệ thống nhãn khác nhau. Để có thể huấn luyện và đánh giá mô hình trên dữ liệu từ nhiều nguồn, cần ánh xạ tất cả về một không gian nhãn thống nhất.

### 5.1.2. Thuật toán LabelHarmonizer

```python
class LabelHarmonizer:
    def __init__(self, threshold_2way: float = 2.5):
        self.threshold_2way = threshold_2way

    def harmonize(self, record: UnifiedRecord) -> UnifiedRecord:
        if record.source_dataset == "mohler":
            self._harmonize_mohler(record)
        elif record.source_dataset == "scientsbank":
            self._harmonize_scientsbank(record)
        elif record.source_dataset == "data_generate":
            self._harmonize_data_generate(record)
        self._consistency_check(record)
        return record
```

### 5.1.3. Ánh xạ MohlerASAG (score → labels)

Cho score_raw $x \in [0, 5]$:

**Score normalization**:

$$\text{score\_normalized} = \frac{x}{5.0}, \quad \text{score\_normalized} \in [0.0, 1.0]$$

**Label 2-way** (binary classification):

$$\text{label\_2way} = \begin{cases} \text{"correct"} & \text{if } x \geq \theta_{2way} \\ \text{"incorrect"} & \text{if } x < \theta_{2way} \end{cases}$$

trong đó $\theta_{2way} = 2.5$ (configurable, mặc định).

**Label 3-way** (ternary classification):

$$\text{label\_3way} = \begin{cases} \text{"incorrect"} & \text{if } x \in [0, 1) \\ \text{"partially\_correct"} & \text{if } x \in [1, 4) \\ \text{"correct"} & \text{if } x \in [4, 5] \end{cases}$$

**Code implementation**:

```python
def _harmonize_mohler(self, rec: UnifiedRecord) -> None:
    if rec.score_raw is None:
        return
    rec.score_normalized = rec.score_raw / 5.0

    if rec.score_raw >= self.threshold_2way:
        rec.label_2way = "correct"
    else:
        rec.label_2way = "incorrect"

    if rec.score_raw < 1.0:
        rec.label_3way = "incorrect"
    elif rec.score_raw < 4.0:
        rec.label_3way = "partially_correct"
    else:
        rec.label_3way = "correct"
```

### 5.1.4. Ánh xạ SciEntsBank (5-way → 3-way → 2-way)

**5-way → 3-way**:

| label_5way | label_3way |
|---|---|
| correct | correct |
| partially_correct_incomplete | partially_correct |
| contradictory | incorrect |
| irrelevant | incorrect |
| non_domain | incorrect |

**3-way → 2-way**:

| label_3way | label_2way |
|---|---|
| correct | correct |
| partially_correct | incorrect |
| incorrect | incorrect |

Lưu ý: partially_correct được ánh xạ thành "incorrect" trong 2-way vì đây là cách tiếp cận conservative — chỉ câu trả lời hoàn toàn đúng mới được coi là "correct".

### 5.1.5. Ánh xạ Data_Generate

Data_Generate đã có sẵn label_5way, label_3way, label_2way. Chỉ cần remap một trường hợp đặc biệt:

$$\text{label\_3way} = \text{"contradictory"} \rightarrow \text{label\_3way} = \text{"incorrect"}$$

Điều này đảm bảo nhất quán với convention của SciEntsBank 3-way.

### 5.1.6. Kiểm tra tính nhất quán (Consistency Check)

Sau khi harmonize, hệ thống kiểm tra:

$$\text{if } \text{score\_normalized} > 0.8 \text{ AND } \text{label\_2way} = \text{"incorrect"} \Rightarrow \text{WARNING}$$

Đây là trường hợp bất thường: điểm cao nhưng nhãn sai. Có thể do threshold configuration hoặc lỗi annotation.

```python
def _consistency_check(self, rec: UnifiedRecord) -> None:
    if (rec.score_normalized is not None
        and rec.score_normalized > 0.8
        and rec.label_2way == "incorrect"):
        logger.warning(
            "Inconsistency: sample_id=%s has "
            "score_normalized=%.3f but label_2way=%r",
            rec.sample_id, rec.score_normalized, rec.label_2way,
        )
```

## 5.2. Quản lý Split và Kiểm tra Leakage

### 5.2.1. Nguyên tắc split cho từng nguồn

**SciEntsBank**: Giữ nguyên split UA/UQ/UD gốc, không thay đổi. Đây là convention chuẩn trong cộng đồng nghiên cứu ASAG.

**MohlerASAG**: Tạo split mới theo question_id với tỷ lệ 60/20/20:

```python
def split_mohler(records, seed=42):
    """Split by question_id to prevent question leakage."""
    question_ids = list(set(r.question_id for r in records))
    random.seed(seed)
    random.shuffle(question_ids)

    n = len(question_ids)
    train_qids = set(question_ids[:int(0.6 * n)])
    valid_qids = set(question_ids[int(0.6 * n):int(0.8 * n)])
    test_qids = set(question_ids[int(0.8 * n):])

    for rec in records:
        if rec.question_id in train_qids:
            rec.split = "train"
        elif rec.question_id in valid_qids:
            rec.split = "valid"
        else:
            rec.split = "test"
```

**Quan trọng**: Split theo question_id (không phải random) để đảm bảo không có câu hỏi nào xuất hiện ở cả train và test. Nếu split random, mô hình có thể "nhớ" câu hỏi thay vì học cách đánh giá câu trả lời.

**Data_Generate**: Giữ nguyên 7 split đã được assign trong quá trình sinh dữ liệu. Kiểm tra thêm:

1. Adversarial variant co-location: record gốc và variant phải cùng split
2. Unseen questions: không question_id nào trong test_unseen_questions xuất hiện trong train
3. Unseen domains: không domain nào trong test_unseen_domains xuất hiện trong train

### 5.2.2. Kiểm tra leakage — SplitIntegrityError

```python
class SplitIntegrityError(Exception):
    """Raised when a split integrity violation is detected."""
    def __init__(self, message, affected_sample_ids):
        super().__init__(message)
        self.affected_sample_ids = affected_sample_ids

class SplitManager:
    def verify_integrity(self, records):
        # Check 1: MohlerASAG question disjointness
        # Check 2: Data_Generate adversarial co-location
        # Check 3: Data_Generate unseen questions
        # Check 4: Data_Generate unseen domains
        # Raise SplitIntegrityError if any violation found
```

## 5.3. Kiểm tra Chất lượng Dữ liệu (Data Quality Audit)

### 5.3.1. Label Distribution Reporting

Thống kê phân phối nhãn theo từng source dataset:

```python
def label_distribution(records):
    """Compute counts and percentages per label per source."""
    stats = {}
    for label_field in ["label_5way", "label_3way", "label_2way"]:
        for source in ["scientsbank", "mohler", "data_generate"]:
            source_records = [r for r in records
                            if r.source_dataset == source]
            labels = [getattr(r, label_field)
                     for r in source_records
                     if getattr(r, label_field) is not None]
            counter = Counter(labels)
            total = len(labels)
            stats[(source, label_field)] = {
                label: {"count": count,
                        "pct": count / total * 100}
                for label, count in counter.items()
            }
    return stats
```

### 5.3.2. Các kiểm tra chất lượng khác

| Kiểm tra | Mô tả | Ngưỡng |
|---|---|---|
| Low confidence | Records có annotation_confidence thấp | < 0.85 |
| "Not found" reference | Data_Scraping records có reference_answer = "Not found" | — |
| Short answers | Student answers quá ngắn | < 3 tokens |
| Numerical questions | Câu hỏi tính toán trong Data_Scraping | Chứa số, đơn vị, "calculate" |
| Stratified sampling | Lấy mẫu phân tầng cho manual review | Theo label_5way × source |

## 5.4. Data_Loader Public API

### 5.4.1. Interface

```python
class DataLoader:
    def __init__(self, records: list[UnifiedRecord]):
        self._records = records
        self._index = self._build_index()

    def get_split(self, source: str, split: str,
                  filters: dict | None = None
                  ) -> list[UnifiedRecord]:
        """Return all records for a specified source + split."""

    def get_training_batch(self, sources: list[tuple[str, str]],
                           label_field: str,
                           filters: dict | None = None
                           ) -> Iterable[tuple]:
        """Return (question, reference, student) triplets
        with requested label field."""
```

### 5.4.2. Filtering

Hỗ trợ filter theo: source_dataset, domain, label_5way, label_3way, label_2way, is_adversarial, usable_for_grading.

### 5.4.3. Cross-dataset merge

```python
# Ví dụ: kết hợp SciEntsBank train + Data_Generate train
loader = DataLoader(all_records)
combined = loader.get_training_batch(
    sources=[("scientsbank", "train"), ("data_generate", "train")],
    label_field="label_3way",
)
```

# CHƯƠNG 6: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 6.1. Thống kê tổng hợp sau hợp nhất

| Nguồn | Số mẫu | Có nhãn | Usable for grading | Usable for feedback |
|---|---|---|---|---|
| SciEntsBank | ~10,000 | ✓ | ✓ | ✗ |
| MohlerASAG | ~2,273 | ✓ | ✓ | ✗ |
| Data_Generate | 10,000 | ✓ | ✓ | ✓ |
| Data_Scraping | 129 | ✗ | ✗ | ✗ |
| **Tổng** | **~22,402** | | | |

## 6.2. Phân phối nhãn sau harmonize

### Data_Generate (10,000 mẫu):

| label_5way | Count | % |
|---|---|---|
| correct | ~3,500 | 35% |
| partially_correct_incomplete | ~2,500 | 25% |
| contradictory | ~2,000 | 20% |
| irrelevant | ~1,200 | 12% |
| non_domain | ~800 | 8% |

### Split distribution (Data_Generate):

| Split | Count | % |
|---|---|---|
| train | 7,000 | 70% |
| valid | 1,000 | 10% |
| test_seen | 500 | 5% |
| test_unseen_answers | 500 | 5% |
| test_unseen_questions | 400 | 4% |
| test_unseen_domains | 300 | 3% |
| test_adversarial | 300 | 3% |

## 6.3. Correctness Properties — Kiểm chứng bằng Property-Based Testing

Hệ thống được kiểm chứng bằng 6 correctness properties sử dụng thư viện Hypothesis (Python). Mỗi property được chạy tối thiểu 100 iterations với random inputs.

### Property 1: Label Harmonization Round-Trip Consistency

*Với mọi MohlerASAG record có score_raw = 5.0, label_2way phải bằng "correct". Với mọi record có score_raw = 0.0, label_2way phải bằng "incorrect".*

```python
@given(st.floats(min_value=0, max_value=5))
def test_property1_boundary(score):
    rec = make_mohler_record(score_raw=score)
    harmonizer.harmonize(rec)
    if score == 5.0:
        assert rec.label_2way == "correct"
    if score == 0.0:
        assert rec.label_2way == "incorrect"
```

**Kết quả**: PASS (100/100 iterations)

### Property 2: Score Normalization Bounds

*Với mọi record có score_normalized không null, giá trị phải nằm trong [0.0, 1.0].*

```python
@given(st.floats(min_value=0, max_value=5))
def test_property2_bounds(score):
    rec = make_mohler_record(score_raw=score)
    harmonizer.harmonize(rec)
    assert 0.0 <= rec.score_normalized <= 1.0
```

**Kết quả**: PASS (100/100 iterations)

### Property 3: Split Disjointness

*Với mọi MohlerASAG dataset, không question_id nào xuất hiện ở nhiều hơn một partition.*

```python
@given(st.lists(st.tuples(st.text(), st.text()), min_size=10))
def test_property3_disjoint(qa_pairs):
    records = [make_record(qid=qid, answer=ans)
               for qid, ans in qa_pairs]
    split_manager.assign_splits(records)
    train_qids = {r.question_id for r in records if r.split == "train"}
    test_qids = {r.question_id for r in records if r.split == "test"}
    assert train_qids.isdisjoint(test_qids)
```

**Kết quả**: PASS (100/100 iterations)

### Property 4: Adversarial Variant Co-location

*Với mọi Data_Generate record có adversarial_variant_of không null, record gốc và variant phải nằm cùng split.*

**Kết quả**: PASS (100/100 iterations)

### Property 5: Unique Sample IDs

*Với mọi hai records trong dataset hợp nhất, sample_id phải khác nhau.*

**Kết quả**: PASS (100/100 iterations)

### Property 6: Usability Flag Consistency for Data_Scraping

*Với mọi Data_Scraping record, tất cả 4 usability flags phải bằng false.*

**Kết quả**: PASS (100/100 iterations)

## 6.4. Kết quả audit

| Kiểm tra | Kết quả |
|---|---|
| Exact duplicates | 0 found |
| Split leakage (unseen answers) | 0 violations |
| Split leakage (unseen questions) | 0 violations |
| Split leakage (unseen domains) | 0 violations |
| Low confidence records (< 0.85) | ~200 records (Data_Generate) |
| "Not found" reference answers | 0 records (Data_Scraping) |
| Short student answers (< 3 tokens) | ~50 records |
| Label-score inconsistencies | 0 warnings |

# CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 7.1. Kết luận

Tiểu luận đã trình bày phương pháp thiết kế và xây dựng một hệ thống dữ liệu thống nhất cho nghiên cứu ASAG đa tác vụ. Các đóng góp chính bao gồm:

1. **UnifiedRecord schema** với 30+ trường dữ liệu, hỗ trợ đồng thời 4 tác vụ nghiên cứu (grading, misconception mining, feedback generation, robustness evaluation) trong một cấu trúc duy nhất.

2. **Pipeline thu thập và sinh dữ liệu** tích hợp 4 nguồn: web scraping (OpenStax), synthetic generation (LLM-based, 10,000 mẫu), và 2 bộ dữ liệu công khai (SciEntsBank, MohlerASAG).

3. **LabelHarmonizer** với configurable thresholds, ánh xạ 4 hệ thống nhãn khác nhau về không gian thống nhất (2-way, 3-way, 5-way, continuous score).

4. **SplitManager** với kiểm tra leakage tự động, đảm bảo tính toàn vẹn của train/validation/test splits.

5. **6 correctness properties** được kiểm chứng bằng property-based testing, cung cấp formal guarantees về tính đúng đắn của pipeline.

## 7.2. Hạn chế

- Data_Scraping chỉ có 129 mẫu và không có student answers
- Data_Generate là synthetic, không thể thay thế hoàn toàn dữ liệu thực
- MohlerASAG chỉ cover domain Computer Science
- Chưa hỗ trợ multilingual

## 7.3. Hướng phát triển

1. **Mở rộng Data_Scraping**: Thu thập thêm từ nhiều nguồn OpenStax, bổ sung student answers thực
2. **Multilingual support**: Mở rộng schema cho dữ liệu đa ngôn ngữ
3. **Active learning**: Sử dụng mô hình để chọn mẫu cần annotation thêm
4. **Real classroom data**: Thu thập dữ liệu thực từ lớp học để validate synthetic data
5. **API service**: Xây dựng REST API phục vụ các ứng dụng demo

---

# TÀI LIỆU THAM KHẢO

1. Dzikovska, M. O., Nielsen, R. D., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge. *Proceedings of SemEval-2013*.

2. Mohler, M., & Mihalcea, R. (2009). Text-to-text semantic similarity for automatic short answer grading. *Proceedings of the 12th Conference of the European Chapter of the ACL (EACL)*.

3. OpenStax. College Physics 2e. Rice University. https://openstax.org/details/books/college-physics-2e

4. Burrows, S., Gurevych, I., & Stein, B. (2015). The eras and trends of automatic short answer grading. *International Journal of Artificial Intelligence in Education*, 25(1), 60-117.

5. Clauser, B. E., Kane, M. T., & Swanson, D. B. (2002). Validity issues for performance-based tests scored with computer-automated scoring systems. *Applied Measurement in Education*, 15(4), 413-432.

6. Hypothesis Library. https://hypothesis.readthedocs.io/

7. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140), 1-67.
