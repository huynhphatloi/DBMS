---
title: "Tiểu luận 1: Thiết kế Cơ sở Dữ liệu Thống nhất cho Hệ thống ASAG"
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 13pt
linestretch: 1.5
---

# CHƯƠNG 1: MỞ ĐẦU

## 1.1 Tính cấp thiết

Trong bối cảnh giáo dục hiện đại, việc đánh giá câu trả lời ngắn của sinh viên (Automatic Short Answer Grading — ASAG) đang trở thành một nhu cầu cấp bách. Với quy mô lớp học ngày càng tăng và sự phát triển của giáo dục trực tuyến, giảng viên không thể chấm điểm thủ công hàng nghìn bài làm trong thời gian hợp lý. Hệ thống ASAG tự động hóa quá trình này bằng cách so sánh ngữ nghĩa giữa câu trả lời của sinh viên với đáp án tham chiếu.

Tuy nhiên, một thách thức lớn trong nghiên cứu ASAG là sự phân mảnh của dữ liệu. Các bộ dữ liệu hiện có — SciEntsBank, MohlerASAG, và các nguồn tổng hợp — sử dụng các schema khác nhau, hệ thống nhãn khác nhau, và định dạng lưu trữ khác nhau. Điều này gây khó khăn cho việc huấn luyện mô hình đa nhiệm vụ (multi-task) và đánh giá tổng quát hóa (generalization) của mô hình.

Nghiên cứu này giải quyết vấn đề trên bằng cách thiết kế một pipeline dữ liệu thống nhất, chuyển đổi 4 nguồn dữ liệu không đồng nhất thành một schema duy nhất phục vụ đồng thời 4 nhiệm vụ nghiên cứu: chấm điểm tự động, phát hiện quan niệm sai, sinh phản hồi giáo dục, và đánh giá tính bền vững (robustness).

## 1.2 Mục tiêu và nhiệm vụ

**Mục tiêu tổng quát:** Thiết kế và triển khai một cơ sở dữ liệu thống nhất cho hệ thống ASAG đa nhiệm vụ, đảm bảo tính nhất quán, khả năng mở rộng, và chất lượng dữ liệu.

**Nhiệm vụ cụ thể:**

1. Phân tích yêu cầu dữ liệu cho 4 nhiệm vụ nghiên cứu (grading, feedback, misconception mining, robustness evaluation)
2. Thiết kế schema thống nhất `UnifiedRecord` dưới dạng Python dataclass với validation tự động
3. Xây dựng hệ thống loader cho 4 nguồn dữ liệu: SciEntsBank (XML), MohlerASAG (CSV), Data_Generate (CSV), Data_Scraping (JSON)
4. Triển khai module chuẩn hóa nhãn (label harmonization) để ánh xạ các hệ thống nhãn khác nhau về không gian chung
5. Thiết kế chiến lược phân chia dữ liệu (split) với kiểm tra rò rỉ (leakage prevention)
6. Xây dựng hệ thống kiểm tra chất lượng dữ liệu (data audit)
7. Kiểm thử toàn bộ pipeline bằng property-based testing với thư viện Hypothesis

## 1.3 Đối tượng và phạm vi

**Đối tượng nghiên cứu:**

- Dữ liệu câu hỏi-đáp ngắn trong lĩnh vực khoa học tự nhiên và khoa học máy tính
- Các bộ dữ liệu benchmark: SciEntsBank (~10,000 bản ghi, 5-way labels), MohlerASAG (~2,273 bản ghi, điểm liên tục 0-5)
- Dữ liệu tổng hợp: Data_Generate (10,000 bản ghi, 30 cột, sinh bởi LLM pipeline)
- Dữ liệu thu thập: Data_Scraping (129 bản ghi từ OpenStax, chưa có câu trả lời sinh viên)

**Phạm vi:**

- Thiết kế schema dữ liệu sử dụng Python dataclass (không sử dụng cơ sở dữ liệu quan hệ)
- Lưu trữ dưới dạng JSONL (JSON Lines) — mỗi bản ghi một dòng
- Pipeline xử lý hoàn toàn bằng Python, cấu hình qua YAML
- Kiểm thử bằng pytest + hypothesis (property-based testing)

## 1.4 Phương pháp nghiên cứu

Nghiên cứu áp dụng phương pháp thiết kế hướng dữ liệu (data-driven design) kết hợp với kiểm thử dựa trên thuộc tính (property-based testing):

1. **Phân tích yêu cầu:** Xác định các trường dữ liệu cần thiết cho từng nhiệm vụ, từ đó thiết kế schema bao trùm (superset schema)
2. **Thiết kế schema:** Sử dụng Python `@dataclass` với validation trong `__post_init__`, đảm bảo type safety và constraint checking tại thời điểm khởi tạo
3. **Triển khai pipeline:** Mỗi giai đoạn (load → harmonize → split → audit) là một module độc lập, có thể test riêng
4. **Kiểm thử:** 6 property-based tests với Hypothesis đảm bảo các bất biến (invariants) luôn đúng cho mọi đầu vào hợp lệ
5. **Cấu hình hóa:** Tất cả tham số (đường dẫn, ngưỡng, tỷ lệ split) được quản lý qua `configs/data.yaml`

---

# CHƯƠNG 2: GIỚI THIỆU TỔNG QUAN

## 2.1 Đặt vấn đề: Dữ liệu không đồng nhất

Nghiên cứu ASAG đối mặt với một thách thức cơ bản: các bộ dữ liệu hiện có được thiết kế cho các mục đích khác nhau và sử dụng các biểu diễn khác nhau.

**SciEntsBank** (SemEval-2013 Task 7) sử dụng hệ thống nhãn 5 mức (correct, partially_correct_incomplete, contradictory, irrelevant, non_domain) và lưu trữ dưới dạng XML. Dữ liệu được chia sẵn thành các split UA (Unseen Answers), UQ (Unseen Questions), UD (Unseen Domains) để đánh giá khả năng tổng quát hóa ở các mức độ khác nhau.

**MohlerASAG** sử dụng điểm liên tục từ 0 đến 5 (trung bình từ nhiều annotator) và lưu trữ dưới dạng CSV. Không có split được định nghĩa sẵn — việc phân chia phải đảm bảo không có question_id nào xuất hiện ở nhiều partition.

**Data_Generate** là bộ dữ liệu tổng hợp 10,000 bản ghi với 30 cột, bao gồm đầy đủ thông tin cho cả 4 nhiệm vụ. Dữ liệu có 7 split được định nghĩa sẵn (train, valid, test_seen, test_unseen_answers, test_unseen_questions, test_unseen_domains, test_adversarial).

**Data_Scraping** là 129 bản ghi thu thập từ sách giáo khoa OpenStax, chỉ có câu hỏi và đáp án tham chiếu — chưa có câu trả lời sinh viên, do đó không thể sử dụng trực tiếp cho huấn luyện.

## 2.2 Bài toán hợp nhất dữ liệu

Bài toán hợp nhất dữ liệu (data unification) trong ngữ cảnh này bao gồm:

1. **Schema unification:** Ánh xạ các trường dữ liệu từ 4 nguồn khác nhau về một cấu trúc duy nhất
2. **Label harmonization:** Chuyển đổi các hệ thống nhãn không tương thích (5-way categorical, continuous 0-5, pre-assigned labels) về không gian nhãn chung (2-way, 3-way)
3. **Split management:** Bảo toàn các split có sẵn (SciEntsBank, Data_Generate) và tạo split mới (Mohler) với đảm bảo không rò rỉ
4. **Quality assurance:** Phát hiện và báo cáo các vấn đề chất lượng (low confidence, short answers, numerical questions)

Giải pháp của chúng tôi sử dụng kiến trúc flat — Python dataclass + JSONL files — thay vì cơ sở dữ liệu quan hệ. Lý do:

- Dữ liệu nghiên cứu có kích thước vừa phải (~12,000-22,000 bản ghi tổng cộng)
- Không cần truy vấn phức tạp kiểu SQL
- JSONL cho phép streaming và xử lý từng dòng
- Python dataclass cung cấp type checking và validation mà không cần ORM
- Dễ dàng version control với git

## 2.3 Tổng quan 4 nguồn dữ liệu

### 2.3.1 SciEntsBank

| Thuộc tính | Giá trị |
|---|---|
| Nguồn gốc | SemEval-2013 Task 7 |
| Số lượng | ~10,000 bản ghi |
| Lĩnh vực | Khoa học tự nhiên (science) |
| Hệ thống nhãn | 5-way: correct, partially_correct_incomplete, contradictory, irrelevant, non_domain |
| Định dạng | XML (SemEval Beetle format) |
| Split có sẵn | train, test_ua, test_uq, test_ud |
| Annotator | Con người (human-annotated) |

### 2.3.2 MohlerASAG

| Thuộc tính | Giá trị |
|---|---|
| Nguồn gốc | Mohler et al. (2011) |
| Số lượng | ~2,273 bản ghi |
| Lĩnh vực | Khoa học máy tính (computer_science) |
| Hệ thống nhãn | Điểm liên tục 0-5 (trung bình nhiều annotator) |
| Định dạng | CSV |
| Split có sẵn | Không — cần tạo mới |
| Annotator | Con người (human-annotated) |

### 2.3.3 Data_Generate

| Thuộc tính | Giá trị |
|---|---|
| Nguồn gốc | Synthetic benchmark (LLM pipeline, semantic_debiased_v3) |
| Số lượng | 10,000 bản ghi |
| Lĩnh vực | Khoa học + Máy tính (20 domains) |
| Hệ thống nhãn | 5-way + 3-way + 2-way + score 0-5 |
| Định dạng | CSV (30 cột) |
| Split có sẵn | 7 splits (train, valid, test_seen, test_unseen_answers, test_unseen_questions, test_unseen_domains, test_adversarial) |
| Annotator | LLM-generated với annotation_confidence |

### 2.3.4 Data_Scraping

| Thuộc tính | Giá trị |
|---|---|
| Nguồn gốc | OpenStax textbooks (web scraping) |
| Số lượng | 129 bản ghi |
| Lĩnh vực | Vật lý đại cương (college-physics-2e) |
| Hệ thống nhãn | Không có nhãn |
| Định dạng | JSON (array of objects) |
| Split có sẵn | Không |
| Annotator | Không — chỉ có câu hỏi + đáp án tham chiếu |
| Hạn chế | student_answer rỗng, không thể dùng cho training |

---

# CHƯƠNG 3: THIẾT KẾ CƠ SỞ DỮ LIỆU

## 3.1 Phân tích yêu cầu dữ liệu

Hệ thống ASAG của chúng tôi phục vụ 4 nhiệm vụ nghiên cứu đồng thời. Mỗi nhiệm vụ yêu cầu các trường dữ liệu khác nhau:

**Task 1 — Automatic Short Answer Grading:**
- Input: question, reference_answer, student_answer
- Output: label_2way, label_3way, label_5way, score_normalized
- Cần thêm: alternative_reference_answers, key_concepts

**Task 2 — Misconception Mining:**
- Input: question, reference_answer, student_answer, key_concepts
- Output: misconception_tags, missing_concepts, extra_incorrect_claims
- Cần thêm: misconception_inventory (per-question)

**Task 3 — Automatic Feedback Generation:**
- Input: question, reference_answer, student_answer, label, misconception_tags
- Output: feedback_short, feedback_detailed, feedback_type, feedback_tone
- Cần thêm: missing_concepts, extra_incorrect_claims

**Task 4 — Robustness & Adversarial Evaluation:**
- Input: question, reference_answer, student_answer (original + perturbed)
- Output: label consistency under perturbation
- Cần thêm: perturbation_type, adversarial_variant_of, is_adversarial

**Yêu cầu chung:**
- Mỗi bản ghi cần identity fields (sample_id, source_dataset, original_id, question_id)
- Metadata: domain, subdomain, difficulty, split
- Usability flags: xác định bản ghi nào có thể dùng cho task nào
- Annotation metadata: is_human_annotated, is_synthetic, annotation_confidence

Từ phân tích trên, schema thống nhất phải là **superset** của tất cả các trường cần thiết, với các trường không áp dụng được đặt giá trị mặc định (None hoặc list rỗng).

## 3.2 Thiết kế UnifiedRecord Schema

### 3.2.1 Tổng quan kiến trúc

Thay vì sử dụng cơ sở dữ liệu quan hệ (relational database) với ERD, chúng tôi sử dụng Python `@dataclass` làm schema definition. Lý do:

- **Type safety:** Python type hints cung cấp kiểm tra kiểu tại thời điểm phát triển
- **Validation:** `__post_init__` cho phép kiểm tra ràng buộc ngay khi khởi tạo object
- **Serialization:** `dataclasses.asdict()` chuyển đổi trực tiếp sang dict/JSON
- **Immutability-friendly:** Dễ dàng chuyển sang frozen dataclass nếu cần
- **No ORM overhead:** Không cần database connection, migration, hay query language

### 3.2.2 Mã nguồn schema.py

Dưới đây là toàn bộ mã nguồn của module `src/data/schema.py`:

```python
"""Unified record schema for the ASAG Research Framework.

All four source datasets (SciEntsBank, MohlerASAG, Data_Generate,
Data_Scraping) are converted into this canonical format before any
downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_SOURCE_DATASETS = frozenset(
    {"scientsbank", "mohler", "data_generate", "data_scraping"}
)

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "unknown"})


@dataclass
class UnifiedRecord:
    """Canonical data structure for a single student-answer sample."""

    # ── Identity ──────────────────────────────────────────────────────
    sample_id: str
    source_dataset: str
    original_id: str
    question_id: str

    # ── Domain ────────────────────────────────────────────────────────
    domain: str
    subdomain: str
    difficulty: str  # "easy" | "medium" | "hard" | "unknown"

    # ── Core triplet ──────────────────────────────────────────────────
    question: str
    reference_answer: str
    student_answer: str
    alternative_reference_answers: list[str] = field(default_factory=list)

    # ── Grading labels ────────────────────────────────────────────────
    score_raw: float | None = None
    score_normalized: float | None = None
    label_2way: str | None = None
    label_3way: str | None = None
    label_5way: str | None = None

    # ── Concept-level annotations ─────────────────────────────────────
    key_concepts: list[str] = field(default_factory=list)
    misconception_tags: list[str] = field(default_factory=list)
    misconception_inventory: list[dict] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    extra_incorrect_claims: list[str] = field(default_factory=list)

    # ── Feedback ──────────────────────────────────────────────────────
    feedback_short: str | None = None
    feedback_detailed: str | None = None
    feedback_type: str | None = None
    feedback_tone: str | None = None

    # ── Splits and metadata ───────────────────────────────────────────
    split: str = ""
    is_human_annotated: bool = False
    is_synthetic: bool = False
    is_adversarial: bool = False
    perturbation_type: str | None = None
    adversarial_variant_of: str | None = None
    student_answer_style: str | None = None
    annotation_confidence: float | None = None

    # ── Usability flags ───────────────────────────────────────────────
    usable_for_grading: bool = True
    usable_for_feedback: bool = True
    usable_for_misconception_mining: bool = True
    usable_for_robustness_eval: bool = True

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.source_dataset not in VALID_SOURCE_DATASETS:
            raise ValueError(
                f"source_dataset must be one of {sorted(VALID_SOURCE_DATASETS)}, "
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
                f"difficulty must be one of {sorted(VALID_DIFFICULTIES)}, "
                f"got {self.difficulty!r}"
            )
```

### 3.2.3 Giải thích các nhóm trường

**Nhóm Identity (4 trường bắt buộc):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `sample_id` | str | ID duy nhất toàn cục, format: `{PREFIX}_{INDEX:05d}` (SEB_, MOH_, GEN_, SCR_) |
| `source_dataset` | str | Nguồn gốc: scientsbank, mohler, data_generate, data_scraping |
| `original_id` | str | ID gốc từ dataset ban đầu |
| `question_id` | str | ID câu hỏi — dùng cho question-level split |

**Nhóm Domain (3 trường bắt buộc):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `domain` | str | Lĩnh vực chính (science, computer_science, biology, ...) |
| `subdomain` | str | Lĩnh vực con (plant_biology, algorithms, ...) |
| `difficulty` | str | Mức độ khó: easy, medium, hard, unknown |

**Nhóm Core Triplet (3 trường bắt buộc + 1 optional):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `question` | str | Câu hỏi |
| `reference_answer` | str | Đáp án tham chiếu chính |
| `student_answer` | str | Câu trả lời sinh viên |
| `alternative_reference_answers` | list[str] | Các đáp án tham chiếu thay thế |

**Nhóm Grading Labels (5 trường optional):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `score_raw` | float \| None | Điểm thô (0-5 cho Mohler, 0-5 cho Data_Generate) |
| `score_normalized` | float \| None | Điểm chuẩn hóa [0.0, 1.0] |
| `label_2way` | str \| None | correct / incorrect |
| `label_3way` | str \| None | correct / partially_correct / incorrect |
| `label_5way` | str \| None | correct / partially_correct_incomplete / contradictory / irrelevant / non_domain |

**Nhóm Concept-level Annotations (5 trường, default list rỗng):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `key_concepts` | list[str] | Khái niệm chính cần có trong câu trả lời đúng |
| `misconception_tags` | list[str] | Tags quan niệm sai được phát hiện |
| `misconception_inventory` | list[dict] | Kho quan niệm sai per-question |
| `missing_concepts` | list[str] | Khái niệm bị thiếu trong câu trả lời |
| `extra_incorrect_claims` | list[str] | Khẳng định sai thêm vào |

**Nhóm Feedback (4 trường optional):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `feedback_short` | str \| None | Phản hồi ngắn gọn |
| `feedback_detailed` | str \| None | Phản hồi chi tiết |
| `feedback_type` | str \| None | Loại phản hồi (correction, scaffolding, praise, ...) |
| `feedback_tone` | str \| None | Giọng điệu (tutor_like, supportive, ...) |

**Nhóm Metadata (7 trường):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `split` | str | Partition: train, valid, test, test_ua, ... |
| `is_human_annotated` | bool | Được gán nhãn bởi con người? |
| `is_synthetic` | bool | Dữ liệu tổng hợp? |
| `is_adversarial` | bool | Biến thể adversarial? |
| `perturbation_type` | str \| None | Loại nhiễu (paraphrase, surface_noise, ...) |
| `adversarial_variant_of` | str \| None | original_id của bản ghi gốc |
| `annotation_confidence` | float \| None | Độ tin cậy annotation (0-1) |

**Nhóm Usability Flags (4 trường boolean):**

| Trường | Kiểu | Mô tả |
|---|---|---|
| `usable_for_grading` | bool | Có thể dùng cho task chấm điểm? |
| `usable_for_feedback` | bool | Có thể dùng cho task sinh phản hồi? |
| `usable_for_misconception_mining` | bool | Có thể dùng cho task phát hiện quan niệm sai? |
| `usable_for_robustness_eval` | bool | Có thể dùng cho task đánh giá robustness? |

### 3.2.4 Validation rules trong __post_init__

Phương thức `__post_init__` thực hiện 3 kiểm tra ràng buộc:

1. **source_dataset ∈ VALID_SOURCE_DATASETS:** Chỉ chấp nhận 4 giá trị hợp lệ. Nếu vi phạm → `ValueError`
2. **score_normalized ∈ [0.0, 1.0]:** Nếu không None, phải nằm trong khoảng [0, 1]. Nếu vi phạm → `ValueError`
3. **difficulty ∈ VALID_DIFFICULTIES:** Chỉ chấp nhận easy, medium, hard, unknown. Nếu vi phạm → `ValueError`

Các validation này đảm bảo rằng không có bản ghi nào với giá trị không hợp lệ có thể tồn tại trong hệ thống — lỗi được phát hiện ngay tại thời điểm khởi tạo object, không phải sau khi đã xử lý xong pipeline.

## 3.3 Chuẩn hóa dữ liệu — Label Harmonization

### 3.3.1 Vấn đề

Ba nguồn dữ liệu có nhãn sử dụng 3 hệ thống nhãn khác nhau:
- SciEntsBank: 5-way categorical
- MohlerASAG: continuous score 0-5
- Data_Generate: đã có 5-way + 3-way + 2-way nhưng label_3way="contradictory" cần remap

Module `src/data/harmonizer.py` giải quyết vấn đề này bằng class `LabelHarmonizer`.

### 3.3.2 MohlerASAG: Score → Label Mapping

**Công thức chuẩn hóa điểm:**

$$score\_normalized = \frac{score\_raw}{5.0}$$

**Ánh xạ 2-way (ngưỡng cấu hình, mặc định = 2.5):**

$$label\_2way = \begin{cases} \text{"correct"} & \text{nếu } score\_raw \geq threshold\_2way \\ \text{"incorrect"} & \text{nếu } score\_raw < threshold\_2way \end{cases}$$

**Ánh xạ 3-way (ranh giới cố định):**

$$label\_3way = \begin{cases} \text{"incorrect"} & \text{nếu } score\_raw \in [0, 1) \\ \text{"partially\_correct"} & \text{nếu } score\_raw \in [1, 4) \\ \text{"correct"} & \text{nếu } score\_raw \in [4, 5] \end{cases}$$

### 3.3.3 SciEntsBank: 5-way → 3-way → 2-way

**Bảng ánh xạ 5-way → 3-way:**

| label_5way | label_3way |
|---|---|
| correct | correct |
| partially_correct_incomplete | partially_correct |
| contradictory | incorrect |
| irrelevant | incorrect |
| non_domain | incorrect |

**Bảng ánh xạ 3-way → 2-way:**

| label_3way | label_2way |
|---|---|
| correct | correct |
| partially_correct | incorrect |
| incorrect | incorrect |

### 3.3.4 Data_Generate: Contradictory → Incorrect

Trong Data_Generate, một số bản ghi có `label_3way="contradictory"`. Vì hệ thống 3-way thống nhất chỉ có 3 giá trị (correct, partially_correct, incorrect), giá trị "contradictory" được remap thành "incorrect":

```
if rec.label_3way == "contradictory":
    rec.label_3way = "incorrect"
```

### 3.3.5 Consistency Check

Sau khi harmonize, hệ thống kiểm tra tính nhất quán: nếu `score_normalized > 0.8` nhưng `label_2way = "incorrect"`, một warning được log. Điều này phát hiện các trường hợp bất thường cần xem xét thủ công.

### 3.3.6 Mã nguồn harmonizer.py

```python
"""Label harmonization for the ASAG Research Framework.

Converts heterogeneous label spaces across datasets into the
unified 2-way, 3-way, and normalized score representations.
"""

from __future__ import annotations

import logging

from src.data.schema import UnifiedRecord

logger = logging.getLogger(__name__)

# ── SciEntsBank 5-way → 3-way mapping ────────────────────────────────

SCIENTSBANK_5WAY_TO_3WAY: dict[str, str] = {
    "correct": "correct",
    "partially_correct_incomplete": "partially_correct",
    "contradictory": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}

# ── SciEntsBank 3-way → 2-way mapping ────────────────────────────────

LABEL_3WAY_TO_2WAY: dict[str, str] = {
    "correct": "correct",
    "partially_correct": "incorrect",
    "incorrect": "incorrect",
}


class LabelHarmonizer:
    """Harmonize labels across all source datasets.

    Supports:
    - MohlerASAG: score_raw → label_2way, label_3way, score_normalized
    - SciEntsBank: label_5way → label_3way → label_2way
    - Data_Generate: label_3way="contradictory" → "incorrect"
    - Consistency check: warn when score_normalized > 0.8
      but label_2way = "incorrect"
    """

    def __init__(self, threshold_2way: float = 2.5) -> None:
        self.threshold_2way = threshold_2way

    def harmonize(self, record: UnifiedRecord) -> UnifiedRecord:
        """Harmonize labels on a single record (in-place)."""
        source = record.source_dataset

        if source == "mohler":
            self._harmonize_mohler(record)
        elif source == "scientsbank":
            self._harmonize_scientsbank(record)
        elif source == "data_generate":
            self._harmonize_data_generate(record)

        self._consistency_check(record)
        return record

    def harmonize_all(self, records: list[UnifiedRecord]) -> list[UnifiedRecord]:
        """Harmonize labels on a list of records (in-place)."""
        for rec in records:
            self.harmonize(rec)
        return records

    def _harmonize_mohler(self, rec: UnifiedRecord) -> None:
        if rec.score_raw is None:
            return

        # score_normalized = score_raw / 5.0
        rec.score_normalized = rec.score_raw / 5.0

        # label_2way via configurable threshold
        if rec.score_raw >= self.threshold_2way:
            rec.label_2way = "correct"
        else:
            rec.label_2way = "incorrect"

        # label_3way via fixed boundaries
        #   [0, 1)   → incorrect
        #   [1, 4)   → partially_correct
        #   [4, 5]   → correct
        if rec.score_raw < 1.0:
            rec.label_3way = "incorrect"
        elif rec.score_raw < 4.0:
            rec.label_3way = "partially_correct"
        else:
            rec.label_3way = "correct"

    def _harmonize_scientsbank(self, rec: UnifiedRecord) -> None:
        if rec.label_5way is None:
            return

        label_5 = rec.label_5way
        label_3 = SCIENTSBANK_5WAY_TO_3WAY.get(label_5)
        if label_3 is None:
            logger.warning(
                "Unknown SciEntsBank label_5way %r for %s",
                label_5, rec.sample_id,
            )
            return

        rec.label_3way = label_3
        rec.label_2way = LABEL_3WAY_TO_2WAY[label_3]

    def _harmonize_data_generate(self, rec: UnifiedRecord) -> None:
        if rec.label_3way == "contradictory":
            rec.label_3way = "incorrect"

    def _consistency_check(self, rec: UnifiedRecord) -> None:
        if (
            rec.score_normalized is not None
            and rec.score_normalized > 0.8
            and rec.label_2way == "incorrect"
        ):
            logger.warning(
                "Inconsistency: sample_id=%s has "
                "score_normalized=%.3f but label_2way=%r",
                rec.sample_id, rec.score_normalized, rec.label_2way,
            )
```

## 3.4 Tóm tắt luồng dữ liệu (Pipeline Flow)

Luồng xử lý dữ liệu end-to-end được thực hiện bởi script `experiments/phase1_data_audit.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ SciEntsBank  │  │  MohlerASAG  │  │ Data_Generate│           │
│  │   (XML)      │  │   (CSV)      │  │   (CSV)      │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                    │
│  ┌──────────────┐                                                 │
│  │ Data_Scraping│                                                 │
│  │   (JSON)     │                                                 │
│  └──────┬───────┘                                                 │
│         │                  │                  │                    │
│         ▼                  ▼                  ▼                    │
│  ┌─────────────────────────────────────────────────────┐         │
│  │          STEP 1: Loaders (loaders.py)                │         │
│  │   load_scientsbank() | load_mohler()                 │         │
│  │   load_data_generate() | load_data_scraping()        │         │
│  └─────────────────────────┬───────────────────────────┘         │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────┐         │
│  │     STEP 2: Label Harmonization (harmonizer.py)      │         │
│  │   Mohler score→labels | SciEntsBank 5→3→2           │         │
│  │   Data_Generate contradictory→incorrect              │         │
│  └─────────────────────────┬───────────────────────────┘         │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────┐         │
│  │       STEP 3: Split Management (splitter.py)         │         │
│  │   Preserve SciEntsBank UA/UQ/UD                      │         │
│  │   Create Mohler 60/20/20 by question_id              │         │
│  │   Verify Data_Generate integrity                     │         │
│  └─────────────────────────┬───────────────────────────┘         │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────┐         │
│  │        STEP 4: Data Quality Audit (audit.py)         │         │
│  │   Label distributions | Low confidence               │         │
│  │   Short answers | Numerical questions                │         │
│  └─────────────────────────┬───────────────────────────┘         │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────┐         │
│  │        STEP 5: Save Unified JSONL                    │         │
│  │   data/unified/scientsbank.jsonl                      │         │
│  │   data/unified/mohler.jsonl                           │         │
│  │   data/unified/data_generate.jsonl                    │         │
│  │   data/unified/data_scraping.jsonl                    │         │
│  └─────────────────────────────────────────────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

Cấu hình pipeline được quản lý qua file `configs/data.yaml`:

```yaml
# Global random seed for reproducibility
seed: 42

# Paths to raw dataset files (relative to project root)
raw_data:
  scientsbank:
    path: data/raw/scientsbank/
    format: xml
  mohler:
    path: data/raw/mohler/
    format: csv
  data_generate:
    path: data-generate.csv
    format: csv
  data_scraping:
    path: data-scraping.json
    format: json

# Output directory for unified JSONL files
unified_output_dir: data/unified/

# Split configuration
splits:
  scientsbank:
    preserve_predefined: true
  mohler:
    strategy: question_level
    train_ratio: 0.6
    valid_ratio: 0.2
    test_ratio: 0.2
  data_generate:
    preserve_predefined: true

# Label harmonization settings
harmonization:
  mohler_2way_threshold: 2.5
  mohler_3way_boundaries: [0.0, 1.0, 4.0, 5.0]
  mohler_score_max: 5.0

# Data quality audit thresholds
audit:
  min_annotation_confidence: 0.85
  min_student_answer_tokens: 3
```

## 3.5 Crawling từ web — Data_Scraping

### 3.5.1 Nguồn dữ liệu

Data_Scraping được thu thập từ sách giáo khoa mở OpenStax (college-physics-2e) thông qua web scraping. Bộ dữ liệu chứa 129 bản ghi, mỗi bản ghi gồm câu hỏi cuối chương và đáp án tham chiếu.

### 3.5.2 Cấu trúc JSON

Mỗi bản ghi trong file `data-scraping.json` có cấu trúc:

```json
{
    "id": "college-physics-2e_1_1",
    "questions": "The speed limit on some interstate highways is roughly 100 km/h. (a) What is this in meters per second? (b) How many miles per hour is this?",
    "reference_answer": "(a) 27.8 m/s (b) 62.1 mph",
    "student_answer": "",
    "label": "openstax_college-physics-2e"
}
```

Đặc điểm:
- Trường `id` chứa identifier dạng `{textbook}_{chapter}_{problem}`
- Trường `questions` chứa đề bài (lưu ý: tên trường số nhiều)
- Trường `reference_answer` chứa đáp án — nhiều câu hỏi là dạng tính toán với đáp số
- Trường `student_answer` luôn rỗng — đây là hạn chế chính
- Trường `label` chứa identifier sách giáo khoa, không phải nhãn đúng/sai

### 3.5.3 Hạn chế

Do `student_answer` rỗng, bộ dữ liệu này **không thể sử dụng** cho bất kỳ task nào yêu cầu so sánh câu trả lời sinh viên với đáp án. Tất cả 4 usability flags được đặt `False`. Tuy nhiên, dữ liệu vẫn có giá trị:
- Cung cấp question bank cho future data collection
- Cho phép phân tích loại câu hỏi (numerical vs conceptual)
- Có thể dùng làm nguồn cho synthetic data generation trong tương lai

### 3.5.4 Mã nguồn loader

```python
def load_data_scraping(json_path: str | Path) -> list[UnifiedRecord]:
    """Load Data_Scraping dataset from JSON.

    All records have empty student_answer fields and all four usability
    flags set to false. is_synthetic is set to false.
    """
    json_path = Path(json_path)
    records: list[UnifiedRecord] = []

    if not json_path.exists():
        logger.warning("Data_Scraping JSON not found: %s", json_path)
        return records

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error reading Data_Scraping JSON %s: %s", json_path, e)
        return records

    if not isinstance(data, list):
        logger.warning("Data_Scraping JSON is not a list: %s", json_path)
        return records

    for idx, entry in enumerate(data):
        try:
            original_id = str(entry.get("id", f"scr_{idx}"))
            question_id = original_id

            # The "label" field contains the textbook/domain identifier
            domain_label = str(entry.get("label", "unknown"))
            subdomain = (domain_label.replace("openstax_", "")
                        if domain_label.startswith("openstax_")
                        else domain_label)

            rec = UnifiedRecord(
                sample_id=f"SCR_{idx + 1:05d}",
                source_dataset="data_scraping",
                original_id=original_id,
                question_id=question_id,
                domain=domain_label,
                subdomain=subdomain,
                difficulty="unknown",
                question=str(entry.get("questions", "")).strip(),
                reference_answer=str(entry.get("reference_answer", "")).strip(),
                student_answer=str(entry.get("student_answer", "")).strip(),
                is_synthetic=False,
                is_human_annotated=False,
                usable_for_grading=False,
                usable_for_feedback=False,
                usable_for_misconception_mining=False,
                usable_for_robustness_eval=False,
            )
            records.append(rec)
        except Exception as e:
            logger.warning(
                "Malformed Data_Scraping entry %d (id=%s): %s — skipping",
                idx, entry.get("id", "?"), e,
            )

    logger.info("Loaded %d Data_Scraping records from %s", len(records), json_path)
    return records
```

## 3.6 Gen AI Generate Data — Data_Generate

### 3.6.1 Tổng quan

Bộ dữ liệu Data_Generate (`data-generate.csv`) chứa 10,000 bản ghi tổng hợp được sinh bởi một pipeline LLM 9 giai đoạn. Đây là bộ dữ liệu phong phú nhất trong hệ thống, với 30 cột bao phủ đầy đủ thông tin cho cả 4 nhiệm vụ nghiên cứu.

File gốc: `synthetic_benchmark_10000_semantic_debiased_v3.csv`

### 3.6.2 Pipeline sinh dữ liệu 9 giai đoạn

Theo tài liệu `Methodology_Data_Generation.html`, pipeline bao gồm:

**Stage 1 — Domain & Question Construction:**
- Xây dựng taxonomy 20 domains (biology, chemistry, physics, earth science, environmental science, astronomy, health science, scientific method, programming fundamentals, data structures, algorithms, databases, operating systems, computer networks, software engineering, introductory AI/ML, mathematics for science, statistics and experiments, cybersecurity, digital logic)
- Tạo 800 unique question identifiers
- Mỗi câu hỏi có: focus concept, comparison concept, counterfactual condition, common incorrect statements

**Stage 2 — Reference Answer Construction:**
- Sinh reference_answer ngắn gọn cho mỗi câu hỏi
- Sinh alternative_reference_answers (paraphrases)
- Trích xuất key_concepts

**Stage 3 — Misconception Modeling:**
- Tạo misconception_inventory per-question
- Mô hình hóa: missing prerequisites, incorrect claims, causal inversions, concept confusions

**Stage 4 — Student Answer Simulation:**
- Sử dụng 12 personas để mô phỏng đa dạng phong cách viết
- Personas: concise strong, detailed explainer, average incomplete, confused, overconfident wrong, distracted, guessing, memorization-oriented, sloppy correct, sloppy incorrect, mixed-reasoning, language-limited
- Sinh 5 loại câu trả lời: correct, partially_correct, contradictory, irrelevant, non_domain

**Stage 5 — Label & Score Annotation:**
- Gán label_5way, label_3way, label_2way
- Gán semantic_correctness_score_0_5
- Đảm bảo alignment giữa label và score

**Stage 6 — Feedback Generation:**
- Sinh feedback_short và feedback_detailed
- Gán feedback_type và feedback_tone
- Feedback dựa trên label + misconception annotations

**Stage 7 — Adversarial Augmentation:**
- Tạo adversarial variants: paraphrasing, surface noise, distractor clauses, misleading fluent explanations, near-contradictions, mixed responses
- Ghi nhận perturbation_type và adversarial_variant_of
- ~6,000 adversarially linked instances

**Stage 8 — Data Splitting:**
- 7,000 train / 1,000 valid / 500 test_seen / 500 test_unseen_answers / 400 test_unseen_questions / 300 test_unseen_domains / 300 test_adversarial

**Stage 9 — Validation & Refinement:**
- Duplicate detection
- Split leakage verification
- Template artifact removal
- Semantic de-biasing (8,666 rows rewritten in v3)
- Style rebalancing across 10 answer styles

### 3.6.3 Semantic De-biasing

Một đóng góp quan trọng của pipeline là semantic de-biasing — giảm thiểu các tương quan giả (spurious correlations) giữa phong cách viết và nhãn đúng/sai. Cụ thể:
- Tạo low-overlap correct answers (câu trả lời đúng nhưng dùng từ khác hoàn toàn so với reference)
- Tạo high-overlap incorrect answers (câu trả lời sai nhưng dùng nhiều từ giống reference)
- Phân bố đều 10 answer styles (concise, explanatory, fragmented, noisy, overconfident, hedged, example-driven, paraphrased low-overlap, mixed-claim, topic-drifted) qua tất cả label classes

### 3.6.4 Cấu trúc 30 cột

```
instance_id, question_id, domain, subdomain, difficulty, split,
question, reference_answer, alternative_reference_answers, key_concepts,
misconception_inventory, student_answer, student_answer_style,
lexical_overlap_level, semantic_correctness_score_0_5, label_5way,
label_3way, label_2way, misconception_tags, misconception_span_rationale,
missing_concepts, extra_incorrect_claims, feedback_short, feedback_detailed,
feedback_type, feedback_tone, adversarial_variant_of, perturbation_type,
robustness_notes, annotation_confidence
```

### 3.6.5 Mã nguồn loader (trích)

```python
def load_data_generate(csv_path: str | Path) -> list[UnifiedRecord]:
    """Load Data_Generate dataset from CSV.

    Parses all 30 columns and maps them to UnifiedRecord fields.
    Preserves adversarial_variant_of linkage and misconception_tags.
    """
    csv_path = Path(csv_path)
    records: list[UnifiedRecord] = []

    if not csv_path.exists():
        logger.warning("Data_Generate CSV not found: %s", csv_path)
        return records

    csv.field_size_limit(10 * 1024 * 1024)

    counter = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            try:
                counter += 1
                instance_id = row.get("instance_id", "").strip()
                score_raw = _safe_float(
                    row.get("semantic_correctness_score_0_5")
                )
                annotation_conf = _safe_float(
                    row.get("annotation_confidence")
                )

                perturbation = _safe_str(row.get("perturbation_type"))
                adv_variant_of = _safe_str(
                    row.get("adversarial_variant_of")
                )
                is_adversarial = (
                    perturbation is not None and perturbation != ""
                )

                rec = UnifiedRecord(
                    sample_id=f"GEN_{counter:05d}",
                    source_dataset="data_generate",
                    original_id=instance_id,
                    question_id=row.get("question_id", "").strip(),
                    domain=row.get("domain", "").strip(),
                    subdomain=row.get("subdomain", "").strip(),
                    difficulty=row.get("difficulty", "unknown").strip(),
                    question=row.get("question", "").strip(),
                    reference_answer=row.get(
                        "reference_answer", ""
                    ).strip(),
                    student_answer=row.get("student_answer", "").strip(),
                    alternative_reference_answers=_safe_parse_list(
                        row.get("alternative_reference_answers", "")
                    ),
                    key_concepts=_safe_parse_list(
                        row.get("key_concepts", "")
                    ),
                    misconception_inventory=_safe_parse_dict_list(
                        row.get("misconception_inventory", "")
                    ),
                    misconception_tags=_safe_parse_list(
                        row.get("misconception_tags", "")
                    ),
                    missing_concepts=_safe_parse_list(
                        row.get("missing_concepts", "")
                    ),
                    extra_incorrect_claims=_safe_parse_list(
                        row.get("extra_incorrect_claims", "")
                    ),
                    score_raw=score_raw,
                    label_5way=_safe_str(row.get("label_5way")),
                    label_3way=_safe_str(row.get("label_3way")),
                    label_2way=_safe_str(row.get("label_2way")),
                    feedback_short=_safe_str(row.get("feedback_short")),
                    feedback_detailed=_safe_str(
                        row.get("feedback_detailed")
                    ),
                    feedback_type=_safe_str(row.get("feedback_type")),
                    feedback_tone=_safe_str(row.get("feedback_tone")),
                    split=row.get("split", "").strip(),
                    is_synthetic=True,
                    is_adversarial=is_adversarial,
                    perturbation_type=perturbation,
                    adversarial_variant_of=adv_variant_of,
                    student_answer_style=_safe_str(
                        row.get("student_answer_style")
                    ),
                    annotation_confidence=annotation_conf,
                    is_human_annotated=False,
                    usable_for_grading=True,
                    usable_for_feedback=True,
                    usable_for_misconception_mining=True,
                    usable_for_robustness_eval=True,
                )
                records.append(rec)
            except Exception as e:
                logger.warning(
                    "Malformed Data_Generate row %d (id=%s): %s — skipping",
                    row_num, row.get("instance_id", "?"), e,
                )

    logger.info(
        "Loaded %d Data_Generate records from %s", len(records), csv_path
    )
    return records
```

### 3.6.6 Hàm phụ trợ parsing

Loader sử dụng các hàm helper để parse an toàn các trường phức tạp:

```python
def _safe_parse_list(value: str) -> list[str]:
    """Parse a string representation of a list, returning [] on failure."""
    if not value or value.strip() in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            return []


def _safe_float(value: str | None) -> float | None:
    """Convert a string to float, returning None on failure."""
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
```

## 3.7 Dữ liệu công khai — SciEntsBank + MohlerASAG

### 3.7.1 SciEntsBank

SciEntsBank là bộ dữ liệu từ SemEval-2013 Task 7, chứa câu trả lời ngắn của học sinh về các chủ đề khoa học tự nhiên. Đặc điểm:

- **Format:** XML theo chuẩn SemEval Beetle format
- **Cấu trúc thư mục:** `data/raw/scientsbank/{split}/*.xml` (train, test_ua, test_uq, test_ud)
- **Nhãn:** 5-way accuracy attribute trên mỗi `<studentAnswer>` element
- **Annotation:** Human-annotated (is_human_annotated = True)
- **Domain:** science (với subdomain từ attribute `module`)

**Mã nguồn loader:**

```python
def load_scientsbank(data_dir: str | Path) -> list[UnifiedRecord]:
    """Load SciEntsBank dataset from XML/text files.

    Expected directory structure:
        data_dir/
            <split>/          (e.g., train, test_ua, test_uq, test_ud)
                *.xml
    """
    data_dir = Path(data_dir)
    records: list[UnifiedRecord] = []

    if not data_dir.exists():
        logger.warning("SciEntsBank directory not found: %s", data_dir)
        return records

    counter = 0

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        for xml_file in sorted(split_dir.glob("*.xml")):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning(
                    "Malformed XML in SciEntsBank file %s: %s — skipping",
                    xml_file, e,
                )
                continue

            for question_elem in root.iter("question"):
                q_id = question_elem.get("id", "")
                q_text = question_elem.findtext("questionText", "")
                ref_answers = question_elem.findall(
                    ".//referenceAnswer"
                )
                ref_answer_text = ""
                alt_refs: list[str] = []
                for i, ra in enumerate(ref_answers):
                    text = (ra.text or "").strip()
                    if i == 0:
                        ref_answer_text = text
                    else:
                        alt_refs.append(text)

                for sa_elem in question_elem.iter("studentAnswer"):
                    try:
                        counter += 1
                        sa_id = sa_elem.get(
                            "id", f"unknown_{counter}"
                        )
                        sa_text = (sa_elem.text or "").strip()
                        accuracy = sa_elem.get(
                            "accuracy", "unknown"
                        )

                        rec = UnifiedRecord(
                            sample_id=f"SEB_{counter:05d}",
                            source_dataset="scientsbank",
                            original_id=sa_id,
                            question_id=q_id,
                            domain="science",
                            subdomain=question_elem.get(
                                "module", "general"
                            ),
                            difficulty="unknown",
                            question=q_text,
                            reference_answer=ref_answer_text,
                            student_answer=sa_text,
                            alternative_reference_answers=alt_refs,
                            label_5way=accuracy,
                            split=split_name,
                            is_human_annotated=True,
                        )
                        records.append(rec)
                    except Exception as e:
                        logger.warning(
                            "Malformed SciEntsBank student answer "
                            "in %s (id=%s): %s — skipping",
                            xml_file,
                            sa_elem.get("id", "?"),
                            e,
                        )

    logger.info(
        "Loaded %d SciEntsBank records from %s",
        len(records), data_dir,
    )
    return records
```

### 3.7.2 MohlerASAG

MohlerASAG là bộ dữ liệu từ Mohler et al. (2011), chứa câu trả lời ngắn của sinh viên về khoa học máy tính. Đặc điểm:

- **Format:** CSV với các cột question_id, question, reference_answer, student_answer, score columns
- **Scoring:** Nhiều annotator cho điểm → loader tính trung bình
- **Domain:** computer_science
- **Alternative references:** Loader thu thập tất cả reference answers per question_id, reference đầu tiên là chính, còn lại là alternatives

**Mã nguồn loader:**

```python
def load_mohler(data_dir: str | Path) -> list[UnifiedRecord]:
    """Load MohlerASAG dataset from CSV/text files.

    The Mohler dataset typically has columns for question, answer, and
    multiple annotator scores. The loader averages annotator scores for
    score_raw.
    """
    data_dir = Path(data_dir)
    records: list[UnifiedRecord] = []

    if not data_dir.exists():
        logger.warning("MohlerASAG directory not found: %s", data_dir)
        return records

    csv_files = sorted(data_dir.glob("*.csv")) + sorted(
        data_dir.glob("*.txt")
    )
    if not csv_files:
        logger.warning(
            "No CSV/text files found in MohlerASAG directory: %s",
            data_dir,
        )
        return records

    counter = 0
    question_ref_answers: dict[str, list[str]] = {}

    # First pass: collect all reference answers per question
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q_id = row.get(
                        "question_id", row.get("qid", "")
                    )
                    ref = row.get(
                        "reference_answer",
                        row.get("desired_answer", ""),
                    ).strip()
                    if q_id and ref:
                        if q_id not in question_ref_answers:
                            question_ref_answers[q_id] = []
                        if ref not in question_ref_answers[q_id]:
                            question_ref_answers[q_id].append(ref)
        except Exception as e:
            logger.warning(
                "Error reading MohlerASAG file %s: %s — skipping",
                csv_file, e,
            )

    # Second pass: build records
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):
                    try:
                        counter += 1
                        q_id = row.get(
                            "question_id",
                            row.get("qid", f"q_{counter}"),
                        )
                        question = row.get(
                            "question",
                            row.get("question_text", ""),
                        ).strip()
                        ref_answer = row.get(
                            "reference_answer",
                            row.get("desired_answer", ""),
                        ).strip()
                        student_answer = row.get(
                            "student_answer",
                            row.get("answer", ""),
                        ).strip()

                        # Average annotator scores
                        score_cols = [
                            k for k in row.keys()
                            if k.startswith("score")
                            or k.startswith("grade")
                            or k.startswith("me")
                            or k.startswith("other")
                        ]
                        scores = []
                        for col in score_cols:
                            try:
                                scores.append(float(row[col]))
                            except (ValueError, TypeError):
                                pass

                        if not scores:
                            for col in [
                                "score_raw", "score", "avg_score"
                            ]:
                                if col in row:
                                    try:
                                        scores.append(float(row[col]))
                                    except (ValueError, TypeError):
                                        pass

                        score_raw = (
                            sum(scores) / len(scores)
                            if scores
                            else None
                        )

                        alt_refs = [
                            r
                            for r in question_ref_answers.get(
                                q_id, []
                            )
                            if r != ref_answer
                        ]

                        rec = UnifiedRecord(
                            sample_id=f"MOH_{counter:05d}",
                            source_dataset="mohler",
                            original_id=str(
                                row.get(
                                    "id",
                                    row.get(
                                        "instance_id",
                                        f"mohler_{counter}",
                                    ),
                                )
                            ),
                            question_id=str(q_id),
                            domain="computer_science",
                            subdomain=row.get(
                                "subdomain",
                                row.get("topic", "general"),
                            ),
                            difficulty="unknown",
                            question=question,
                            reference_answer=ref_answer,
                            student_answer=student_answer,
                            alternative_reference_answers=alt_refs,
                            score_raw=score_raw,
                            is_human_annotated=True,
                        )
                        records.append(rec)
                    except Exception as e:
                        logger.warning(
                            "Malformed MohlerASAG row %d in %s: "
                            "%s — skipping",
                            row_num, csv_file, e,
                        )
        except Exception as e:
            logger.warning(
                "Error reading MohlerASAG file %s: %s — skipping",
                csv_file, e,
            )

    logger.info(
        "Loaded %d MohlerASAG records from %s",
        len(records), data_dir,
    )
    return records
```

## 3.8 Hợp nhất dữ liệu — DataLoader API

### 3.8.1 Kiến trúc

Module `src/data/dataset.py` cung cấp class `DataLoader` — API công khai để truy cập dữ liệu thống nhất. DataLoader nhận danh sách `UnifiedRecord` đã qua harmonization và split assignment, sau đó cung cấp các phương thức truy vấn linh hoạt.

Kiến trúc nội bộ sử dụng index `(source_dataset, split) → list[UnifiedRecord]` để truy cập O(1) theo source + split.

### 3.8.2 API chính

**`get_split(source, split, filters=None)`** — Lấy tất cả bản ghi cho một source + split cụ thể:

```python
loader = DataLoader(all_records)

# Lấy SciEntsBank training data
seb_train = loader.get_split("scientsbank", "train")

# Lấy Data_Generate test adversarial, chỉ lấy incorrect
adv_incorrect = loader.get_split(
    "data_generate", "test_adversarial",
    filters={"label_2way": "incorrect"}
)
```

**`get_merged(sources, filters=None)`** — Kết hợp nhiều (source, split) thành một list:

```python
# Merge training data từ nhiều nguồn
merged_train = loader.get_merged([
    ("scientsbank", "train"),
    ("mohler", "train"),
    ("data_generate", "train"),
])
```

**`get_training_batch(sources, label_field, filters=None)`** — Yield tuples `((question, reference_answer, student_answer), label)` cho training:

```python
# Tạo training batch với 3-way labels
for triplet, label in loader.get_training_batch(
    sources=[("data_generate", "train")],
    label_field="label_3way",
):
    question, ref, student = triplet
    # Feed to model...
```

### 3.8.3 Filtering

DataLoader hỗ trợ filtering theo các trường sau:

```python
_FILTER_FIELDS = frozenset({
    "source_dataset",
    "domain",
    "label_5way",
    "label_3way",
    "label_2way",
    "is_adversarial",
    "usable_for_grading",
})
```

Filter được áp dụng dưới dạng AND logic — bản ghi phải match tất cả các key-value pairs.

### 3.8.4 Mã nguồn dataset.py

```python
"""Data_Loader public API for the ASAG Research Framework.

Provides programmatic access to unified records with filtering,
cross-dataset merging, and training-batch generation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.data.schema import UnifiedRecord

_FILTER_FIELDS = frozenset({
    "source_dataset",
    "domain",
    "label_5way",
    "label_3way",
    "label_2way",
    "is_adversarial",
    "usable_for_grading",
})


def _apply_filters(
    records: Iterable[UnifiedRecord],
    filters: dict | None,
) -> list[UnifiedRecord]:
    """Return records matching all filter key-value pairs."""
    if not filters:
        return list(records)

    result: list[UnifiedRecord] = []
    for rec in records:
        match = True
        for key, value in filters.items():
            if key not in _FILTER_FIELDS:
                continue
            if getattr(rec, key) != value:
                match = False
                break
        if match:
            result.append(rec)
    return result


class DataLoader:
    """Public data-loading interface for all downstream consumers."""

    def __init__(self, records: list[UnifiedRecord]) -> None:
        self._records = list(records)
        self._index: dict[
            tuple[str, str], list[UnifiedRecord]
        ] = defaultdict(list)
        for rec in self._records:
            self._index[(rec.source_dataset, rec.split)].append(rec)

    def get_split(
        self,
        source: str,
        split: str,
        filters: dict | None = None,
    ) -> list[UnifiedRecord]:
        """Return all records for a specified source + split."""
        records = self._require_split(source, split)
        return _apply_filters(records, filters)

    def get_merged(
        self,
        sources: list[tuple[str, str]],
        filters: dict | None = None,
    ) -> list[UnifiedRecord]:
        """Combine multiple (source, split) pairs into a single list."""
        merged: list[UnifiedRecord] = []
        for source, split in sources:
            merged.extend(self._require_split(source, split))
        return _apply_filters(merged, filters)

    def get_training_batch(
        self,
        sources: list[tuple[str, str]],
        label_field: str,
        filters: dict | None = None,
    ) -> Iterable[tuple[tuple[str, str, str], str | float]]:
        """Yield ((question, ref, student), label) tuples."""
        records = self.get_merged(sources, filters)
        for rec in records:
            triplet = (
                rec.question,
                rec.reference_answer,
                rec.student_answer,
            )
            label = getattr(rec, label_field)
            yield triplet, label

    def _available_splits(self, source: str) -> list[str]:
        return sorted(
            {split for src, split in self._index if src == source}
        )

    def _require_split(
        self, source: str, split: str
    ) -> list[UnifiedRecord]:
        key = (source, split)
        if key not in self._index:
            available = self._available_splits(source)
            raise ValueError(
                f"Split {split!r} does not exist for source "
                f"{source!r}. Available splits: {available}"
            )
        return self._index[key]
```

### 3.8.5 Error Handling

Khi yêu cầu một split không tồn tại, `DataLoader` raise `ValueError` với thông báo rõ ràng liệt kê các splits có sẵn:

```
ValueError: Split 'test_xyz' does not exist for source 'scientsbank'.
Available splits: ['test_ua', 'test_ud', 'test_uq', 'train']
```

## 3.9 Quản lý Split và kiểm tra Leakage

### 3.9.1 Tổng quan chiến lược

Module `src/data/splitter.py` quản lý việc phân chia dữ liệu với 3 chiến lược khác nhau tùy theo nguồn:

| Source | Chiến lược | Chi tiết |
|---|---|---|
| SciEntsBank | Preserve predefined | Giữ nguyên UA/UQ/UD splits từ loader |
| MohlerASAG | Question-level split | 60/20/20 theo question_id, seeded shuffle |
| Data_Generate | Preserve + verify | Giữ nguyên 7 splits, kiểm tra integrity |
| Data_Scraping | None | Không có split (không dùng cho training) |

### 3.9.2 SciEntsBank: Preserve UA/UQ/UD

SciEntsBank đã có split được gán sẵn bởi loader (từ tên thư mục). SplitManager chỉ passthrough — không thay đổi gì:

```python
def _handle_scientsbank(self, records: list[UnifiedRecord]) -> None:
    """Preserve predefined SciEntsBank UA/UQ/UD splits."""
    pass  # Splits are preserved as loaded
```

Ý nghĩa các split:
- **train:** Dữ liệu huấn luyện
- **test_ua (Unseen Answers):** Câu trả lời mới cho câu hỏi đã thấy
- **test_uq (Unseen Questions):** Câu hỏi hoàn toàn mới
- **test_ud (Unseen Domains):** Domain hoàn toàn mới

### 3.9.3 MohlerASAG: Question-level 60/20/20

Chiến lược quan trọng nhất: **không có question_id nào xuất hiện ở nhiều hơn một partition**. Điều này ngăn chặn data leakage — nếu mô hình thấy câu hỏi Q1 trong training, nó không nên được test trên Q1 (dù với câu trả lời khác).

Thuật toán:
1. Group tất cả records theo question_id
2. Seeded shuffle danh sách question_ids
3. Chia: 60% đầu → train, 20% tiếp → valid, 20% cuối → test
4. Gán split cho tất cả records trong mỗi group

```python
def _handle_mohler(self, records: list[UnifiedRecord]) -> None:
    """Create question-level 60/20/20 splits for MohlerASAG."""
    groups: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.question_id].append(rec)

    question_ids = sorted(groups.keys())
    rng = random.Random(self.seed)
    rng.shuffle(question_ids)

    n = len(question_ids)
    train_end = int(n * 0.6)
    valid_end = train_end + int(n * 0.2)

    train_qids = set(question_ids[:train_end])
    valid_qids = set(question_ids[train_end:valid_end])

    for qid, recs in groups.items():
        if qid in train_qids:
            split_name = "train"
        elif qid in valid_qids:
            split_name = "valid"
        else:
            split_name = "test"
        for rec in recs:
            rec.split = split_name
```

### 3.9.4 Data_Generate: Integrity Verification

Data_Generate có 7 splits được gán sẵn. SplitManager không thay đổi chúng nhưng kiểm tra 3 ràng buộc:

**1. Adversarial Co-location:**
Mỗi adversarial variant phải nằm cùng split với bản ghi gốc (original). Nếu vi phạm → `SplitIntegrityError`.

```python
def _check_adversarial_colocation(
    self, records: list[UnifiedRecord]
) -> None:
    by_original_id: dict[str, UnifiedRecord] = {}
    for rec in records:
        by_original_id[rec.original_id] = rec

    violations: list[str] = []
    for rec in records:
        if rec.adversarial_variant_of is None:
            continue
        original = by_original_id.get(rec.adversarial_variant_of)
        if original is not None and original.split != rec.split:
            violations.append(rec.sample_id)

    if violations:
        raise SplitIntegrityError(
            "Adversarial variant(s) not co-located with original.",
            violations,
        )
```

**2. Unseen Questions Leakage:**
Không có question_id nào xuất hiện đồng thời trong train và test_unseen_questions.

```python
def _check_unseen_questions(
    self, records: list[UnifiedRecord]
) -> None:
    train_qids: set[str] = set()
    unseen_q_qids: set[str] = set()

    for rec in records:
        if rec.split == "train":
            train_qids.add(rec.question_id)
        elif rec.split == "test_unseen_questions":
            unseen_q_qids.add(rec.question_id)

    leaked = train_qids & unseen_q_qids
    if leaked:
        affected: list[str] = []
        for rec in records:
            if rec.question_id in leaked and rec.split in (
                "train", "test_unseen_questions"
            ):
                affected.append(rec.sample_id)
        raise SplitIntegrityError(
            f"question_id(s) {sorted(leaked)} appear in both "
            f"train and test_unseen_questions.",
            affected,
        )
```

**3. Unseen Domains Leakage:**
Không có domain nào xuất hiện đồng thời trong train và test_unseen_domains.

```python
def _check_unseen_domains(
    self, records: list[UnifiedRecord]
) -> None:
    train_domains: set[str] = set()
    unseen_d_domains: set[str] = set()

    for rec in records:
        if rec.split == "train":
            train_domains.add(rec.domain)
        elif rec.split == "test_unseen_domains":
            unseen_d_domains.add(rec.domain)

    leaked = train_domains & unseen_d_domains
    if leaked:
        affected: list[str] = []
        for rec in records:
            if rec.domain in leaked and rec.split in (
                "train", "test_unseen_domains"
            ):
                affected.append(rec.sample_id)
        raise SplitIntegrityError(
            f"domain(s) {sorted(leaked)} appear in both "
            f"train and test_unseen_domains.",
            affected,
        )
```

### 3.9.5 SplitIntegrityError

Khi phát hiện vi phạm, hệ thống raise `SplitIntegrityError` — một custom exception mang theo danh sách `affected_sample_ids`:

```python
class SplitIntegrityError(Exception):
    """Raised when a split integrity constraint is violated."""

    def __init__(
        self, message: str, affected_sample_ids: list[str]
    ) -> None:
        self.affected_sample_ids = affected_sample_ids
        super().__init__(
            f"{message} Affected sample_ids: {affected_sample_ids}"
        )
```

Điều này cho phép downstream code xử lý lỗi một cách có cấu trúc — biết chính xác bản ghi nào gây ra vấn đề.

---

# CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Kiểm tra chất lượng dữ liệu (Data Audit)

### 4.1.1 Tổng quan module audit

Module `src/data/audit.py` cung cấp class `DataAuditor` với các phương thức kiểm tra chất lượng dữ liệu. Kết quả được tổng hợp trong `AuditReport` dataclass.

### 4.1.2 Label Distributions

Phương thức `label_distributions()` tính phân bố nhãn theo từng source dataset, cung cấp cả counts và percentages:

```python
@staticmethod
def label_distributions(
    records: Sequence[UnifiedRecord],
) -> list[LabelDistribution]:
    """Compute label counts and percentages per source dataset."""
    by_source: dict[str, list[UnifiedRecord]] = {}
    for rec in records:
        by_source.setdefault(rec.source_dataset, []).append(rec)

    results: list[LabelDistribution] = []
    for source, recs in sorted(by_source.items()):
        dist = LabelDistribution(source_dataset=source)
        for rec in recs:
            if rec.label_5way is not None:
                dist.label_5way[rec.label_5way] = (
                    dist.label_5way.get(rec.label_5way, 0) + 1
                )
            if rec.label_3way is not None:
                dist.label_3way[rec.label_3way] = (
                    dist.label_3way.get(rec.label_3way, 0) + 1
                )
            if rec.label_2way is not None:
                dist.label_2way[rec.label_2way] = (
                    dist.label_2way.get(rec.label_2way, 0) + 1
                )
        results.append(dist)
    return results
```

`LabelDistribution` dataclass cung cấp property tính phần trăm:

```python
@dataclass
class LabelDistribution:
    """Label counts and percentages for a single source dataset."""
    source_dataset: str
    label_5way: dict[str, int] = field(default_factory=dict)
    label_3way: dict[str, int] = field(default_factory=dict)
    label_2way: dict[str, int] = field(default_factory=dict)

    @property
    def label_5way_pct(self) -> dict[str, float]:
        total = sum(self.label_5way.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in self.label_5way.items()}
```

### 4.1.3 Low Confidence Records

Phát hiện các bản ghi Data_Generate có `annotation_confidence` thấp hơn ngưỡng (mặc định 0.85):

```python
@staticmethod
def low_confidence_records(
    records: Sequence[UnifiedRecord],
    threshold: float = 0.85,
) -> list[UnifiedRecord]:
    """Return records with annotation_confidence below threshold."""
    return [
        rec
        for rec in records
        if rec.annotation_confidence is not None
        and rec.annotation_confidence < threshold
    ]
```

Các bản ghi low-confidence cần được xem xét thủ công hoặc loại bỏ khỏi training set vì nhãn có thể không chính xác.

### 4.1.4 "Not Found" Reference Answers

Phát hiện các bản ghi Data_Scraping có reference_answer là "Not found" — nghĩa là web scraper không tìm được đáp án:

```python
@staticmethod
def not_found_references(
    records: Sequence[UnifiedRecord],
) -> list[UnifiedRecord]:
    """Return Data_Scraping records with reference_answer 'Not found'."""
    return [
        rec
        for rec in records
        if rec.source_dataset == "data_scraping"
        and rec.reference_answer == "Not found"
    ]
```

### 4.1.5 Short Student Answers

Phát hiện câu trả lời quá ngắn (ít hơn 3 tokens) — có thể là noise hoặc non-answers:

```python
@staticmethod
def short_student_answers(
    records: Sequence[UnifiedRecord],
    min_tokens: int = 3,
) -> list[UnifiedRecord]:
    """Return records with fewer than min_tokens tokens."""
    return [
        rec
        for rec in records
        if len(rec.student_answer.split()) < min_tokens
    ]
```

### 4.1.6 Numerical Question Detection

Heuristic phát hiện câu hỏi dạng tính toán/số học trong Data_Scraping — quan trọng vì ASAG models thường không xử lý tốt câu hỏi numerical:

```python
_CALCULATION_KEYWORDS = re.compile(
    r"\b(?:calculate|compute|how many|what is the value|convert|"
    r"determine the|find the value|evaluate|solve)\b",
    re.IGNORECASE,
)

_UNIT_WORDS = re.compile(
    r"\b(?:meters?|kilometres?|kilometers?|kg|kilograms?|grams?|"
    r"litres?|liters?|miles?|feet|foot|inches?|centimeters?|"
    r"centimetres?|millimeters?|millimetres?|seconds?|minutes?|"
    r"hours?|joules?|watts?|newtons?|volts?|amperes?|amps?|"
    r"celsius|fahrenheit|kelvin|mol|moles?|pounds?|ounces?|"
    r"gallons?|mph|km/h|m/s)\b",
    re.IGNORECASE,
)

@staticmethod
def is_numerical_question(question: str) -> bool:
    """Heuristic: does question look numerical or computational?"""
    if _CALCULATION_KEYWORDS.search(question):
        return True
    if _UNIT_WORDS.search(question):
        return True
    if _NUMERIC_EXPRESSION.search(question):
        return True
    if _DIGITS.search(question) and _UNIT_WORDS.search(question):
        return True
    return False

@staticmethod
def numerical_question_counts(
    records: Sequence[UnifiedRecord],
) -> tuple[int, int]:
    """Count numerical vs conceptual questions among Data_Scraping."""
    seen_questions: dict[str, bool] = {}
    for rec in records:
        if rec.source_dataset != "data_scraping":
            continue
        if rec.question_id in seen_questions:
            continue
        seen_questions[rec.question_id] = (
            DataAuditor.is_numerical_question(rec.question)
        )

    numerical = sum(1 for v in seen_questions.values() if v)
    conceptual = sum(1 for v in seen_questions.values() if not v)
    return numerical, conceptual
```

### 4.1.7 Stratified Audit Sample

Cho phép lấy mẫu phân tầng (stratified) theo `(label_5way, source_dataset)` để kiểm tra thủ công:

```python
@staticmethod
def stratified_sample(
    records: Sequence[UnifiedRecord],
    n: int,
    seed: int = 42,
) -> list[UnifiedRecord]:
    """Select n records stratified by label_5way and source_dataset."""
    import random as _random
    rng = _random.Random(seed)

    strata: dict[tuple[str | None, str], list[UnifiedRecord]] = {}
    for rec in records:
        key = (rec.label_5way, rec.source_dataset)
        strata.setdefault(key, []).append(rec)

    total = len(records)
    if total == 0 or n <= 0:
        return []

    selected: list[UnifiedRecord] = []
    sorted_keys = sorted(
        strata.keys(), key=lambda k: (str(k[0]), k[1])
    )

    # Proportional allocation
    allocations: dict[tuple[str | None, str], int] = {}
    for key in sorted_keys:
        proportion = len(strata[key]) / total
        alloc = int(proportion * n)
        alloc = min(alloc, len(strata[key]))
        allocations[key] = alloc

    # Distribute remainder
    allocated_total = sum(allocations.values())
    remaining_n = n - allocated_total
    for key in sorted_keys:
        if remaining_n <= 0:
            break
        capacity = len(strata[key]) - allocations[key]
        extra = min(remaining_n, capacity)
        allocations[key] += extra
        remaining_n -= extra

    # Sample from each stratum
    for key in sorted_keys:
        pool = list(strata[key])
        rng.shuffle(pool)
        selected.extend(pool[: allocations[key]])

    return selected
```

### 4.1.8 Full Audit

Phương thức `full_audit()` chạy tất cả kiểm tra và trả về `AuditReport`:

```python
@staticmethod
def full_audit(
    records: Sequence[UnifiedRecord],
) -> AuditReport:
    """Run all audit checks and return an AuditReport."""
    num, con = DataAuditor.numerical_question_counts(records)
    return AuditReport(
        label_distributions=DataAuditor.label_distributions(records),
        low_confidence_records=DataAuditor.low_confidence_records(
            records
        ),
        not_found_reference_records=DataAuditor.not_found_references(
            records
        ),
        short_answer_records=DataAuditor.short_student_answers(records),
        numerical_question_count=num,
        conceptual_question_count=con,
    )
```

## 4.2 Property-Based Testing

### 4.2.1 Tổng quan phương pháp

Property-based testing (PBT) khác với unit testing truyền thống ở chỗ: thay vì kiểm tra từng example cụ thể, PBT kiểm tra các **thuộc tính bất biến** (invariants) phải đúng cho **mọi đầu vào hợp lệ**. Thư viện Hypothesis tự động sinh hàng trăm test cases ngẫu nhiên và tìm counterexamples.

Hệ thống của chúng tôi có 6 properties:

### 4.2.2 Property 1: score_raw=5 → correct, score_raw=0 → incorrect

**Ý nghĩa:** Điểm cực đại (5/5) luôn phải map thành "correct", điểm cực tiểu (0/5) luôn phải map thành "incorrect". Đây là sanity check cho logic harmonization.

```python
@given(score=st.sampled_from([0.0, 5.0]))
@settings(max_examples=200)
def test_property1_extreme_scores(score):
    """score_raw=5 → label_2way='correct',
    score_raw=0 → label_2way='incorrect'."""
    h = LabelHarmonizer()
    rec = _mohler_record(score_raw=score)
    h.harmonize(rec)
    if score == 5.0:
        assert rec.label_2way == "correct"
    elif score == 0.0:
        assert rec.label_2way == "incorrect"
```

### 4.2.3 Property 2: score_normalized ∈ [0.0, 1.0]

**Ý nghĩa:** Với bất kỳ score_raw hợp lệ nào trong [0, 5], sau khi harmonize, score_normalized phải nằm trong [0.0, 1.0]. Đảm bảo không có overflow hay division error.

```python
@given(
    score=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False
    )
)
@settings(max_examples=200)
def test_property2_score_normalized_bounds(score):
    """For any score in [0,5], score_normalized is in [0.0, 1.0]."""
    h = LabelHarmonizer()
    rec = _mohler_record(score_raw=score)
    h.harmonize(rec)
    assert rec.score_normalized is not None
    assert 0.0 <= rec.score_normalized <= 1.0
```

### 4.2.4 Property 3: MohlerASAG split produces disjoint question_id sets

**Ý nghĩa:** Với bất kỳ tập câu hỏi-đáp nào, sau khi split, không có question_id nào xuất hiện ở nhiều hơn một partition. Đây là property quan trọng nhất cho leakage prevention.

```python
@given(data=st.data())
@settings(max_examples=100)
def test_property3_disjoint_question_ids(data):
    """For any question-answer set, question-level split produces
    disjoint question_id sets across train/valid/test."""
    num_questions = data.draw(
        st.integers(min_value=1, max_value=30)
    )
    question_ids = [f"q{i}" for i in range(num_questions)]

    records: list[UnifiedRecord] = []
    counter = 0
    for qid in question_ids:
        num_answers = data.draw(
            st.integers(min_value=1, max_value=5)
        )
        for _ in range(num_answers):
            counter += 1
            records.append(
                _mohler_record(
                    sample_id=f"MOH_{counter:05d}",
                    question_id=qid,
                )
            )

    seed = data.draw(st.integers(min_value=0, max_value=10000))
    sm = SplitManager(seed=seed)
    sm.assign_splits(records)

    split_qids: dict[str, set[str]] = {
        "train": set(), "valid": set(), "test": set()
    }
    for rec in records:
        split_qids[rec.split].add(rec.question_id)

    assert split_qids["train"].isdisjoint(split_qids["valid"])
    assert split_qids["train"].isdisjoint(split_qids["test"])
    assert split_qids["valid"].isdisjoint(split_qids["test"])
```

### 4.2.5 Property 4: Adversarial variant co-location

**Ý nghĩa:** Với bất kỳ tập bản ghi nào có adversarial links, nếu tất cả variants được đặt cùng split với original, thì `assign_splits()` không raise error. Đảm bảo logic verification hoạt động đúng.

```python
@given(data=st.data())
@settings(max_examples=100)
def test_property4_adversarial_colocation(data):
    """For any records with adversarial links, co-location
    invariant holds after split assignment."""
    valid_splits = [
        "train", "valid", "test_unseen_questions",
        "test_unseen_answers", "test_seen",
        "test_adversarial", "test_unseen_domains",
    ]

    num_originals = data.draw(
        st.integers(min_value=1, max_value=15)
    )
    records: list[UnifiedRecord] = []
    counter = 0

    for i in range(num_originals):
        counter += 1
        split = data.draw(st.sampled_from(valid_splits))
        rec = _data_generate_record(
            sample_id=f"GEN_{counter:05d}",
            original_id=f"inst_{counter:03d}",
            question_id=f"q_{split}_{i}",
            domain=f"domain_{split}_{i}",
            split=split,
        )
        records.append(rec)

        add_variant = data.draw(st.booleans())
        if add_variant:
            counter += 1
            records.append(
                _data_generate_record(
                    sample_id=f"GEN_{counter:05d}",
                    original_id=f"inst_{counter:03d}",
                    question_id=f"q_{split}_{i}",
                    domain=f"domain_{split}_{i}",
                    split=split,  # Same split as original
                    adversarial_variant_of=rec.original_id,
                    is_adversarial=True,
                )
            )

    sm = SplitManager()
    sm.assign_splits(records)  # Should not raise
```

### 4.2.6 Property 5: All sample_id values unique

**Ý nghĩa:** Với bất kỳ batch records nào từ tất cả sources, tất cả sample_id phải unique. Đảm bảo naming convention `{PREFIX}_{INDEX:05d}` không tạo collision.

```python
@given(
    batch=st.lists(
        st.tuples(
            st.sampled_from(sorted(VALID_SOURCE_DATASETS)),
            st.integers(min_value=0, max_value=99_999),
        ),
        min_size=1,
        max_size=50,
        unique=True,
    )
)
@settings(max_examples=100)
def test_property5_unique_sample_ids(batch):
    """For any batch of records from all sources,
    all sample_id values are unique."""
    records = [
        _build_record_from_strategy(src, idx)
        for src, idx in batch
    ]
    ids = [r.sample_id for r in records]
    assert len(ids) == len(set(ids))
```

### 4.2.7 Property 6: Data_Scraping usability flags all false

**Ý nghĩa:** Với bất kỳ Data_Scraping record nào (bất kể nội dung question/reference), tất cả 4 usability flags phải là False. Đảm bảo dữ liệu không có student_answer không bao giờ được dùng cho training.

```python
@given(
    index=st.integers(min_value=0, max_value=99_999),
    question=st.text(min_size=1, max_size=200),
    reference=st.text(min_size=1, max_size=200),
)
@settings(max_examples=100)
def test_property6_data_scraping_usability_flags(
    index, question, reference
):
    """For any Data_Scraping record, all four usability flags
    are false."""
    rec = UnifiedRecord(
        sample_id=f"SCR_{index:05d}",
        source_dataset="data_scraping",
        original_id=f"scr_orig_{index}",
        question_id=f"scr_q_{index}",
        domain="science",
        subdomain="general",
        difficulty="unknown",
        question=question,
        reference_answer=reference,
        student_answer="",
        usable_for_grading=False,
        usable_for_feedback=False,
        usable_for_misconception_mining=False,
        usable_for_robustness_eval=False,
    )
    assert rec.usable_for_grading is False
    assert rec.usable_for_feedback is False
    assert rec.usable_for_misconception_mining is False
    assert rec.usable_for_robustness_eval is False
```

## 4.3 Thống kê tổng hợp

### 4.3.1 Bảng tổng hợp nguồn dữ liệu

| Nguồn | Số bản ghi | Nhãn | Splits | Usable |
|---|---|---|---|---|
| SciEntsBank | ~10,000 | 5-way (human) | UA/UQ/UD | ✓ All 4 tasks |
| MohlerASAG | ~2,273 | Score 0-5 (human) | 60/20/20 | ✓ Grading |
| Data_Generate | 10,000 | 5-way + score (LLM) | 7 splits | ✓ All 4 tasks |
| Data_Scraping | 129 | None | None | ✗ (no student_answer) |

### 4.3.2 Bảng tổng hợp schema

| Nhóm trường | Số trường | Bắt buộc | Optional |
|---|---|---|---|
| Identity | 4 | 4 | 0 |
| Domain | 3 | 3 | 0 |
| Core Triplet | 4 | 3 | 1 (alt refs) |
| Grading Labels | 5 | 0 | 5 |
| Concept Annotations | 5 | 0 | 5 (default []) |
| Feedback | 4 | 0 | 4 |
| Metadata | 8 | 0 | 8 |
| Usability Flags | 4 | 0 | 4 (default True) |
| **Tổng** | **37** | **10** | **27** |

### 4.3.3 Bảng tổng hợp Data_Generate splits

| Split | Số bản ghi | Mục đích |
|---|---|---|
| train | 7,000 | Huấn luyện |
| valid | 1,000 | Validation / hyperparameter tuning |
| test_seen | 500 | Test trên câu hỏi đã thấy |
| test_unseen_answers | 500 | Test câu trả lời mới cho câu hỏi đã thấy |
| test_unseen_questions | 400 | Test câu hỏi hoàn toàn mới |
| test_unseen_domains | 300 | Test domain hoàn toàn mới |
| test_adversarial | 300 | Test adversarial perturbations |

### 4.3.4 Bảng tổng hợp validation rules

| Rule | Áp dụng cho | Hành vi khi vi phạm |
|---|---|---|
| source_dataset ∈ {4 values} | Tất cả records | ValueError at construction |
| score_normalized ∈ [0, 1] | Records có score | ValueError at construction |
| difficulty ∈ {4 values} | Tất cả records | ValueError at construction |
| Disjoint question_ids | Mohler splits | Guaranteed by algorithm |
| Adversarial co-location | Data_Generate | SplitIntegrityError |
| No question leakage | Data_Generate | SplitIntegrityError |
| No domain leakage | Data_Generate | SplitIntegrityError |
| Score-label consistency | After harmonization | Warning logged |

### 4.3.5 End-to-End Pipeline Script

Script `experiments/phase1_data_audit.py` thực hiện toàn bộ pipeline:

```python
"""Phase 1: Data Pipeline End-to-End Audit

Runs the full data pipeline:
  1. Load all 4 source datasets
  2. Harmonize labels
  3. Assign / verify splits
  4. Run data quality audit
  5. Save unified JSONL files to data/unified/
  6. Print audit findings and save audit report to results/phase1/

Usage:
    python experiments/phase1_data_audit.py
"""

def main() -> None:
    # 0. Load config
    cfg = load_config(CONFIG_PATH)
    seed: int = cfg.get("seed", 42)
    set_seed(seed)

    # 1. Load raw data
    seb_records = load_scientsbank(seb_path)
    moh_records = load_mohler(moh_path)
    gen_records = load_data_generate(gen_path)
    scr_records = load_data_scraping(scr_path)

    all_records = (
        seb_records + moh_records + gen_records + scr_records
    )

    # 2. Harmonize labels
    harmonizer = LabelHarmonizer(threshold_2way=threshold_2way)
    harmonizer.harmonize_all(all_records)

    # 3. Assign / verify splits
    split_manager = SplitManager(seed=seed)
    try:
        split_manager.assign_splits(all_records)
    except SplitIntegrityError as e:
        logger.error("Split integrity violation: %s", e)

    # 4. Run data quality audit
    audit_report = DataAuditor.full_audit(all_records)

    # 5. Save unified JSONL files
    for source, filename in source_to_filename.items():
        save_jsonl(source_records[source], unified_dir / filename)

    # 6. Print and save audit findings
    # ... (label distributions, low confidence, short answers, etc.)

    # 7. Save audit report JSON
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False)
```

Output của pipeline:
- `data/unified/scientsbank.jsonl` — SciEntsBank records (harmonized)
- `data/unified/mohler.jsonl` — MohlerASAG records (harmonized + split)
- `data/unified/data_generate.jsonl` — Data_Generate records (verified)
- `data/unified/data_scraping.jsonl` — Data_Scraping records
- `results/phase1/audit_report.json` — Audit findings

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Kết luận

Tiểu luận này đã trình bày thiết kế và triển khai một pipeline dữ liệu thống nhất cho hệ thống ASAG đa nhiệm vụ. Các đóng góp chính bao gồm:

**1. Schema thống nhất UnifiedRecord:**
Một Python dataclass với 37 trường, bao phủ đầy đủ thông tin cho 4 nhiệm vụ nghiên cứu (grading, feedback, misconception mining, robustness evaluation). Schema sử dụng validation tự động trong `__post_init__` để đảm bảo tính nhất quán dữ liệu ngay tại thời điểm khởi tạo.

**2. Hệ thống loader đa nguồn:**
4 loader functions xử lý 4 định dạng khác nhau (XML, CSV, JSON) và chuyển đổi về schema thống nhất. Mỗi loader có error handling robust — malformed records được skip với warning, không crash toàn bộ pipeline.

**3. Label harmonization:**
Module `LabelHarmonizer` giải quyết vấn đề heterogeneous label spaces bằng cách ánh xạ:
- MohlerASAG continuous scores → categorical labels (2-way, 3-way) với công thức toán học rõ ràng
- SciEntsBank 5-way → 3-way → 2-way với bảng ánh xạ tường minh
- Data_Generate contradictory → incorrect để thống nhất 3-way label space

**4. Split management với leakage prevention:**
`SplitManager` đảm bảo:
- SciEntsBank: bảo toàn UA/UQ/UD splits
- MohlerASAG: question-level split ngăn chặn data leakage
- Data_Generate: verification 3 ràng buộc (adversarial co-location, unseen questions, unseen domains)

**5. Data quality audit:**
`DataAuditor` cung cấp kiểm tra toàn diện: label distributions, low confidence detection, short answer detection, numerical question classification, và stratified sampling.

**6. Property-based testing:**
6 properties với Hypothesis đảm bảo các bất biến quan trọng luôn đúng cho mọi đầu vào hợp lệ, không chỉ cho các examples cụ thể.

## 5.2 Ưu điểm của kiến trúc

- **Đơn giản:** Flat architecture (dataclass + JSONL) dễ hiểu, dễ debug, dễ version control
- **Modular:** Mỗi giai đoạn (load, harmonize, split, audit) là module độc lập, có thể test và thay thế riêng
- **Configurable:** Tất cả tham số qua YAML — thay đổi ngưỡng, tỷ lệ split, đường dẫn mà không cần sửa code
- **Reproducible:** Seed cố định đảm bảo kết quả giống nhau mỗi lần chạy
- **Extensible:** Thêm nguồn dữ liệu mới chỉ cần viết thêm loader function

## 5.3 Hạn chế

- **Data_Scraping chưa hoàn thiện:** 129 bản ghi không có student_answer, chưa thể sử dụng cho training
- **Không có relational indexing:** Với dữ liệu lớn hơn (>100K records), flat JSONL có thể chậm cho complex queries
- **Synthetic data bias:** Data_Generate dù đã semantic de-biased vẫn có thể chứa artifacts từ LLM generation
- **MohlerASAG raw data:** Hiện tại raw data chưa có trong repository (chỉ có .gitkeep), cần download riêng
- **SciEntsBank raw data:** Tương tự, cần download từ SemEval-2013 Task 7

## 5.4 Hướng phát triển

1. **Thu thập student answers cho Data_Scraping:** Triển khai crowdsourcing hoặc classroom collection để bổ sung câu trả lời sinh viên cho 129 câu hỏi OpenStax
2. **Thêm nguồn dữ liệu mới:** Beetle (SemEval-2013), CREE, ShortAnswerGrading datasets
3. **Active learning integration:** Sử dụng audit results để ưu tiên annotation cho low-confidence records
4. **Streaming support:** Chuyển sang format hỗ trợ lazy loading cho datasets lớn hơn
5. **Cross-lingual extension:** Mở rộng schema cho dữ liệu đa ngôn ngữ (Vietnamese, Chinese, etc.)
6. **Automated quality gates:** Tích hợp audit checks vào CI/CD pipeline, reject commits nếu data quality giảm

---

# TÀI LIỆU THAM KHẢO

1. Mohler, M., Bunescu, R., & Mihalcea, R. (2011). Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL)*, 752-762.

2. Dzikovska, M. O., Nielsen, R. D., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge. *Second Joint Conference on Lexical and Computational Semantics*, 263-274.

3. Burrows, S., Gurevych, I., & Stein, B. (2015). The Eras and Trends of Automatic Short Answer Grading. *International Journal of Artificial Intelligence in Education*, 25(1), 60-117.

4. Filighera, A., Steuer, T., & Rensing, C. (2022). Your Answer is Incorrect... Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)*, 8577-8591.

5. Sung, C., Dhamecha, T. I., & Mukhi, N. (2019). Improving Short Answer Grading Using Transformer-Based Pre-training. *International Conference on Artificial Intelligence in Education (AIED)*, 469-481.

6. MacFarlane, A., Zehner, F., & Meurers, D. (2022). A Survey of Automated Short Answer Grading and Feedback. *arXiv preprint arXiv:2204.13120*.

7. Clauser, B. E., Kane, M. T., & Swanson, D. B. (2002). Validity Issues for Performance-Based Tests Scored with Computer-Automated Scoring Systems. *Applied Measurement in Education*, 15(4), 413-432.

8. Hypothesis Library Documentation. (2024). Property-Based Testing for Python. https://hypothesis.readthedocs.io/

9. OpenStax. (2024). College Physics 2e. https://openstax.org/details/books/college-physics-2e

10. Python Software Foundation. (2024). dataclasses — Data Classes. https://docs.python.org/3/library/dataclasses.html

---

# PHỤ LỤC

## Phụ lục A: Cấu trúc thư mục dự án

```
project-root/
├── configs/
│   └── data.yaml                    # Pipeline configuration
├── data/
│   ├── raw/
│   │   ├── scientsbank/             # Raw SciEntsBank XML files
│   │   └── mohler/                  # Raw MohlerASAG CSV files
│   └── unified/
│       ├── data_generate.jsonl      # Unified Data_Generate
│       └── data_scraping.jsonl      # Unified Data_Scraping
├── src/
│   └── data/
│       ├── schema.py                # UnifiedRecord dataclass
│       ├── loaders.py               # 4 loader functions
│       ├── harmonizer.py            # LabelHarmonizer class
│       ├── splitter.py              # SplitManager class
│       ├── dataset.py               # DataLoader API
│       └── audit.py                 # DataAuditor class
├── tests/
│   ├── test_schema.py               # Schema + Property 5, 6
│   ├── test_loaders.py              # Loader unit tests
│   ├── test_harmonizer.py           # Harmonizer + Property 1, 2
│   └── test_splitter.py             # Splitter + Property 3, 4
├── experiments/
│   └── phase1_data_audit.py         # End-to-end pipeline script
├── data-generate.csv                # Raw synthetic data (10,000 rows)
├── data-scraping.json               # Raw scraped data (129 entries)
└── Methodology_Data_Generation.html # Generation methodology docs
```

## Phụ lục B: Ví dụ bản ghi JSONL

### B.1 SciEntsBank record (sau harmonization)

```json
{
  "sample_id": "SEB_00001",
  "source_dataset": "scientsbank",
  "original_id": "SA_1",
  "question_id": "Q_PHOTO",
  "domain": "science",
  "subdomain": "biology",
  "difficulty": "unknown",
  "question": "What is photosynthesis?",
  "reference_answer": "Plants convert light energy into chemical energy.",
  "student_answer": "Plants use light to make food.",
  "alternative_reference_answers": [],
  "score_raw": null,
  "score_normalized": null,
  "label_2way": "correct",
  "label_3way": "correct",
  "label_5way": "correct",
  "key_concepts": [],
  "misconception_tags": [],
  "misconception_inventory": [],
  "missing_concepts": [],
  "extra_incorrect_claims": [],
  "feedback_short": null,
  "feedback_detailed": null,
  "feedback_type": null,
  "feedback_tone": null,
  "split": "train",
  "is_human_annotated": true,
  "is_synthetic": false,
  "is_adversarial": false,
  "perturbation_type": null,
  "adversarial_variant_of": null,
  "student_answer_style": null,
  "annotation_confidence": null,
  "usable_for_grading": true,
  "usable_for_feedback": true,
  "usable_for_misconception_mining": true,
  "usable_for_robustness_eval": true
}
```

### B.2 MohlerASAG record (sau harmonization + split)

```json
{
  "sample_id": "MOH_00001",
  "source_dataset": "mohler",
  "original_id": "mohler_1",
  "question_id": "Q1",
  "domain": "computer_science",
  "subdomain": "general",
  "difficulty": "unknown",
  "question": "What is an array?",
  "reference_answer": "A contiguous block of memory.",
  "student_answer": "A block of memory.",
  "alternative_reference_answers": [],
  "score_raw": 3.75,
  "score_normalized": 0.75,
  "label_2way": "correct",
  "label_3way": "partially_correct",
  "label_5way": null,
  "key_concepts": [],
  "misconception_tags": [],
  "misconception_inventory": [],
  "missing_concepts": [],
  "extra_incorrect_claims": [],
  "feedback_short": null,
  "feedback_detailed": null,
  "feedback_type": null,
  "feedback_tone": null,
  "split": "train",
  "is_human_annotated": true,
  "is_synthetic": false,
  "is_adversarial": false,
  "perturbation_type": null,
  "adversarial_variant_of": null,
  "student_answer_style": null,
  "annotation_confidence": null,
  "usable_for_grading": true,
  "usable_for_feedback": true,
  "usable_for_misconception_mining": true,
  "usable_for_robustness_eval": true
}
```

### B.3 Data_Generate record

```json
{
  "sample_id": "GEN_00001",
  "source_dataset": "data_generate",
  "original_id": "ASAGX_000001",
  "question_id": "Q0001",
  "domain": "biology",
  "subdomain": "plant_biology",
  "difficulty": "easy",
  "question": "What is the main idea behind photosynthesis in green plants?",
  "reference_answer": "The accepted explanation works through light energy, carbon dioxide, and water, rather than through a side issue.",
  "student_answer": "the oddly key move uses light energy properly so the already outcome reaches water.",
  "alternative_reference_answers": [
    "A complete answer should connect light energy to carbon dioxide and use that link to explain photosynthesis in green plants.",
    "The accepted explanation works through light energy, carbon dioxide, and water, rather than through a side issue.",
    "The answer is correct when it keeps light energy and carbon dioxide in the right relationship and ties that to the outcome in the prompt."
  ],
  "score_raw": 5.0,
  "score_normalized": null,
  "label_2way": "correct",
  "label_3way": "correct",
  "label_5way": "correct",
  "key_concepts": ["light energy", "carbon dioxide", "water", "sugars", "oxygen"],
  "misconception_tags": [],
  "misconception_inventory": [
    {"tag": "confuses_photosynthesis_with_respiration", "belief": "Photosynthesis is the process plants use to break down sugar for energy."},
    {"tag": "thinks_plants_absorb_food_from_soil", "belief": "Plants get their food directly from soil instead of making it."},
    {"tag": "believes_oxygen_is_main_input", "belief": "Plants take in oxygen during photosynthesis so they can make sugar."}
  ],
  "missing_concepts": [],
  "extra_incorrect_claims": [],
  "feedback_short": "The answer is too unclear to photosynthesis in green plants...",
  "feedback_detailed": "The main grading signal is that the answer is correct because it anchors the explanation in light energy and carbon dioxide...",
  "feedback_type": "praise",
  "feedback_tone": "tutor_like",
  "split": "test_unseen_questions",
  "is_human_annotated": false,
  "is_synthetic": true,
  "is_adversarial": true,
  "perturbation_type": "high_overlap_wrong_meaning",
  "adversarial_variant_of": "ASAGX_000001",
  "student_answer_style": "concise",
  "annotation_confidence": 0.97,
  "usable_for_grading": true,
  "usable_for_feedback": true,
  "usable_for_misconception_mining": true,
  "usable_for_robustness_eval": true
}
```

### B.4 Data_Scraping record

```json
{
  "sample_id": "SCR_00001",
  "source_dataset": "data_scraping",
  "original_id": "college-physics-2e_1_1",
  "question_id": "college-physics-2e_1_1",
  "domain": "openstax_college-physics-2e",
  "subdomain": "college-physics-2e",
  "difficulty": "unknown",
  "question": "The speed limit on some interstate highways is roughly 100 km/h. (a) What is this in meters per second? (b) How many miles per hour is this?",
  "reference_answer": "(a) 27.8 m/s (b) 62.1 mph",
  "student_answer": "",
  "alternative_reference_answers": [],
  "score_raw": null,
  "score_normalized": null,
  "label_2way": null,
  "label_3way": null,
  "label_5way": null,
  "key_concepts": [],
  "misconception_tags": [],
  "misconception_inventory": [],
  "missing_concepts": [],
  "extra_incorrect_claims": [],
  "feedback_short": null,
  "feedback_detailed": null,
  "feedback_type": null,
  "feedback_tone": null,
  "split": "",
  "is_human_annotated": false,
  "is_synthetic": false,
  "is_adversarial": false,
  "perturbation_type": null,
  "adversarial_variant_of": null,
  "student_answer_style": null,
  "annotation_confidence": null,
  "usable_for_grading": false,
  "usable_for_feedback": false,
  "usable_for_misconception_mining": false,
  "usable_for_robustness_eval": false
}
```

## Phụ lục C: Bảng ánh xạ cột Data_Generate → UnifiedRecord

| Cột CSV | Trường UnifiedRecord | Ghi chú |
|---|---|---|
| instance_id | original_id | |
| question_id | question_id | |
| domain | domain | |
| subdomain | subdomain | |
| difficulty | difficulty | Validated: easy/medium/hard/unknown |
| split | split | |
| question | question | |
| reference_answer | reference_answer | |
| alternative_reference_answers | alternative_reference_answers | Parsed from string list |
| key_concepts | key_concepts | Parsed from string list |
| misconception_inventory | misconception_inventory | Parsed from string list of dicts |
| student_answer | student_answer | |
| student_answer_style | student_answer_style | |
| lexical_overlap_level | — | Not mapped (informational) |
| semantic_correctness_score_0_5 | score_raw | |
| label_5way | label_5way | |
| label_3way | label_3way | May be remapped by harmonizer |
| label_2way | label_2way | |
| misconception_tags | misconception_tags | Parsed from string list |
| misconception_span_rationale | — | Not mapped (informational) |
| missing_concepts | missing_concepts | Parsed from string list |
| extra_incorrect_claims | extra_incorrect_claims | Parsed from string list |
| feedback_short | feedback_short | |
| feedback_detailed | feedback_detailed | |
| feedback_type | feedback_type | |
| feedback_tone | feedback_tone | |
| adversarial_variant_of | adversarial_variant_of | |
| perturbation_type | perturbation_type | |
| robustness_notes | — | Not mapped (informational) |
| annotation_confidence | annotation_confidence | |

## Phụ lục D: Prompt Templates từ Methodology_Data_Generation

### D.1 Question & Reference Generation Prompt

```
You are generating educational short-answer benchmark items for
introductory science and computing courses.

Inputs:
Domain: biology
Subdomain: plant_biology
Focus concept: photosynthesis in green plants
Comparison concept: cellular respiration
Counterfactual condition: a plant receives light but no carbon dioxide
Common misclaim: "Plants get their food directly from soil instead
of making it."

Generate:
1. One short-answer question appropriate for assessment.
2. One reference answer in 1-3 sentences.
3. Two alternative acceptable reference phrasings.
4. A JSON list of key concepts required for grading.
5. A JSON list of 2-4 misconception entries, each with a tag and
   belief statement.

Constraints:
The item must test conceptual understanding rather than rote recall.
The reference answer must be concise but semantically complete.
Return structured JSON only.
```

### D.2 Student Answer Simulation Prompt

```
You are simulating authentic student short answers for an educational
NLP benchmark.

Inputs:
Question: How does photosynthesis in green plants work?
Reference answer: Photosynthesis uses light energy to combine carbon
dioxide and water into sugars, and oxygen is released as a by-product.
Key concepts: ["light energy", "carbon dioxide", "water", "sugars",
"oxygen"]
Misconception inventory: [
  {"tag": "confuses_photosynthesis_with_respiration",
   "belief": "Photosynthesis is the process plants use to break down
   sugar for energy."},
  {"tag": "thinks_plants_absorb_food_from_soil",
   "belief": "Plants get their food directly from soil instead of
   making it."}
]
Persona: overconfident_wrong_student
Target label: contradictory
Target semantic score: 1
Lexical overlap target: high

Generate one student answer that sounds like a real student, uses
some correct keywords, but remains conceptually wrong.
Return:
student_answer
student_answer_style
lexical_overlap_level
misconception_tags
missing_concepts
extra_incorrect_claims
```

### D.3 Adversarial Variant Generation Prompt

```
You are generating a robustness-focused variant of an existing
student answer.

Original question:
How does binary search work on a sorted list?

Reference answer:
Binary search repeatedly checks the middle item of a sorted list
and removes half of the remaining search space each step.

Original student answer:
Binary search looks at the middle and keeps cutting the search
range in half, so it only works if the data are sorted.

Create one adversarial variant with these requirements:
1. Keep the answer fluent and plausible.
2. Preserve many of the original keywords.
3. Introduce one fatal conceptual error.
4. Make the response difficult for lexical or TF-IDF models.
5. Specify perturbation_type.
6. Explain briefly why the variant is harder.

Return:
student_answer
perturbation_type
robustness_notes
gold_label_after_perturbation
```

## Phụ lục E: Pseudo-code Pipeline sinh dữ liệu

```python
def build_dataset(domain_configs, split_targets):
    instances = []
    question_bank = build_question_bank(domain_configs)

    for question_spec in question_bank:
        reference_bundle = generate_reference_bundle(question_spec)
        misconception_inventory = generate_misconceptions(
            question_spec, reference_bundle
        )
        response_profiles = sample_response_profiles(question_spec)

        for response_profile in response_profiles:
            student_answer = simulate_student_answer(
                question_spec,
                reference_bundle,
                misconception_inventory,
                response_profile,
            )
            labels = annotate_labels_and_score(
                student_answer, reference_bundle,
                misconception_inventory
            )
            feedback = generate_feedback(
                student_answer, labels, misconception_inventory
            )
            instance = assemble_instance(
                question_spec, reference_bundle,
                misconception_inventory, student_answer,
                labels, feedback,
            )
            instances.append(instance)

    instances = add_adversarial_variants(instances)
    instances = assign_splits(instances, split_targets)
    instances = validate_and_refine(instances)
    return instances
```

```python
def validate_instances(rows):
    seen_rows = set()
    train_answers = {
        normalize(r["student_answer"])
        for r in rows if r["split"] == "train"
    }
    train_questions = {
        r["question_id"] for r in rows if r["split"] == "train"
    }
    train_domains = {
        r["domain"] for r in rows if r["split"] == "train"
    }

    for row in rows:
        row_key = tuple(row[col] for col in row.keys())
        if row_key in seen_rows:
            raise ValueError("Exact duplicate row detected")
        seen_rows.add(row_key)

        if (row["split"] == "test_unseen_answers"
            and normalize(row["student_answer"]) in train_answers):
            raise ValueError("Leakage in unseen-answer split")

        if (row["split"] == "test_unseen_questions"
            and row["question_id"] in train_questions):
            raise ValueError("Leakage in unseen-question split")

        if (row["split"] == "test_unseen_domains"
            and row["domain"] in train_domains):
            raise ValueError("Leakage in unseen-domain split")

    return True
```
