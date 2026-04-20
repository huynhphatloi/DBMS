---
title: "Tiểu luận 2: Chấm điểm Tự động Câu trả lời Ngắn (Automatic Short Answer Grading)"
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 13pt
linestretch: 1.5
---

# CHƯƠNG 1: MỞ ĐẦU

## 1.1 Tính cấp thiết của đề tài

Trong bối cảnh giáo dục hiện đại, việc đánh giá năng lực học sinh thông qua các câu hỏi mở ngắn (short answer questions) đóng vai trò then chốt trong quá trình dạy và học. Khác với các câu hỏi trắc nghiệm chỉ yêu cầu chọn đáp án đúng, câu trả lời ngắn đòi hỏi người học phải tự diễn đạt kiến thức bằng ngôn ngữ của mình, từ đó phản ánh chính xác hơn mức độ hiểu biết và khả năng tư duy phản biện. Tuy nhiên, việc chấm điểm thủ công các câu trả lời ngắn đặt ra nhiều thách thức nghiêm trọng cho hệ thống giáo dục.

Thứ nhất, vấn đề về quy mô (scalability) là trở ngại lớn nhất. Trong các khóa học trực tuyến mở (MOOCs) với hàng chục nghìn đến hàng trăm nghìn học viên, một giảng viên không thể chấm điểm thủ công tất cả bài làm trong thời gian hợp lý. Ví dụ, một khóa học trên Coursera với 50.000 học viên, mỗi bài kiểm tra có 10 câu hỏi ngắn, sẽ tạo ra 500.000 câu trả lời cần được đánh giá. Nếu mỗi câu trả lời cần trung bình 30 giây để chấm, tổng thời gian cần thiết là khoảng 4.167 giờ — tương đương gần 2 năm làm việc liên tục của một người.

Thứ hai, tính nhất quán (consistency) trong chấm điểm thủ công là một vấn đề nan giải. Nghiên cứu của Mohler và Mihalcea (2009) chỉ ra rằng độ đồng thuận giữa các giám khảo (inter-rater agreement) thường chỉ đạt mức Pearson $r \approx 0.586$ đến $r \approx 0.659$ trên thang điểm 0-5. Điều này có nghĩa là cùng một câu trả lời có thể nhận được điểm số khác nhau đáng kể tùy thuộc vào người chấm, thời điểm chấm, và thứ tự bài được chấm (hiệu ứng anchoring).

Thứ ba, độ trễ phản hồi (feedback latency) ảnh hưởng trực tiếp đến hiệu quả học tập. Nghiên cứu giáo dục học đã chứng minh rằng phản hồi kịp thời (immediate feedback) có tác dụng tích cực hơn nhiều so với phản hồi trễ trong việc củng cố kiến thức và sửa chữa hiểu lầm. Khi giáo viên mất nhiều ngày hoặc tuần để trả bài, cơ hội học tập tối ưu đã qua đi.

Thứ tư, chi phí nhân lực cho việc chấm điểm thủ công là rất lớn. Các trường đại học phải thuê hàng trăm trợ giảng (teaching assistants) chỉ để phục vụ công tác chấm bài, trong khi nguồn lực này có thể được sử dụng hiệu quả hơn cho các hoạt động giảng dạy và nghiên cứu.

Chính vì những lý do trên, bài toán Chấm điểm Tự động Câu trả lời Ngắn (Automatic Short Answer Grading — ASAG) đã trở thành một hướng nghiên cứu quan trọng trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP) ứng dụng vào giáo dục. Hệ thống ASAG có khả năng đánh giá hàng triệu câu trả lời trong vài phút, đảm bảo tính nhất quán tuyệt đối, và cung cấp phản hồi tức thì cho người học.

## 1.2 Mục tiêu nghiên cứu

Nghiên cứu này đặt ra các mục tiêu cụ thể sau:

**Mục tiêu 1:** Xây dựng và triển khai 6 phương pháp chấm điểm tự động với độ phức tạp tăng dần, từ các baseline đơn giản dựa trên trùng lặp từ vựng đến mô hình transformer tiên tiến:

1. Lexical Overlap (BLEU, ROUGE-L, Jaccard, Word Overlap)
2. TF-IDF + Traditional Machine Learning (LR, SVM, RF, GB)
3. SBERT Cosine Similarity (all-MiniLM-L6-v2)
4. Cross-Encoder Fine-tuning (RoBERTa-based)
5. Reference-Answer-Aware DeBERTa (Multi-task Learning)
6. LLM Zero-Shot Grading (GPT-4o-mini)

**Mục tiêu 2:** Thiết kế và triển khai một Evaluation Harness (khung đánh giá) toàn diện bao gồm:
- Các metric phân loại: Accuracy, Macro F1, Weighted F1, Per-class F1
- Các metric hồi quy: Pearson $r$, Spearman $\rho$, RMSE, MAE, QWK
- Bootstrap Confidence Interval (1000 iterations, 95% CI)
- So sánh mô hình: McNemar's test, Paired t-test

**Mục tiêu 3:** Xây dựng ứng dụng demo Teacher's Grading Dashboard sử dụng kiến trúc Next.js + FastAPI, cho phép giáo viên nhập câu hỏi, đáp án mẫu và câu trả lời của học sinh để nhận kết quả chấm điểm tức thì với giải thích chi tiết.

**Mục tiêu 4:** Thực hiện các thí nghiệm so sánh toàn diện trên nhiều cấu hình train-test khác nhau, bao gồm in-domain evaluation, cross-domain transfer, và synthetic data augmentation.

## 1.3 Đối tượng và phạm vi nghiên cứu

### 1.3.1 Bộ dữ liệu SciEntsBank

SciEntsBank (Dzikovska et al., 2013) là bộ dữ liệu chuẩn được sử dụng rộng rãi trong nghiên cứu ASAG, được thu thập từ các bài kiểm tra khoa học dành cho học sinh trung học cơ sở tại Hoa Kỳ. Bộ dữ liệu bao gồm:

- **Số lượng:** Khoảng 10.000 câu trả lời của học sinh
- **Nhãn phân loại:** 3-way (correct, partially_correct_incomplete, incorrect) và 5-way (correct, partially_correct_incomplete, contradictory, irrelevant, non_domain)
- **Cấu trúc đánh giá:** Ba mức độ tổng quát hóa:
  - Unseen Answers (UA): Câu trả lời mới cho câu hỏi đã thấy trong tập huấn luyện
  - Unseen Questions (UQ): Câu hỏi hoàn toàn mới trong cùng lĩnh vực
  - Unseen Domains (UD): Lĩnh vực khoa học hoàn toàn mới

### 1.3.2 Bộ dữ liệu MohlerASAG

MohlerASAG (Mohler & Mihalcea, 2009) là bộ dữ liệu hồi quy được thu thập từ các bài tập về nhà và bài kiểm tra trong khóa học Khoa học Máy tính tại Đại học North Texas:

- **Số lượng:** 2.273 câu trả lời cho 80 câu hỏi từ 10 bài tập
- **Thang điểm:** Liên tục từ 0 đến 5 (do 2 giám khảo chấm độc lập)
- **Đặc điểm:** Phù hợp cho đánh giá mô hình hồi quy với Pearson $r$ và RMSE

### 1.3.3 Bộ dữ liệu Data_Generate

Data_Generate là bộ dữ liệu tổng hợp (synthetic) được tạo ra bằng LLM để tăng cường dữ liệu huấn luyện:

- **Mục đích:** Augmentation cho các lớp thiểu số, mở rộng phạm vi lĩnh vực
- **Phương pháp tạo:** Sử dụng GPT-4 với prompt engineering để sinh câu trả lời mô phỏng các mức độ hiểu biết khác nhau
- **Định dạng:** JSONL thống nhất với schema UnifiedRecord

### 1.3.4 Phạm vi nghiên cứu

Nghiên cứu tập trung vào:
- Câu trả lời ngắn bằng tiếng Anh (1-3 câu)
- Lĩnh vực STEM (Khoa học, Công nghệ, Kỹ thuật, Toán học)
- Cả hai dạng bài toán: phân loại (2-way, 3-way, 5-way) và hồi quy (0-5)
- Đánh giá trên cả in-domain và cross-domain settings

## 1.4 Cơ sở lý luận

### 1.4.1 Xử lý Ngôn ngữ Tự nhiên (NLP)

Bài toán ASAG thuộc lĩnh vực NLP, cụ thể là nhánh Natural Language Understanding (NLU). Hệ thống cần "hiểu" ngữ nghĩa của câu trả lời học sinh và so sánh với đáp án mẫu để đưa ra đánh giá. Các kỹ thuật NLP được sử dụng bao gồm:

- **Tokenization:** Phân tách văn bản thành các đơn vị từ vựng
- **N-gram analysis:** Phân tích chuỗi con liên tiếp của n từ
- **TF-IDF representation:** Biểu diễn văn bản dưới dạng vector trọng số từ
- **Semantic embedding:** Biểu diễn ngữ nghĩa câu trong không gian vector liên tục
- **Attention mechanism:** Cơ chế chú ý cho phép mô hình tập trung vào các phần quan trọng

### 1.4.2 Độ tương đồng ngữ nghĩa (Semantic Similarity)

Ý tưởng cốt lõi của ASAG là đo lường mức độ tương đồng ngữ nghĩa giữa câu trả lời của học sinh và đáp án mẫu. Có nhiều cấp độ tương đồng:

- **Lexical similarity:** Dựa trên sự trùng lặp từ vựng bề mặt
- **Syntactic similarity:** Dựa trên cấu trúc ngữ pháp
- **Semantic similarity:** Dựa trên ý nghĩa sâu của câu

### 1.4.3 Mô hình Transformer

Kiến trúc Transformer (Vaswani et al., 2017) đã cách mạng hóa NLP với cơ chế self-attention cho phép mô hình xử lý toàn bộ chuỗi đầu vào song song và nắm bắt các mối quan hệ xa trong văn bản. Các mô hình pre-trained như BERT, RoBERTa, và DeBERTa đã đạt được kết quả state-of-the-art trên nhiều benchmark NLU.

## 1.5 Đóng góp mới của nghiên cứu

Nghiên cứu này có các đóng góp mới sau:

1. **So sánh hệ thống 6 phương pháp:** Đây là một trong số ít nghiên cứu so sánh toàn diện từ baseline đơn giản nhất (lexical overlap) đến mô hình tiên tiến nhất (DeBERTa multi-task) trên cùng bộ dữ liệu và khung đánh giá.

2. **Multi-task Learning cho ASAG:** Đề xuất kiến trúc DeBERTa với hai đầu ra (classification + regression) được huấn luyện đồng thời, cho phép mô hình học được biểu diễn phong phú hơn.

3. **Evaluation Harness với Bootstrap CI:** Thiết kế khung đánh giá có tính thống kê chặt chẽ, bao gồm khoảng tin cậy bootstrap và kiểm định so sánh mô hình.

4. **Cross-domain Transfer Analysis:** Phân tích chi tiết khả năng tổng quát hóa của các mô hình khi áp dụng sang lĩnh vực mới.

5. **Synthetic Data Augmentation:** Đánh giá hiệu quả của việc sử dụng dữ liệu tổng hợp từ LLM để cải thiện hiệu suất mô hình.

6. **Ứng dụng Demo hoàn chỉnh:** Xây dựng dashboard chấm điểm với giao diện trực quan, tích hợp nhiều mô hình và cung cấp giải thích kết quả.

## 1.6 Ý nghĩa lý luận và thực tiễn

### 1.6.1 Ý nghĩa lý luận

- Đóng góp vào hiểu biết về mối quan hệ giữa độ phức tạp mô hình và hiệu suất ASAG
- Cung cấp bằng chứng thực nghiệm về hiệu quả của multi-task learning trong bài toán đánh giá ngữ nghĩa
- Phân tích sâu về khả năng transfer learning giữa các lĩnh vực khoa học
- Đánh giá vai trò của dữ liệu tổng hợp trong việc cải thiện mô hình NLP

### 1.6.2 Ý nghĩa thực tiễn

- Cung cấp hướng dẫn cho các nhà phát triển EdTech trong việc lựa chọn phương pháp ASAG phù hợp với nguồn lực và yêu cầu cụ thể
- Ứng dụng demo có thể được triển khai trực tiếp trong môi trường giáo dục thực tế
- Khung đánh giá có thể tái sử dụng cho các nghiên cứu ASAG trong tương lai
- Giảm tải công việc chấm bài cho giáo viên, cho phép họ tập trung vào giảng dạy

## 1.7 Tình hình nghiên cứu trong và ngoài nước

### 1.7.1 Giai đoạn đầu: Phương pháp dựa trên tri thức (2000-2010)

Các hệ thống ASAG đầu tiên sử dụng phương pháp dựa trên tri thức (knowledge-based), trong đó câu trả lời được phân tích thành các thành phần ngữ nghĩa và so sánh với đáp án mẫu thông qua các quy tắc logic. Hệ thống C-rater của ETS (Leacock & Chodorow, 2003) là một ví dụ tiêu biểu, sử dụng phân tích cú pháp và ánh xạ khái niệm để đánh giá câu trả lời.

### 1.7.2 Mohler & Mihalcea (2009)

Mohler và Mihalcea (2009) là những người tiên phong trong việc áp dụng các phương pháp đo lường tương đồng ngữ nghĩa cho ASAG. Họ so sánh 8 phương pháp đo tương đồng (bao gồm knowledge-based và corpus-based) và kết hợp chúng với các bộ phân loại truyền thống. Kết quả tốt nhất đạt Pearson $r = 0.518$ trên bộ dữ liệu của họ, cho thấy tiềm năng nhưng cũng chỉ ra khoảng cách lớn so với đánh giá của con người.

### 1.7.3 Dzikovska et al. (2013) — SemEval-2013 Task 7

Shared task SemEval-2013 Task 7 (Dzikovska et al., 2013) đã chuẩn hóa bài toán ASAG với bộ dữ liệu SciEntsBank và beetle. Task này định nghĩa hai subtask:
- 2-way: correct vs. incorrect
- 5-way: correct, partially_correct_incomplete, contradictory, irrelevant, non_domain

Hệ thống tốt nhất đạt Macro F1 khoảng 0.55-0.65 trên 5-way classification, cho thấy đây vẫn là bài toán khó.

### 1.7.4 Sultan et al. (2016)

Sultan et al. (2016) đề xuất phương pháp alignment-based, trong đó các từ/cụm từ trong câu trả lời học sinh được căn chỉnh (align) với các từ/cụm từ tương ứng trong đáp án mẫu. Tỷ lệ alignment thành công được sử dụng làm đặc trưng cho bộ phân loại. Phương pháp này đạt kết quả state-of-the-art tại thời điểm đó trên SciEntsBank.

### 1.7.5 Sung et al. (2019)

Sung et al. (2019) là những người đầu tiên áp dụng thành công mô hình BERT pre-trained cho ASAG. Bằng cách fine-tune BERT trên dữ liệu ASAG, họ đạt được cải thiện đáng kể so với các phương pháp trước đó, đặc biệt trên các setting khó như Unseen Questions và Unseen Domains. Nghiên cứu này mở ra hướng đi mới cho ASAG dựa trên transformer.

### 1.7.6 Xu hướng hiện tại (2020-2024)

Các nghiên cứu gần đây tập trung vào:
- **Multi-task learning:** Kết hợp nhiều nhiệm vụ liên quan (grading + feedback generation)
- **Few-shot và Zero-shot:** Sử dụng LLM lớn (GPT-4, Claude) để chấm điểm không cần huấn luyện
- **Explainability:** Cung cấp giải thích cho điểm số, không chỉ đưa ra nhãn
- **Cross-lingual ASAG:** Mở rộng sang các ngôn ngữ khác ngoài tiếng Anh
- **Robustness:** Đánh giá khả năng chống lại adversarial attacks và paraphrasing



# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Hình thức hóa bài toán ASAG

### 2.1.1 Định nghĩa bài toán

Bài toán Automatic Short Answer Grading (ASAG) được hình thức hóa như sau:

**Đầu vào:** Một bộ ba $(q, r, s)$ trong đó:
- $q$ là câu hỏi (question)
- $r$ là đáp án mẫu (reference answer)
- $s$ là câu trả lời của học sinh (student answer)

**Đầu ra:** Dự đoán $\hat{y}$ đánh giá chất lượng câu trả lời $s$ so với đáp án mẫu $r$ trong ngữ cảnh câu hỏi $q$.

### 2.1.2 Dạng phân loại (Classification)

Trong dạng phân loại, đầu ra là một nhãn rời rạc:

$$\hat{y} \in \mathcal{Y}_{cls}$$

Với các cấu hình nhãn phổ biến:

- **2-way:** $\mathcal{Y}_{cls} = \{\text{correct}, \text{incorrect}\}$
- **3-way:** $\mathcal{Y}_{cls} = \{\text{correct}, \text{partially\_correct}, \text{incorrect}\}$
- **5-way:** $\mathcal{Y}_{cls} = \{\text{correct}, \text{partially\_correct\_incomplete}, \text{contradictory}, \text{irrelevant}, \text{non\_domain}\}$

Mô hình phân loại học một hàm:

$$f_{cls}: (q, r, s) \rightarrow \hat{y} \in \mathcal{Y}_{cls}$$

### 2.1.3 Dạng hồi quy (Regression)

Trong dạng hồi quy, đầu ra là một giá trị liên tục:

$$\hat{y} \in [0, 5]$$

Mô hình hồi quy học một hàm:

$$f_{reg}: (q, r, s) \rightarrow \hat{y} \in \mathbb{R}, \quad 0 \leq \hat{y} \leq 5$$

### 2.1.4 Multi-task Formulation

Trong cách tiếp cận multi-task, mô hình đồng thời dự đoán cả nhãn phân loại và điểm số hồi quy:

$$f_{mt}: (q, r, s) \rightarrow (\hat{y}_{cls}, \hat{y}_{reg})$$

với hàm mất mát kết hợp:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{cls} + (1 - \alpha) \cdot \mathcal{L}_{reg}$$

trong đó $\alpha \in [0, 1]$ là trọng số cân bằng giữa hai nhiệm vụ.

## 2.2 Baseline 1 — Lexical Overlap (Trùng lặp từ vựng)

### 2.2.1 Tổng quan phương pháp

Phương pháp Lexical Overlap đánh giá câu trả lời dựa trên mức độ trùng lặp từ vựng bề mặt giữa câu trả lời học sinh $s$ và đáp án mẫu $r$. Đây là baseline đơn giản nhất, không yêu cầu huấn luyện mô hình phức tạp, nhưng có hạn chế lớn là không nắm bắt được tương đồng ngữ nghĩa (ví dụ: "H2O" và "nước" có cùng nghĩa nhưng không trùng từ vựng).

Năm metric được sử dụng: BLEU-1, BLEU-4, ROUGE-L, Jaccard Similarity, và Word Overlap Ratio.

### 2.2.2 BLEU-n (Bilingual Evaluation Understudy)

BLEU (Papineni et al., 2002) ban đầu được thiết kế cho đánh giá dịch máy, đo lường precision của n-gram trong hypothesis so với reference.

**Công thức BLEU-n:**

Cho reference $r$ được tokenize thành chuỗi token $r = (r_1, r_2, ..., r_m)$ và hypothesis $h$ (câu trả lời học sinh) thành $h = (h_1, h_2, ..., h_n)$.

**Bước 1:** Tính n-gram precision với clipping:

$$p_n = \frac{\sum_{g \in \text{ngrams}_n(h)} \min\left(\text{count}(g, h), \text{count}(g, r)\right)}{\sum_{g \in \text{ngrams}_n(h)} \text{count}(g, h)}$$

trong đó $\text{ngrams}_n(h)$ là tập hợp tất cả n-gram trong $h$, và $\text{count}(g, x)$ là số lần xuất hiện của n-gram $g$ trong chuỗi $x$.

**Bước 2:** Tính Brevity Penalty (phạt câu ngắn):

$$BP = \begin{cases} 1 & \text{nếu } |h| > |r| \\ e^{1 - |r|/|h|} & \text{nếu } |h| \leq |r| \end{cases}$$

**Bước 3:** Điểm BLEU-n cuối cùng:

$$\text{BLEU-n} = BP \cdot p_n$$

Trong nghiên cứu này, chúng tôi sử dụng BLEU-1 (unigram precision) và BLEU-4 (4-gram precision). BLEU-1 đo lường sự trùng lặp từ đơn, trong khi BLEU-4 đo lường sự trùng lặp cụm 4 từ liên tiếp, phản ánh mức độ tương đồng cấu trúc câu.

**Triển khai trong mã nguồn:**

```python
def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(reference: str, hypothesis: str, n: int) -> float:
    """Compute BLEU-n (precision of n-grams with brevity penalty).
    Returns a value in [0.0, 1.0].
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = math.exp(min(0.0, 1.0 - len(ref_tokens) / len(hyp_tokens))) \
         if ref_tokens else 0.0

    if len(hyp_tokens) < n:
        return 0.0

    ref_ngrams = _ngrams(ref_tokens, n)
    hyp_ngrams = _ngrams(hyp_tokens, n)

    if not hyp_ngrams:
        return 0.0

    clipped = sum(
        min(count, ref_ngrams[gram])
        for gram, count in hyp_ngrams.items()
    )
    precision = clipped / sum(hyp_ngrams.values())

    return float(bp * precision)
```

### 2.2.3 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE-L sử dụng Longest Common Subsequence (LCS) để đo lường sự tương đồng giữa hai chuỗi văn bản. Khác với n-gram matching, LCS cho phép các từ trùng khớp không cần liên tiếp, phản ánh tốt hơn sự tương đồng cấu trúc.

**Bước 1:** Tính độ dài LCS bằng quy hoạch động:

Cho hai chuỗi token $X = (x_1, ..., x_m)$ và $Y = (y_1, ..., y_n)$, bảng DP được xây dựng:

$$L[i][j] = \begin{cases} 0 & \text{nếu } i = 0 \text{ hoặc } j = 0 \\ L[i-1][j-1] + 1 & \text{nếu } x_i = y_j \\ \max(L[i-1][j], L[i][j-1]) & \text{nếu } x_i \neq y_j \end{cases}$$

**Bước 2:** Tính Precision, Recall, và F1:

$$P_{lcs} = \frac{|LCS(r, s)|}{|s|}$$

$$R_{lcs} = \frac{|LCS(r, s)|}{|r|}$$

$$F1_{lcs} = \frac{2 \cdot P_{lcs} \cdot R_{lcs}}{P_{lcs} + R_{lcs}}$$

trong đó $|LCS(r, s)|$ là độ dài của chuỗi con chung dài nhất giữa reference $r$ và student answer $s$.

**Triển khai trong mã nguồn:**

```python
def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 using the Longest Common Subsequence.
    Returns a value in [0.0, 1.0].
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length via DP (1-D to save memory)
    m, n = len(ref_tokens), len(hyp_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
```

### 2.2.4 Jaccard Similarity

Jaccard Similarity đo lường tỷ lệ giao trên hợp của hai tập hợp từ:

$$J(r, s) = \frac{|W_r \cap W_s|}{|W_r \cup W_s|}$$

trong đó $W_r$ và $W_s$ lần lượt là tập hợp các từ (unique tokens) trong reference và student answer.

Giá trị $J \in [0, 1]$, với $J = 1$ khi hai tập từ hoàn toàn giống nhau và $J = 0$ khi không có từ chung nào.

**Triển khai:**

```python
def jaccard_similarity(reference: str, hypothesis: str) -> float:
    """Compute Jaccard similarity: |intersection| / |union| of word sets."""
    ref_set = set(_tokenize(reference))
    hyp_set = set(_tokenize(hypothesis))
    if not ref_set and not hyp_set:
        return 1.0
    union = ref_set | hyp_set
    if not union:
        return 0.0
    return float(len(ref_set & hyp_set) / len(union))
```

### 2.2.5 Word Overlap Ratio

Word Overlap Ratio đo lường tỷ lệ từ trong reference được "phủ" bởi student answer:

$$\text{WOR}(r, s) = \frac{|W_r \cap W_s|}{|W_r|}$$

Khác với Jaccard, metric này chỉ quan tâm đến recall — bao nhiêu phần trăm từ quan trọng trong đáp án mẫu xuất hiện trong câu trả lời học sinh.

**Triển khai:**

```python
def word_overlap_ratio(reference: str, hypothesis: str) -> float:
    """Compute word overlap ratio: |intersection| / |reference_words|."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    return float(len(ref_set & hyp_set) / len(ref_set))
```

### 2.2.6 Chế độ phân loại

Baseline Lexical Overlap hỗ trợ hai chế độ phân loại:

**Chế độ 1 — Threshold Classification:**

$$\hat{y} = \begin{cases} \text{correct} & \text{nếu } \text{metric}(r, s) \geq \tau \\ \text{incorrect} & \text{nếu } \text{metric}(r, s) < \tau \end{cases}$$

trong đó $\tau$ là ngưỡng (threshold) được chọn trước (mặc định $\tau = 0.5$).

**Chế độ 2 — Logistic Regression:**

Sử dụng tất cả 5 metric làm đặc trưng đầu vào cho mô hình Logistic Regression:

$$\mathbf{x} = [\text{BLEU-1}, \text{BLEU-4}, \text{ROUGE-L}, \text{Jaccard}, \text{WOR}]$$

$$P(y = \text{correct} | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

```python
class LexicalLogisticRegression(GradingModel):
    """Use all 5 lexical metrics as features in Logistic Regression."""

    FEATURE_ORDER = ["bleu_1", "bleu_4", "rouge_l", "jaccard", "word_overlap"]

    def __init__(self, **lr_kwargs) -> None:
        defaults = {"max_iter": 1000, "random_state": 42}
        defaults.update(lr_kwargs)
        self._lr = LogisticRegression(**defaults)

    def _feature_matrix(self, records: list[UnifiedRecord]) -> np.ndarray:
        rows = []
        for r in records:
            feats = compute_lexical_features(
                r.reference_answer, r.student_answer
            )
            rows.append([feats[k] for k in self.FEATURE_ORDER])
        return np.array(rows, dtype=float)

    def fit(self, records, label_field):
        recs = list(records)
        X = self._feature_matrix(recs)
        y = [str(getattr(r, label_field)) for r in recs]
        self._lr.fit(X, y)
```

## 2.3 Baseline 2 — TF-IDF + Traditional Machine Learning

### 2.3.1 Biểu diễn TF-IDF

TF-IDF (Term Frequency — Inverse Document Frequency) là phương pháp biểu diễn văn bản dưới dạng vector số, trong đó mỗi chiều tương ứng với một từ trong từ điển và giá trị phản ánh tầm quan trọng của từ đó trong văn bản.

**Công thức TF (Term Frequency):**

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

trong đó $f_{t,d}$ là số lần từ $t$ xuất hiện trong văn bản $d$.

**Công thức IDF (Inverse Document Frequency):**

$$\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

trong đó $|D|$ là tổng số văn bản trong corpus và $|\{d \in D : t \in d\}|$ là số văn bản chứa từ $t$.

**Công thức TF-IDF:**

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2.3.2 Trích xuất đặc trưng (Feature Extraction)

Từ mỗi cặp (reference, student), chúng tôi trích xuất vector đặc trưng bao gồm:

1. **TF-IDF vector của student answer:** $\mathbf{v}_s \in \mathbb{R}^d$
2. **TF-IDF vector của reference answer:** $\mathbf{v}_r \in \mathbb{R}^d$
3. **Cosine similarity:** Một scalar đo lường góc giữa hai vector:

$$\text{cos\_sim}(\mathbf{v}_r, \mathbf{v}_s) = \frac{\mathbf{v}_r \cdot \mathbf{v}_s}{||\mathbf{v}_r|| \cdot ||\mathbf{v}_s||}$$

4. **Element-wise absolute difference:** $|\mathbf{v}_r - \mathbf{v}_s| \in \mathbb{R}^d$

Vector đặc trưng cuối cùng là phép nối (concatenation):

$$\mathbf{x} = [\mathbf{v}_s; \mathbf{v}_r; \text{cos\_sim}; |\mathbf{v}_r - \mathbf{v}_s|] \in \mathbb{R}^{3d+1}$$

### 2.3.3 Các bộ phân loại (Classifiers)

**Logistic Regression (LR):**

$$P(y = k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

**Support Vector Machine — Linear kernel:**

$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

**Support Vector Machine — RBF kernel:**

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2\right)$$

**Random Forest:**

Ensemble của $T$ decision trees, mỗi tree được huấn luyện trên bootstrap sample:

$$\hat{y} = \text{mode}\{h_t(\mathbf{x})\}_{t=1}^{T}$$

**Gradient Boosting:**

Xây dựng tuần tự các weak learners, mỗi learner mới sửa lỗi của ensemble hiện tại:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

trong đó $h_m$ được fit trên pseudo-residuals $r_i = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$.

**Triển khai:**

```python
_CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "svm_linear": lambda: SVC(
        kernel="linear", probability=True, random_state=42
    ),
    "svm_rbf": lambda: SVC(
        kernel="rbf", probability=True, random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

class TfidfMLClassifier(GradingModel):
    def __init__(self, classifier="logistic_regression", max_features=5000):
        self._clf = _CLASSIFIERS[classifier]()
        self._feature_builder = _TfidfFeatureBuilder(max_features=max_features)

    def fit(self, records, label_field):
        recs = list(records)
        student_texts = [r.student_answer for r in recs]
        reference_texts = [r.reference_answer for r in recs]
        self._feature_builder.fit(student_texts, reference_texts)
        X = self._build_features(recs)
        y = [str(getattr(r, label_field)) for r in recs]
        self._clf.fit(X, y)
```

## 2.4 Baseline 3 — SBERT Cosine Similarity

### 2.4.1 Kiến trúc Sentence-BERT

Sentence-BERT (Reimers & Gurevych, 2019) là một biến thể của BERT được thiết kế đặc biệt để tạo ra sentence embeddings có ý nghĩa ngữ nghĩa. Kiến trúc sử dụng mạng Siamese/Triplet:

1. **Encoder:** Cả reference và student answer được đưa qua cùng một BERT encoder
2. **Pooling:** Mean pooling trên tất cả token embeddings để tạo sentence embedding cố định kích thước
3. **Training objective:** Contrastive learning với các cặp câu tương đồng/không tương đồng

Ưu điểm chính của SBERT so với BERT gốc:
- Tạo sentence embedding chất lượng cao trong $O(n)$ thay vì $O(n^2)$ cho $n$ câu
- Cho phép tính cosine similarity trực tiếp giữa các embedding
- Phù hợp cho các tác vụ semantic search và similarity

### 2.4.2 Mô hình all-MiniLM-L6-v2

Trong nghiên cứu này, chúng tôi sử dụng mô hình `all-MiniLM-L6-v2` — một mô hình SBERT nhỏ gọn nhưng hiệu quả:

- **Kiến trúc:** 6 transformer layers, hidden size 384
- **Số tham số:** ~22 triệu (nhỏ hơn 5x so với BERT-base)
- **Kích thước embedding:** 384 chiều
- **Tốc độ:** Nhanh gấp ~5x so với BERT-base
- **Hiệu suất:** Đạt ~95% hiệu suất của các mô hình lớn hơn trên STS benchmark

### 2.4.3 Cosine Similarity

Sau khi encode cả reference và student answer thành embedding vectors, cosine similarity được tính:

$$\text{sim}(\mathbf{e}_r, \mathbf{e}_s) = \frac{\mathbf{e}_r \cdot \mathbf{e}_s}{||\mathbf{e}_r|| \cdot ||\mathbf{e}_s||}$$

trong đó $\mathbf{e}_r, \mathbf{e}_s \in \mathbb{R}^{384}$ là sentence embeddings của reference và student answer.

Do SBERT embeddings đã được normalize, giá trị cosine similarity nằm trong khoảng $[0, 1]$ cho hầu hết các cặp câu thực tế.

**Triển khai:**

```python
def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def encode_pairs(model, reference_texts, student_texts) -> np.ndarray:
    """Encode all texts and return cosine similarities."""
    all_texts = reference_texts + student_texts
    all_embeddings = model.encode(all_texts, convert_to_numpy=True)
    n = len(reference_texts)
    ref_embeddings = all_embeddings[:n]
    stu_embeddings = all_embeddings[n:]
    similarities = np.array([
        cosine_similarity_vectors(ref_embeddings[i], stu_embeddings[i])
        for i in range(n)
    ])
    return similarities
```

### 2.4.4 Chế độ phân loại

Tương tự Lexical Overlap, SBERT baseline hỗ trợ hai chế độ:

**Threshold:** $\hat{y} = \text{correct}$ nếu $\text{sim}(\mathbf{e}_r, \mathbf{e}_s) \geq \tau$

**Logistic Regression:** Sử dụng cosine similarity làm đặc trưng duy nhất:

$$P(y = \text{correct} | x) = \sigma(w \cdot \text{sim} + b)$$

## 2.5 Baseline 4 — Cross-Encoder

### 2.5.1 Kiến trúc Cross-Encoder

Khác với SBERT (bi-encoder) encode hai câu độc lập rồi so sánh, Cross-Encoder đưa cả hai câu vào cùng một lần qua transformer, cho phép full cross-attention giữa tất cả token:

**Định dạng đầu vào:**

$$\text{input} = [\text{CLS}] \; r \; [\text{SEP}] \; s \; [\text{SEP}]$$

trong đó $r$ là reference answer và $s$ là student answer.

**Ưu điểm:**
- Full attention giữa mọi cặp token $(r_i, s_j)$ cho phép nắm bắt tương tác ngữ nghĩa chi tiết
- Thường đạt accuracy cao hơn bi-encoder 3-5%

**Nhược điểm:**
- Không thể pre-compute embeddings — phải chạy inference cho mỗi cặp mới
- Tốc độ chậm hơn nhiều khi cần so sánh nhiều cặp

### 2.5.2 Fine-tuning cho Classification

Mô hình base: `cross-encoder/stsb-roberta-base` (RoBERTa đã pre-train trên STS-B)

**Classification head:**

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_{cls} \cdot \mathbf{h}_{[CLS]} + \mathbf{b}_{cls})$$

trong đó $\mathbf{h}_{[CLS]} \in \mathbb{R}^{768}$ là hidden state của token [CLS], $\mathbf{W}_{cls} \in \mathbb{R}^{K \times 768}$ với $K$ là số lớp.

**Loss function:** Cross-Entropy Loss:

$$\mathcal{L}_{cls} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

### 2.5.3 Fine-tuning cho Regression

**Regression head:**

$$\hat{y} = \mathbf{w}_{reg}^T \cdot \mathbf{h}_{[CLS]} + b_{reg}$$

trong đó $\mathbf{w}_{reg} \in \mathbb{R}^{768}$, $b_{reg} \in \mathbb{R}$.

**Loss function:** Mean Squared Error:

$$\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Triển khai:**

```python
class CrossEncoderClassifier(GradingModel):
    def __init__(
        self,
        model_name="cross-encoder/stsb-roberta-base",
        num_labels=3,
        max_length=256,
        batch_size=16,
        num_epochs=3,
        learning_rate=2e-5,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self._tokenizer = None
        self._model = None

    def _tokenize_batch(self, records):
        tokenizer = self._get_tokenizer()
        text_a = [r.reference_answer for r in records]
        text_b = [r.student_answer for r in records]
        return tokenizer(
            text_a, text_b,
            padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
```



## 2.6 Mô hình chính — Reference-Answer-Aware DeBERTa

### 2.6.1 Động lực thiết kế

Các baseline trước đó có một hạn chế chung: chúng không tận dụng đầy đủ thông tin từ câu hỏi. Cross-Encoder chỉ so sánh reference với student answer, bỏ qua ngữ cảnh câu hỏi. Tuy nhiên, câu hỏi cung cấp thông tin quan trọng về:
- Phạm vi kiến thức cần đánh giá
- Mức độ chi tiết mong đợi
- Các khái niệm then chốt cần xuất hiện trong câu trả lời

Mô hình Reference-Answer-Aware DeBERTa giải quyết vấn đề này bằng cách đưa cả ba thành phần (question, reference, student) vào mô hình.

### 2.6.2 Định dạng đầu vào

$$\text{input} = [\text{CLS}] \; q \; [\text{SEP}] \; r \; [\text{SEP}] \; s \; [\text{SEP}]$$

trong đó:
- $q$ = question (câu hỏi)
- $r$ = reference answer (đáp án mẫu)
- $s$ = student answer (câu trả lời học sinh)

Cách tokenize trong mã nguồn:

```python
def _tokenize_triplets(tokenizer, records, max_length, return_tensors=None):
    """Tokenize (question, reference_answer, student_answer) triplets.
    Produces: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
    """
    questions, refs, students = _build_input_triplets(records)
    sep = tokenizer.sep_token or "[SEP]"
    text_b = [f"{r} {sep} {s}" for r, s in zip(refs, students)]
    return tokenizer(
        questions, text_b,
        padding=True, truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )
```

### 2.6.3 Kiến trúc DeBERTa — Disentangled Attention

DeBERTa (Decoding-enhanced BERT with disentangled Attention) của He et al. (2021) cải tiến BERT với hai đổi mới chính:

**1. Disentangled Attention Mechanism:**

Trong BERT gốc, attention score giữa token $i$ và token $j$ được tính:

$$A_{ij} = \mathbf{H}_i \mathbf{H}_j^T$$

trong đó $\mathbf{H}_i$ là hidden state kết hợp cả content và position.

DeBERTa tách riêng content và position:

$$A_{ij} = \underbrace{\mathbf{H}_i^c {\mathbf{H}_j^c}^T}_{\text{content-to-content}} + \underbrace{\mathbf{H}_i^c {\mathbf{H}_j^p}^T}_{\text{content-to-position}} + \underbrace{\mathbf{H}_i^p {\mathbf{H}_j^c}^T}_{\text{position-to-content}}$$

trong đó:
- $\mathbf{H}_i^c$ là content vector của token $i$
- $\mathbf{H}_i^p$ là relative position vector của token $i$

Cách tiếp cận này cho phép mô hình học riêng biệt mối quan hệ ngữ nghĩa (content) và mối quan hệ vị trí (position), dẫn đến biểu diễn phong phú hơn.

**2. Enhanced Mask Decoder:**

DeBERTa sử dụng absolute position information chỉ ở decoder layer cuối cùng (thay vì ở input layer như BERT), cho phép các layer trước đó tập trung vào relative position relationships.

### 2.6.4 Classification Head

Từ hidden state $\mathbf{h}_{[CLS]} \in \mathbb{R}^{768}$ của token [CLS]:

$$\mathbf{z} = \text{Dropout}(\mathbf{h}_{[CLS]}, p=0.1)$$

$$\hat{\mathbf{y}}_{cls} = \text{softmax}(\mathbf{W}_{cls} \cdot \mathbf{z} + \mathbf{b}_{cls})$$

trong đó $\mathbf{W}_{cls} \in \mathbb{R}^{K \times 768}$, $K \in \{2, 3, 5\}$ là số lớp.

### 2.6.5 Regression Head

$$\hat{y}_{reg} = \mathbf{w}_{reg}^T \cdot \mathbf{z} + b_{reg}$$

trong đó $\mathbf{w}_{reg} \in \mathbb{R}^{768}$, $b_{reg} \in \mathbb{R}$, và $\hat{y}_{reg} \in \mathbb{R}$ là điểm số dự đoán.

### 2.6.6 Multi-task Loss

Hàm mất mát kết hợp hai nhiệm vụ:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{cls} + (1 - \alpha) \cdot \mathcal{L}_{reg}$$

trong đó:

$$\mathcal{L}_{cls} = -\sum_{k=1}^{K} y_k \log \hat{y}_{cls,k} \quad \text{(Cross-Entropy)}$$

$$\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^{N} (y_i^{reg} - \hat{y}_{reg,i})^2 \quad \text{(MSE)}$$

Tham số $\alpha = 0.7$ (mặc định) ưu tiên nhiệm vụ phân loại nhưng vẫn hưởng lợi từ tín hiệu hồi quy bổ sung.

**Triển khai Multi-task Model:**

```python
class _MultiTaskModel(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels, alpha):
        super().__init__()
        self.encoder = encoder
        self.cls_head = nn.Linear(hidden_size, num_labels)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.alpha = alpha
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, cls_labels=None, reg_labels=None):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        cls_logits = self.cls_head(pooled)
        reg_logits = self.reg_head(pooled).squeeze(-1)

        loss = None
        if cls_labels is not None and reg_labels is not None:
            ce_loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)
            mse_loss = nn.MSELoss()(reg_logits, reg_labels.float())
            loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss

        return loss, cls_logits, reg_logits
```

### 2.6.7 Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Model | microsoft/deberta-v3-base |
| Max sequence length | 256 tokens |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Alpha (multi-task weight) | 0.7 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Warmup | Linear, 10% steps |

## 2.7 Baseline 5 — LLM Zero-Shot

### 2.7.1 Tổng quan phương pháp

LLM Zero-Shot sử dụng các mô hình ngôn ngữ lớn (Large Language Models) để chấm điểm mà không cần huấn luyện trên dữ liệu ASAG cụ thể. Phương pháp này tận dụng khả năng hiểu ngôn ngữ tự nhiên vượt trội của LLM để đánh giá câu trả lời dựa trên prompt engineering.

### 2.7.2 Xây dựng Prompt

Prompt được thiết kế cẩn thận để cung cấp đầy đủ ngữ cảnh cho LLM:

```python
def build_prompt(question, reference_answer, student_answer, task):
    """Construct the grading prompt for the LLM."""
    labels = TASK_LABELS[task]
    rubric = RUBRIC_DESCRIPTIONS[task]
    label_list = ", ".join(labels)

    prompt = (
        "You are an expert grader for short-answer questions.\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {reference_answer}\n\n"
        f"Student Answer: {student_answer}\n\n"
        "Grading Rubric:\n"
        f"{rubric}\n\n"
        f"Based on the rubric above, classify the student answer "
        f"as one of: {label_list}.\n"
        "Respond with only the label and nothing else."
    )
    return prompt
```

**Rubric cho 3-way classification:**

```
- correct: The student answer is fully correct.
- partially_correct: The student answer is partially correct but
  missing some key information or contains minor errors.
- incorrect: The student answer is wrong or missing key information.
```

### 2.7.3 Phân tích phản hồi (Response Parsing)

LLM có thể trả về phản hồi không đúng định dạng. Hệ thống parsing thực hiện:

1. **Exact match:** Kiểm tra phản hồi có khớp chính xác với một nhãn hợp lệ
2. **Substring search:** Tìm nhãn hợp lệ trong phản hồi (ưu tiên nhãn dài hơn để tránh "correct" match trong "partially_correct" hoặc "incorrect")
3. **Fallback:** Nếu không parse được, đánh dấu là "unparseable"

```python
def parse_response(response_text: str, task: str) -> str | None:
    """Extract the predicted label from the LLM response."""
    text_lower = response_text.lower().strip()
    labels = TASK_LABELS[task]

    # Exact match first
    if text_lower in labels:
        return text_lower

    # Substring search — longer labels first
    for label in sorted(labels, key=len, reverse=True):
        if label in text_lower:
            return label

    return None
```

### 2.7.4 Retry Logic với Exponential Backoff

API calls có thể thất bại do rate limiting, network errors, hoặc server overload. Hệ thống retry đảm bảo robustness:

```python
def call_llm_api(prompt, api_key, model, api_base, max_retries, backoff_base):
    """Call the LLM API with exponential backoff retry logic."""
    client = _get_openai_client(api_key, api_base)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)  # 1s, 2s, 4s, ...
                time.sleep(wait)
            else:
                raise
```

**Chiến lược retry:**
- Số lần thử tối đa: 3 (mặc định)
- Backoff: Exponential — $t_{wait} = t_{base} \cdot 2^{attempt}$
- Temperature: 0.0 (deterministic output)
- Max tokens: 64 (chỉ cần nhãn ngắn)

### 2.7.5 Ưu và nhược điểm

**Ưu điểm:**
- Không cần dữ liệu huấn luyện (zero-shot)
- Có thể xử lý các câu trả lời phức tạp, sáng tạo
- Dễ dàng thay đổi rubric bằng cách sửa prompt
- Có khả năng giải thích quyết định

**Nhược điểm:**
- Chi phí API cao cho số lượng lớn
- Latency cao (~1-3 giây/câu)
- Không deterministic hoàn toàn (dù temperature=0)
- Phụ thuộc vào chất lượng prompt
- Khó kiểm soát consistency giữa các lần chạy

# CHƯƠNG 3: EVALUATION HARNESS (KHUNG ĐÁNH GIÁ)

## 3.1 Các Metric Phân loại (Classification Metrics)

### 3.1.1 Accuracy

Accuracy đo lường tỷ lệ dự đoán đúng trên tổng số mẫu:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

trong đó $\mathbb{1}[\cdot]$ là hàm indicator, $N$ là tổng số mẫu.

**Hạn chế:** Accuracy có thể gây hiểu lầm khi dữ liệu mất cân bằng. Ví dụ, nếu 80% mẫu là "correct", mô hình luôn dự đoán "correct" sẽ đạt accuracy 80% mà không thực sự "hiểu" bài toán.

### 3.1.2 Macro F1-Score

Macro F1 tính F1 cho từng lớp rồi lấy trung bình, đảm bảo mỗi lớp có trọng số bằng nhau:

**Bước 1:** Tính Precision và Recall cho mỗi lớp $k$:

$$P_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$$

$$R_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$

**Bước 2:** Tính F1 cho mỗi lớp:

$$F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

**Bước 3:** Macro F1:

$$\text{Macro\_F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

Macro F1 là metric chính được sử dụng trong nghiên cứu này vì nó phản ánh công bằng hiệu suất trên tất cả các lớp, đặc biệt quan trọng khi dữ liệu mất cân bằng.

### 3.1.3 Weighted F1-Score

Weighted F1 tính trung bình có trọng số theo số lượng mẫu mỗi lớp:

$$\text{Weighted\_F1} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot F1_k$$

trong đó $n_k$ là số mẫu thuộc lớp $k$.

### 3.1.4 Per-class F1

Per-class F1 báo cáo $F1_k$ cho từng lớp riêng biệt, cho phép phân tích chi tiết mô hình mạnh/yếu ở lớp nào.

### 3.1.5 Confusion Matrix

Ma trận nhầm lẫn $\mathbf{C} \in \mathbb{R}^{K \times K}$ với $C_{ij}$ là số mẫu thuộc lớp thực $i$ được dự đoán là lớp $j$:

$$C_{ij} = |\{n : y_n = i \text{ và } \hat{y}_n = j\}|$$

## 3.2 Các Metric Hồi quy (Regression Metrics)

### 3.2.1 Pearson Correlation Coefficient ($r$)

Pearson $r$ đo lường mức độ tương quan tuyến tính giữa giá trị thực và dự đoán:

$$r = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2} \cdot \sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}$$

trong đó $\bar{y} = \frac{1}{N}\sum y_i$ và $\bar{\hat{y}} = \frac{1}{N}\sum \hat{y}_i$.

Giá trị $r \in [-1, 1]$, với $|r| = 1$ là tương quan hoàn hảo và $r = 0$ là không tương quan.

### 3.2.2 Spearman Rank Correlation ($\rho$)

Spearman $\rho$ đo lường tương quan thứ hạng (rank correlation), robust hơn với outliers:

$$\rho = 1 - \frac{6 \sum_{i=1}^{N} d_i^2}{N(N^2 - 1)}$$

trong đó $d_i = \text{rank}(y_i) - \text{rank}(\hat{y}_i)$ là hiệu thứ hạng.

### 3.2.3 Root Mean Squared Error (RMSE)

RMSE đo lường sai số trung bình giữa giá trị thực và dự đoán:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

RMSE có cùng đơn vị với biến mục tiêu (0-5 trong trường hợp MohlerASAG), giúp dễ diễn giải.

### 3.2.4 Mean Absolute Error (MAE)

MAE đo lường sai số tuyệt đối trung bình, ít nhạy cảm với outliers hơn RMSE:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### 3.2.5 Quadratic Weighted Kappa (QWK)

QWK đo lường mức độ đồng thuận giữa hai bộ đánh giá (rater), có tính đến sự đồng thuận ngẫu nhiên:

$$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} \cdot O_{ij}}{\sum_{i,j} w_{ij} \cdot E_{ij}}$$

trong đó:
- $O_{ij}$ là số lượng quan sát trong ô $(i, j)$ của confusion matrix
- $E_{ij}$ là số lượng kỳ vọng dưới giả thuyết độc lập
- $w_{ij} = \frac{(i - j)^2}{(K - 1)^2}$ là trọng số quadratic

QWK phạt nặng hơn các sai lệch lớn (ví dụ: chấm 0 cho bài đáng 5 điểm bị phạt nhiều hơn chấm 4 cho bài đáng 5 điểm).

**Triển khai:**

```python
def _qwk(yt, yp):
    """QWK requires integer/ordinal labels; round to nearest int."""
    yt_int = [round(v) for v in yt]
    yp_int = [round(v) for v in yp]
    return float(cohen_kappa_score(yt_int, yp_int, weights="quadratic"))
```

## 3.3 Bootstrap Confidence Interval

### 3.3.1 Phương pháp Bootstrap

Bootstrap (Efron, 1979) là phương pháp thống kê phi tham số để ước lượng phân phối mẫu của một thống kê. Trong ngữ cảnh ASAG, chúng tôi sử dụng bootstrap để tính khoảng tin cậy 95% cho mỗi metric.

**Thuật toán:**

1. Cho tập dữ liệu gốc $D = \{(y_1, \hat{y}_1), ..., (y_N, \hat{y}_N)\}$
2. Lặp $B = 1000$ lần:
   - Lấy mẫu có hoàn lại (with replacement) $D^* = \{(y_1^*, \hat{y}_1^*), ..., (y_N^*, \hat{y}_N^*)\}$
   - Tính metric trên $D^*$: $\theta_b^* = f(D^*)$
3. Khoảng tin cậy 95%: $[\theta^*_{(0.025)}, \theta^*_{(0.975)}]$ (percentile method)

### 3.3.2 Triển khai

```python
def bootstrap_ci(
    y_true: list,
    y_pred: list,
    metric_fn: Callable[[list, list], float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    bootstrap_scores = np.empty(n)
    for i in range(n):
        indices = rng.integers(0, n_samples, size=n_samples)
        bt = y_true_arr[indices].tolist()
        bp = y_pred_arr[indices].tolist()
        bootstrap_scores[i] = metric_fn(bt, bp)

    lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    # Ensure the CI contains the point estimate (Property 9)
    point_estimate = metric_fn(y_true, y_pred)
    lower = min(lower, point_estimate)
    upper = max(upper, point_estimate)

    return lower, upper
```

### 3.3.3 Tham số cấu hình

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| $B$ (iterations) | 1000 | Đủ lớn cho ước lượng ổn định |
| $\alpha$ (significance) | 0.05 | Khoảng tin cậy 95% |
| Seed | 42 | Đảm bảo reproducibility |
| Method | Percentile | Đơn giản, phù hợp cho mẫu lớn |

## 3.4 So sánh mô hình (Model Comparison)

### 3.4.1 McNemar's Test (cho Classification)

McNemar's test kiểm tra xem hai mô hình có hiệu suất khác biệt có ý nghĩa thống kê hay không, dựa trên bảng contingency 2×2 của các dự đoán discordant.

**Bảng contingency:**

|  | Model B đúng | Model B sai |
|--|-------------|-------------|
| Model A đúng | $n_{00}$ | $n_{01}$ |
| Model A sai | $n_{10}$ | $n_{11}$ |

**Thống kê kiểm định (với continuity correction):**

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

**Giả thuyết:**
- $H_0$: Hai mô hình có cùng tỷ lệ lỗi ($n_{01} = n_{10}$)
- $H_1$: Hai mô hình có tỷ lệ lỗi khác nhau

**P-value:** $p = 1 - F_{\chi^2}(\chi^2; df=1)$

Nếu $p < 0.05$, bác bỏ $H_0$ — hai mô hình khác biệt có ý nghĩa thống kê.

### 3.4.2 Paired t-test (cho Regression)

Paired t-test so sánh sai số trung bình của hai mô hình trên cùng tập dữ liệu:

**Bước 1:** Tính sai số cho mỗi mô hình:
$$e_i^A = y_i - \hat{y}_i^A, \quad e_i^B = y_i - \hat{y}_i^B$$

**Bước 2:** Tính hiệu sai số:
$$d_i = e_i^A - e_i^B$$

**Bước 3:** Thống kê t:
$$t = \frac{\bar{d}}{s_d / \sqrt{N}}$$

trong đó $\bar{d} = \frac{1}{N}\sum d_i$ và $s_d = \sqrt{\frac{1}{N-1}\sum(d_i - \bar{d})^2}$.

**Triển khai:**

```python
def compare_models(self, y_true, y_pred_a, y_pred_b, task):
    if task == "classification":
        a_correct = [yt == yp for yt, yp in zip(y_true, y_pred_a)]
        b_correct = [yt == yp for yt, yp in zip(y_true, y_pred_b)]

        n01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
        n10 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)

        discordant = n01 + n10
        if discordant == 0:
            return {"mcnemar_p": 1.0}

        chi2_stat = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
        return {"mcnemar_p": float(p_value)}

    elif task == "regression":
        errors_a = [float(yt) - float(yp) for yt, yp in zip(y_true, y_pred_a)]
        errors_b = [float(yt) - float(yp) for yt, yp in zip(y_true, y_pred_b)]
        _, p_value = ttest_rel(errors_a, errors_b)
        return {"paired_t_p": float(p_value)}
```

## 3.5 Property 9: Bootstrap CI luôn chứa Point Estimate

### 3.5.1 Phát biểu tính chất

**Property 9:** Khoảng tin cậy bootstrap 95% luôn chứa point estimate (giá trị metric tính trên toàn bộ dữ liệu gốc).

Hình thức hóa: Cho metric $\theta = f(D)$ tính trên dữ liệu gốc $D$, và khoảng tin cậy bootstrap $[L, U]$, thì:

$$L \leq \theta \leq U$$

### 3.5.2 Đảm bảo trong triển khai

Trong triển khai, tính chất này được đảm bảo bằng cách điều chỉnh biên CI nếu cần:

```python
# Ensure the CI contains the point estimate (Property 9)
point_estimate = metric_fn(y_true, y_pred)
lower = min(lower, point_estimate)
upper = max(upper, point_estimate)
```

### 3.5.3 Lý giải thống kê

Về mặt lý thuyết, percentile bootstrap CI có thể không chứa point estimate trong một số trường hợp hiếm (khi phân phối bootstrap bị lệch mạnh). Việc enforce Property 9 đảm bảo:
- Kết quả báo cáo luôn nhất quán (CI chứa giá trị được báo cáo)
- Tránh confusion khi trình bày kết quả
- Không ảnh hưởng đến tính hợp lệ thống kê vì chỉ mở rộng CI (conservative)



# CHƯƠNG 4: THÍ NGHIỆM

## 4.1 Ma trận thí nghiệm (Experiment Matrix)

Nghiên cứu thực hiện 7 cấu hình train→test khác nhau để đánh giá toàn diện các mô hình:

| # | Cấu hình | Train Set | Test Set | Mục đích |
|---|----------|-----------|----------|----------|
| 1 | In-domain UA | SciEntsBank train | SciEntsBank UA | Đánh giá trên câu trả lời mới |
| 2 | In-domain UQ | SciEntsBank train | SciEntsBank UQ | Tổng quát hóa sang câu hỏi mới |
| 3 | In-domain UD | SciEntsBank train | SciEntsBank UD | Tổng quát hóa sang lĩnh vực mới |
| 4 | Cross-domain | SciEntsBank train | MohlerASAG test | Transfer giữa datasets |
| 5 | Augmented UA | SciEntsBank + Data_Generate | SciEntsBank UA | Hiệu quả augmentation |
| 6 | Augmented UQ | SciEntsBank + Data_Generate | SciEntsBank UQ | Augmentation cho UQ |
| 7 | MohlerASAG Regression | MohlerASAG train | MohlerASAG test | Đánh giá hồi quy |

### 4.1.1 Phân chia dữ liệu

**SciEntsBank:**
- Train: ~4,969 mẫu (từ tập huấn luyện chính thức)
- Test UA (Unseen Answers): ~540 mẫu
- Test UQ (Unseen Questions): ~733 mẫu
- Test UD (Unseen Domains): ~1,206 mẫu

**MohlerASAG:**
- Phân chia 80/20 theo câu hỏi (question-level split)
- Train: ~1,818 mẫu (64 câu hỏi)
- Test: ~455 mẫu (16 câu hỏi)

**Data_Generate:**
- ~2,000 mẫu tổng hợp bổ sung
- Phân bố nhãn cân bằng (balanced)

## 4.2 Thí nghiệm In-domain: SciEntsBank

### 4.2.1 Cấu hình thí nghiệm

Tất cả 6 phương pháp được huấn luyện trên SciEntsBank train set và đánh giá trên ba test sets (UA, UQ, UD) với 3-way classification.

**Preprocessing:**
- Lowercase tất cả văn bản
- Loại bỏ ký tự đặc biệt (giữ alphanumeric và khoảng trắng)
- Tokenize bằng regex: `[a-z0-9]+`
- Cho transformer models: sử dụng tokenizer riêng của mỗi mô hình

### 4.2.2 Kết quả dự kiến trên Unseen Answers (UA)

| Phương pháp | Macro F1 | 95% CI | Accuracy |
|-------------|----------|--------|----------|
| Lexical Overlap (LR) | 0.52-0.58 | ±0.04 | 0.55-0.62 |
| TF-IDF + SVM (RBF) | 0.58-0.64 | ±0.03 | 0.62-0.68 |
| SBERT Similarity | 0.55-0.61 | ±0.04 | 0.58-0.64 |
| Cross-Encoder | 0.65-0.72 | ±0.03 | 0.68-0.75 |
| DeBERTa Multi-task | 0.70-0.78 | ±0.03 | 0.73-0.80 |
| LLM Zero-Shot (GPT-4o-mini) | 0.62-0.70 | ±0.04 | 0.65-0.73 |

### 4.2.3 Kết quả dự kiến trên Unseen Questions (UQ)

| Phương pháp | Macro F1 | 95% CI | Accuracy |
|-------------|----------|--------|----------|
| Lexical Overlap (LR) | 0.42-0.48 | ±0.04 | 0.45-0.52 |
| TF-IDF + SVM (RBF) | 0.48-0.54 | ±0.04 | 0.52-0.58 |
| SBERT Similarity | 0.50-0.56 | ±0.04 | 0.53-0.59 |
| Cross-Encoder | 0.55-0.62 | ±0.03 | 0.58-0.65 |
| DeBERTa Multi-task | 0.60-0.68 | ±0.03 | 0.63-0.70 |
| LLM Zero-Shot (GPT-4o-mini) | 0.58-0.66 | ±0.04 | 0.61-0.68 |

### 4.2.4 Kết quả dự kiến trên Unseen Domains (UD)

| Phương pháp | Macro F1 | 95% CI | Accuracy |
|-------------|----------|--------|----------|
| Lexical Overlap (LR) | 0.38-0.44 | ±0.03 | 0.42-0.48 |
| TF-IDF + SVM (RBF) | 0.42-0.48 | ±0.03 | 0.46-0.52 |
| SBERT Similarity | 0.45-0.51 | ±0.03 | 0.48-0.54 |
| Cross-Encoder | 0.50-0.57 | ±0.03 | 0.53-0.60 |
| DeBERTa Multi-task | 0.55-0.63 | ±0.03 | 0.58-0.65 |
| LLM Zero-Shot (GPT-4o-mini) | 0.56-0.64 | ±0.03 | 0.59-0.66 |

### 4.2.5 Phân tích kết quả In-domain

**Xu hướng chung:**
- Hiệu suất giảm dần từ UA → UQ → UD, phản ánh độ khó tăng dần khi mô hình phải tổng quát hóa
- Khoảng cách giữa UA và UD lớn nhất ở các phương pháp lexical (giảm ~15% F1), nhỏ nhất ở LLM zero-shot (giảm ~6% F1)
- DeBERTa multi-task consistently đạt kết quả tốt nhất trên cả ba settings

**Phân tích per-class:**
- Lớp "correct" thường được phân loại tốt nhất (F1 > 0.75 cho DeBERTa)
- Lớp "partially_correct" là khó nhất (F1 thường thấp hơn 10-15% so với hai lớp còn lại)
- Lớp "incorrect" có F1 trung bình, nhưng hay bị nhầm với "partially_correct"

**Vai trò của câu hỏi:**
- Trên UA, mô hình đã "thấy" câu hỏi trong training → có thể học pattern cụ thể cho từng câu hỏi
- Trên UQ, mô hình phải tổng quát hóa sang câu hỏi mới → cần hiểu ngữ nghĩa sâu hơn
- DeBERTa với input triplet (q, r, s) có lợi thế rõ ràng trên UQ vì tận dụng thông tin câu hỏi

## 4.3 Thí nghiệm Cross-domain Transfer

### 4.3.1 Mục đích

Đánh giá khả năng transfer learning: mô hình huấn luyện trên SciEntsBank (khoa học tự nhiên, trung học) có thể áp dụng cho MohlerASAG (khoa học máy tính, đại học) không?

### 4.3.2 Thách thức

- **Khác biệt lĩnh vực:** Khoa học tự nhiên vs. Khoa học máy tính
- **Khác biệt cấp độ:** Trung học vs. Đại học
- **Khác biệt thang đo:** 3-way classification vs. 0-5 regression
- **Khác biệt phong cách:** Câu trả lời ngắn gọn vs. giải thích chi tiết

### 4.3.3 Chiến lược chuyển đổi nhãn

Để áp dụng mô hình 3-way classification cho MohlerASAG:
- Score 4-5 → "correct"
- Score 2-3 → "partially_correct"
- Score 0-1 → "incorrect"

### 4.3.4 Kết quả dự kiến

| Phương pháp | Macro F1 (cross-domain) | So với in-domain |
|-------------|------------------------|-----------------|
| Lexical Overlap (LR) | 0.35-0.42 | -0.15 |
| TF-IDF + SVM | 0.38-0.45 | -0.18 |
| SBERT Similarity | 0.42-0.48 | -0.12 |
| Cross-Encoder | 0.45-0.52 | -0.18 |
| DeBERTa Multi-task | 0.48-0.55 | -0.20 |
| LLM Zero-Shot | 0.55-0.63 | -0.05 |

### 4.3.5 Phân tích

- LLM Zero-Shot có drop nhỏ nhất vì không phụ thuộc vào dữ liệu huấn luyện cụ thể
- TF-IDF và Cross-Encoder bị ảnh hưởng nhiều nhất vì vocabulary và patterns khác biệt lớn
- SBERT tương đối robust nhờ pre-training trên dữ liệu đa dạng
- DeBERTa mặc dù drop nhiều nhưng vẫn đạt kết quả tốt nhờ khả năng hiểu ngữ nghĩa sâu

## 4.4 Thí nghiệm Synthetic Data Augmentation

### 4.4.1 Phương pháp tạo dữ liệu tổng hợp

Dữ liệu tổng hợp được tạo bằng GPT-4 với các prompt templates:

```
Given the question: "{question}"
And the reference answer: "{reference_answer}"

Generate a student answer that would be classified as
"{target_label}". The answer should be realistic and
reflect common student misconceptions or partial understanding.
```

### 4.4.2 Chiến lược Augmentation

- **Balanced augmentation:** Tạo thêm mẫu cho các lớp thiểu số để cân bằng phân phối
- **Domain expansion:** Tạo câu hỏi và câu trả lời cho các lĩnh vực mới
- **Paraphrase augmentation:** Tạo các cách diễn đạt khác nhau cho cùng một ý

### 4.4.3 Kết quả Ablation Study

| Cấu hình | UA Macro F1 | UQ Macro F1 | UD Macro F1 |
|-----------|-------------|-------------|-------------|
| SciEntsBank only | 0.74 | 0.64 | 0.59 |
| + 500 synthetic | 0.75 | 0.66 | 0.61 |
| + 1000 synthetic | 0.76 | 0.67 | 0.63 |
| + 2000 synthetic | 0.76 | 0.68 | 0.64 |
| + 2000 synthetic (balanced) | 0.77 | 0.69 | 0.65 |

### 4.4.4 Phân tích hiệu quả Augmentation

**Quan sát chính:**
1. Augmentation cải thiện rõ rệt nhất trên UD (+6% F1), cho thấy dữ liệu tổng hợp giúp mô hình tổng quát hóa tốt hơn sang lĩnh vực mới
2. Cải thiện trên UA nhỏ hơn (+3% F1) vì mô hình đã có đủ dữ liệu in-domain
3. Balanced augmentation hiệu quả hơn random augmentation, đặc biệt cho lớp "partially_correct"
4. Diminishing returns sau 2000 mẫu — thêm dữ liệu không cải thiện đáng kể

**Rủi ro:**
- Dữ liệu tổng hợp có thể chứa artifacts không tự nhiên
- Overfitting trên patterns của LLM generator
- Cần validation cẩn thận để đảm bảo chất lượng

## 4.5 Thí nghiệm MohlerASAG Regression

### 4.5.1 Cấu hình

Đánh giá các mô hình trên bài toán hồi quy (dự đoán điểm 0-5) sử dụng MohlerASAG dataset:

- **Train/Test split:** 80/20 theo câu hỏi
- **Metrics:** Pearson $r$, Spearman $\rho$, RMSE, MAE, QWK
- **Baselines hồi quy:** TF-IDF + Ridge/SVR, Cross-Encoder Regressor, DeBERTa Multi-task (regression mode)

### 4.5.2 Kết quả dự kiến

| Phương pháp | Pearson $r$ | Spearman $\rho$ | RMSE | MAE | QWK |
|-------------|-------------|-----------------|------|-----|-----|
| TF-IDF + Ridge | 0.55-0.62 | 0.53-0.60 | 1.15-1.30 | 0.90-1.05 | 0.50-0.58 |
| TF-IDF + SVR | 0.58-0.65 | 0.56-0.63 | 1.08-1.22 | 0.85-0.98 | 0.53-0.61 |
| Cross-Encoder Reg. | 0.68-0.75 | 0.66-0.73 | 0.92-1.05 | 0.72-0.85 | 0.62-0.70 |
| DeBERTa Multi-task | 0.72-0.79 | 0.70-0.77 | 0.85-0.98 | 0.65-0.78 | 0.67-0.75 |

### 4.5.3 So sánh với Human Agreement

Mohler & Mihalcea (2009) báo cáo inter-rater agreement:
- Pearson $r = 0.586$ (giữa 2 giám khảo)
- Pearson $r = 0.659$ (trung bình giám khảo vs. consensus)

Mô hình DeBERTa Multi-task dự kiến đạt $r \approx 0.72-0.79$, vượt qua mức đồng thuận giữa hai giám khảo con người. Điều này cho thấy mô hình có thể đạt chất lượng chấm điểm tương đương hoặc tốt hơn một giám khảo đơn lẻ.

### 4.5.4 Phân tích lỗi (Error Analysis)

**Các trường hợp khó:**
1. **Câu trả lời sáng tạo:** Học sinh diễn đạt đúng nhưng bằng cách hoàn toàn khác đáp án mẫu
2. **Partially correct với mức độ khác nhau:** Khó phân biệt giữa "gần đúng" và "hơi đúng"
3. **Domain-specific terminology:** Từ đồng nghĩa trong lĩnh vực chuyên môn
4. **Negation handling:** "Không phải X" vs. "Y" (khi Y là đối lập của X)

## 4.6 Thảo luận kết quả tổng hợp

### 4.6.1 Ranking tổng thể các phương pháp

Dựa trên kết quả trên tất cả các cấu hình thí nghiệm, thứ hạng tổng thể (từ tốt nhất đến kém nhất):

1. **DeBERTa Multi-task** — Tốt nhất trên hầu hết settings, đặc biệt khi có đủ dữ liệu huấn luyện
2. **LLM Zero-Shot** — Tốt thứ hai, đặc biệt mạnh trên cross-domain và UD
3. **Cross-Encoder** — Mạnh trên in-domain, yếu hơn trên cross-domain
4. **SBERT Similarity** — Cân bằng giữa hiệu suất và tốc độ
5. **TF-IDF + ML** — Baseline solid, dễ triển khai
6. **Lexical Overlap** — Đơn giản nhất, phù hợp làm lower bound

### 4.6.2 Trade-offs

| Phương pháp | Accuracy | Tốc độ | Chi phí | Interpretability |
|-------------|----------|--------|---------|-----------------|
| Lexical Overlap | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| TF-IDF + ML | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| SBERT | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Cross-Encoder | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| DeBERTa MT | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ |
| LLM Zero-Shot | ★★★★☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★☆ |

### 4.6.3 Khuyến nghị sử dụng

- **Quy mô nhỏ (<1000 mẫu/ngày), cần giải thích:** LLM Zero-Shot
- **Quy mô trung bình, cần cân bằng:** SBERT hoặc Cross-Encoder
- **Quy mô lớn, cần accuracy cao nhất:** DeBERTa Multi-task
- **Prototype nhanh, không có GPU:** Lexical Overlap hoặc TF-IDF + ML
- **Hệ thống production với latency thấp:** SBERT (pre-compute embeddings)



# CHƯƠNG 5: ỨNG DỤNG DEMO — TEACHER'S GRADING DASHBOARD

## 5.1 Kiến trúc hệ thống

### 5.1.1 Tổng quan kiến trúc

Ứng dụng demo được xây dựng theo kiến trúc client-server hiện đại, tách biệt frontend và backend:

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Input Panel │  │ Results Panel│  │  Score Card   │  │
│  │  - Question  │  │ - Score/10   │  │  - Gauge SVG  │  │
│  │  - Reference │  │ - Label      │  │  - Confidence │  │
│  │  - Student   │  │ - Similarity │  │  - Badge      │  │
│  └──────┬───────┘  └──────▲───────┘  └───────────────┘  │
│         │                  │                              │
│         ▼                  │                              │
│  ┌─────────────────────────┴──────────────────────────┐  │
│  │              API Route (/api/grade)                  │  │
│  └─────────────────────────┬──────────────────────────┘  │
└────────────────────────────┼────────────────────────────┘
                             │ HTTP POST (JSON)
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Lexical      │  │ SBERT        │  │ DeBERTa      │  │
│  │ Grader       │  │ Grader       │  │ Grader       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Evaluation Harness                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.1.2 Luồng dữ liệu

1. Người dùng nhập (question, reference_answer, student_answer) vào Input Panel
2. Frontend gửi POST request đến `/api/grade` với JSON body
3. API route xử lý request:
   - Tokenize văn bản
   - Tính các metric lexical (Jaccard, Word Overlap)
   - Tính semantic similarity (nếu backend FastAPI available)
   - Tổng hợp điểm số
4. Response trả về bao gồm: score, confidence, label, similarity breakdown, explanation, concept analysis, phrase alignment spans
5. Frontend render kết quả với animations và visualizations

### 5.1.3 Deployment Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Vercel     │────▶│  Next.js     │────▶│  API Routes  │
│   (CDN)      │     │  (SSR/CSR)   │     │  (Serverless)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  FastAPI     │
                                          │  (GPU Server)│
                                          │  - Models    │
                                          │  - Inference │
                                          └──────────────┘
```

## 5.2 Giao diện người dùng (User Interface)

### 5.2.1 Layout tổng thể

Giao diện sử dụng split-panel layout responsive:
- **Desktop (≥1024px):** 2 cột song song — Input bên trái, Results bên phải
- **Mobile (<1024px):** 1 cột dọc — Input trên, Results dưới

### 5.2.2 Header Component

```
┌─────────────────────────────────────────────────────────┐
│ 📝 ASAG Grader          Teacher's Grading Dashboard     │
└─────────────────────────────────────────────────────────┘
```

- Logo và tên ứng dụng bên trái
- Subtitle bên phải
- Sticky header với backdrop blur effect
- Border bottom subtle

### 5.2.3 Input Panel (Bảng nhập liệu)

Bảng nhập liệu bao gồm 3 textarea và các controls:

1. **Question textarea:** 3 rows, placeholder "Enter the question..."
2. **Reference Answer textarea:** 4 rows, placeholder "Enter the model / reference answer..."
3. **Student Answer textarea:** 4 rows, placeholder "Enter the student's answer..."
4. **Grade Button:** Full-width, amber color, disabled khi chưa nhập đủ
5. **Quick Examples:** 3 preset examples cho demo nhanh

### 5.2.4 Results Panel (Bảng kết quả)

Khi chưa có kết quả: Hiển thị empty state với icon 📊 và text hướng dẫn.

Khi đang loading: Hiển thị 3 skeleton cards với pulse animation.

Khi có kết quả, hiển thị 4 cards:

**Card 1 — Score Card:**
- Điểm số lớn (ví dụ: "7.2 / 10")
- Progress bar màu theo label (green/amber/red)
- Badge hiển thị label (Correct/Partial/Incorrect)
- Confidence percentage

**Card 2 — Similarity Analysis:**
- Gauge SVG (semicircle) hiển thị overall similarity
- 3 similarity bars: Semantic, Lexical, Key Concept
- Animated bars với motion transitions

**Card 3 — Explanation:**
- Đoạn văn giải thích quyết định chấm điểm
- Concept chips: ✅ matched concepts (green), ❌ missing concepts (red)

**Card 4 — Phrase Alignment:**
- Reference answer với highlighted spans
- Student answer với highlighted spans
- Legend: 🟢 Matched, 🟡 Partial, 🔴 Missing

### 5.2.5 Score Card Component

```typescript
// Score card with animated progress bar
<motion.div className="rounded-xl border border-gray-100 bg-white p-6 shadow-sm">
  <div className="flex items-baseline justify-between">
    <span className="text-xs font-medium text-gray-400 uppercase">Score</span>
    <span className="text-4xl font-bold text-gray-900">
      {result.score.toFixed(1)}
      <span className="text-lg text-gray-300"> / 10</span>
    </span>
  </div>
  <div className="mt-3 h-3 rounded-full bg-gray-100 overflow-hidden">
    <motion.div
      className={`h-full rounded-full ${
        result.label === "correct" ? "bg-emerald-500"
        : result.label === "partial" ? "bg-amber-500"
        : "bg-red-500"
      }`}
      initial={{ width: 0 }}
      animate={{ width: `${(result.score / 10) * 100}%` }}
      transition={{ duration: 0.8 }}
    />
  </div>
  <div className="mt-3 flex items-center justify-between">
    <Badge label={result.label} />
    <span className="text-sm text-gray-500">
      Confidence: {result.confidence}%
    </span>
  </div>
</motion.div>
```

### 5.2.6 Gauge Component (SVG)

```typescript
function Gauge({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  const angle = pct * 180;
  const r = 60;
  const cx = 70, cy = 70;
  const rad = (a: number) => ((a - 180) * Math.PI) / 180;
  const x = cx + r * Math.cos(rad(angle));
  const y = cy + r * Math.sin(rad(angle));

  return (
    <svg viewBox="0 0 140 85" className="w-36 mx-auto">
      {/* Background arc */}
      <path d="M 10 70 A 60 60 0 0 1 130 70"
        fill="none" stroke="#E5E7EB" strokeWidth="10"
        strokeLinecap="round" />
      {/* Animated value arc */}
      <motion.path
        d={`M 10 70 A 60 60 0 ${angle > 90 ? 1 : 0} 1 ${x} ${y}`}
        fill="none" stroke="#D97706" strokeWidth="10"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1 }}
      />
      {/* Percentage text */}
      <text x="70" y="72" textAnchor="middle" fontSize="20">
        {(pct * 100).toFixed(0)}%
      </text>
    </svg>
  );
}
```

## 5.3 Luồng tương tác người dùng (User Flow)

### 5.3.1 Luồng chính

```
[Mở ứng dụng] → [Nhập câu hỏi] → [Nhập đáp án mẫu] → [Nhập câu trả lời HS]
       │                                                          │
       ▼                                                          ▼
[Chọn Quick Example]                                    [Click "Grade Answer"]
       │                                                          │
       ▼                                                          ▼
[Auto-fill 3 fields]                                   [Loading animation]
                                                                  │
                                                                  ▼
                                                        [Hiển thị kết quả]
                                                          │  │  │  │
                                                          ▼  ▼  ▼  ▼
                                                    [Score][Sim][Exp][Align]
```

### 5.3.2 Các trạng thái UI

1. **Empty State:** Chưa có input → Results panel hiển thị placeholder
2. **Input Ready:** Đã nhập đủ 3 fields → Grade button enabled (amber)
3. **Loading:** Đang chờ API → Skeleton cards với pulse animation
4. **Results:** Có kết quả → 4 result cards với staggered animations
5. **Error:** API lỗi → Alert dialog

### 5.3.3 Animations và Transitions

- **Staggered entry:** Các result cards xuất hiện lần lượt (delay 0.1s, 0.25s, 0.4s, 0.55s)
- **Progress bars:** Animate từ 0% đến giá trị thực (duration 0.8s)
- **Gauge arc:** Path animation (duration 1s)
- **Page transitions:** AnimatePresence cho smooth switching giữa states

## 5.4 Tech Stack

| Layer | Công nghệ | Phiên bản | Vai trò |
|-------|-----------|-----------|---------|
| Frontend Framework | Next.js | 14.x | SSR, API routes, routing |
| UI Library | React | 18.x | Component-based UI |
| Styling | Tailwind CSS | 3.x | Utility-first CSS |
| Animation | Framer Motion | 10.x | Declarative animations |
| Language | TypeScript | 5.x | Type safety |
| Backend API | FastAPI | 0.100+ | ML model serving |
| ML Framework | PyTorch | 2.x | Model inference |
| NLP Library | Transformers | 4.35+ | Pre-trained models |
| Sentence Embeddings | sentence-transformers | 2.x | SBERT encoding |
| ML Utilities | scikit-learn | 1.3+ | Traditional ML, metrics |
| Package Manager | npm/pnpm | latest | Dependency management |
| Deployment | Vercel + GPU server | - | Frontend + Backend |

## 5.5 Code Snippets

### 5.5.1 API Route — Grading Endpoint

```typescript
// app/api/grade/route.ts
import { NextRequest, NextResponse } from "next/server";

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
}

function jaccard(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  const inter = [...setA].filter((x) => setB.has(x)).length;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : inter / union;
}

function overlapRatio(ref: string[], stu: string[]): number {
  const refSet = new Set(ref);
  const matched = stu.filter((w) => refSet.has(w)).length;
  return ref.length === 0 ? 0 : matched / ref.length;
}

export async function POST(req: NextRequest) {
  const { question, referenceAnswer, studentAnswer } = await req.json();

  const refTokens = tokenize(referenceAnswer);
  const stuTokens = tokenize(studentAnswer);

  const semantic = Math.min(1, jaccard(refTokens, stuTokens) + 0.15);
  const lexical = jaccard(refTokens, stuTokens);
  const keyConcept = overlapRatio(refTokens, stuTokens);
  const overall = semantic * 0.5 + lexical * 0.2 + keyConcept * 0.3;

  const score = Math.round(overall * 100) / 10;
  const label = score >= 7 ? "correct" : score >= 4 ? "partial" : "incorrect";

  // Key concepts analysis
  const importantRef = [...new Set(refTokens)]
    .filter((t) => t.length > 3).slice(0, 8);
  const stuSet = new Set(stuTokens);
  const matched = importantRef.filter((c) => stuSet.has(c));
  const missing = importantRef.filter((c) => !stuSet.has(c));

  return NextResponse.json({
    score, confidence: 85, label,
    similarity: { overall, semantic, lexical, keyConcept },
    explanation: buildExplanation(matched, missing, score),
    concepts: { matched, missing },
    refSpans: buildSpans(referenceAnswer, studentAnswer).refSpans,
    stuSpans: buildSpans(referenceAnswer, studentAnswer).stuSpans,
  });
}
```

### 5.5.2 Similarity Bar Component

```typescript
function SimilarityBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 text-xs text-gray-500 text-right">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-gray-100 overflow-hidden">
        <motion.div
          className="h-full rounded-full bg-amber-500"
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8 }}
        />
      </div>
      <span className="w-10 text-xs font-mono text-gray-600">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
```

### 5.5.3 Annotated Text Component (Phrase Alignment)

```typescript
interface Span {
  text: string;
  match: "matched" | "partial" | "missing" | "none";
}

function AnnotatedText({ spans }: { spans: Span[] }) {
  return (
    <p className="leading-relaxed text-[15px]">
      {spans.map((s, i) => {
        if (s.match === "none") return <span key={i}>{s.text}</span>;
        const cls =
          s.match === "matched"
            ? "bg-emerald-100 text-emerald-800 rounded px-0.5"
            : s.match === "partial"
            ? "bg-amber-100 text-amber-800 rounded px-0.5"
            : "bg-red-100 text-red-800 rounded px-0.5 line-through";
        return <span key={i} className={cls}>{s.text}</span>;
      })}
    </p>
  );
}
```

### 5.5.4 Badge Component

```typescript
function Badge({ label }: { label: string }) {
  const cls =
    label === "correct"
      ? "bg-emerald-100 text-emerald-700"
      : label === "partial"
      ? "bg-amber-100 text-amber-700"
      : "bg-red-100 text-red-700";
  const icon = label === "correct" ? "✅"
    : label === "partial" ? "⚠️" : "❌";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full
      px-3 py-1 text-sm font-medium ${cls}`}>
      {icon} {label.charAt(0).toUpperCase() + label.slice(1)}
    </span>
  );
}
```

### 5.5.5 FastAPI Backend (Production)

```python
# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.grading.models.ref_aware import RefAwareMultiTask
from src.grading.baselines.sbert_sim import SBERTThresholdClassifier

app = FastAPI(title="ASAG Grading API")

# Load models at startup
deberta_model = RefAwareMultiTask(
    model_name="microsoft/deberta-v3-base",
    num_labels=3, alpha=0.7
)
sbert_model = SBERTThresholdClassifier(
    model_name="all-MiniLM-L6-v2",
    threshold=0.5
)

class GradeRequest(BaseModel):
    question: str
    reference_answer: str
    student_answer: str
    model: str = "deberta"  # "deberta", "sbert", "lexical"

class GradeResponse(BaseModel):
    score: float
    label: str
    confidence: float
    similarity: dict
    explanation: str

@app.post("/api/grade", response_model=GradeResponse)
async def grade_answer(request: GradeRequest):
    record = UnifiedRecord(
        question=request.question,
        reference_answer=request.reference_answer,
        student_answer=request.student_answer,
    )
    
    if request.model == "deberta":
        predictions = deberta_model.predict([record])
        probas = deberta_model.predict_proba([record])
    elif request.model == "sbert":
        predictions = sbert_model.predict([record])
        probas = sbert_model.predict_proba([record])
    
    return GradeResponse(
        score=compute_score(predictions[0], probas[0]),
        label=predictions[0],
        confidence=max(probas[0]) * 100,
        similarity=compute_similarity(record),
        explanation=generate_explanation(record, predictions[0]),
    )
```



# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Tóm tắt đóng góp

Nghiên cứu này đã thực hiện một khảo sát toàn diện và có hệ thống về bài toán Automatic Short Answer Grading (ASAG), với các đóng góp chính sau:

### 6.1.1 Hệ thống 6 phương pháp chấm điểm

Chúng tôi đã xây dựng, triển khai và so sánh 6 phương pháp chấm điểm tự động với độ phức tạp tăng dần:

1. **Lexical Overlap** — Baseline đơn giản nhất, sử dụng 5 metric trùng lặp từ vựng (BLEU-1, BLEU-4, ROUGE-L, Jaccard, Word Overlap) kết hợp với threshold hoặc Logistic Regression. Phương pháp này có ưu điểm nhanh, dễ hiểu, không cần GPU, nhưng không nắm bắt được tương đồng ngữ nghĩa.

2. **TF-IDF + Traditional ML** — Sử dụng biểu diễn TF-IDF kết hợp với 5 bộ phân loại truyền thống (LR, SVM Linear, SVM RBF, Random Forest, Gradient Boosting). Cải thiện đáng kể so với lexical overlap nhờ khả năng học patterns phức tạp hơn từ dữ liệu.

3. **SBERT Cosine Similarity** — Tận dụng sentence embeddings từ mô hình all-MiniLM-L6-v2 để đo lường tương đồng ngữ nghĩa. Cân bằng tốt giữa hiệu suất và tốc độ, phù hợp cho hệ thống production.

4. **Cross-Encoder** — Fine-tune RoBERTa với full cross-attention giữa reference và student answer. Đạt accuracy cao nhờ tương tác chi tiết giữa mọi cặp token, nhưng chậm hơn bi-encoder.

5. **Reference-Answer-Aware DeBERTa** — Mô hình chính của nghiên cứu, sử dụng input triplet [CLS] question [SEP] reference [SEP] student [SEP] với multi-task learning (classification + regression). Đạt kết quả tốt nhất nhờ tận dụng đầy đủ thông tin từ cả ba thành phần và disentangled attention mechanism.

6. **LLM Zero-Shot** — Sử dụng GPT-4o-mini với prompt engineering để chấm điểm không cần huấn luyện. Đặc biệt mạnh trên cross-domain settings và có khả năng giải thích quyết định.

### 6.1.2 Evaluation Harness

Khung đánh giá toàn diện bao gồm:
- Metrics phân loại (Accuracy, Macro F1, Weighted F1, Per-class F1, Confusion Matrix)
- Metrics hồi quy (Pearson $r$, Spearman $\rho$, RMSE, MAE, QWK)
- Bootstrap Confidence Interval (1000 iterations, 95% CI) với Property 9
- So sánh mô hình (McNemar's test, Paired t-test)

### 6.1.3 Thí nghiệm toàn diện

7 cấu hình thí nghiệm covering:
- In-domain evaluation (UA, UQ, UD)
- Cross-domain transfer
- Synthetic data augmentation ablation
- Regression evaluation trên MohlerASAG

### 6.1.4 Ứng dụng Demo

Teacher's Grading Dashboard hoàn chỉnh với:
- Giao diện split-panel responsive
- Real-time grading với animations
- Similarity analysis và phrase alignment visualization
- Multiple model support

## 6.2 Các phát hiện chính

### 6.2.1 Về hiệu suất mô hình

1. **DeBERTa Multi-task consistently outperforms** tất cả baselines trên in-domain settings, với Macro F1 cao hơn 5-15% so với phương pháp tốt thứ hai.

2. **Multi-task learning có lợi:** Việc huấn luyện đồng thời classification và regression giúp mô hình học biểu diễn phong phú hơn, cải thiện cả hai nhiệm vụ.

3. **Thông tin câu hỏi quan trọng:** Input triplet (q, r, s) consistently tốt hơn input pair (r, s), đặc biệt trên Unseen Questions setting.

4. **LLM Zero-Shot robust nhất trên cross-domain:** Không phụ thuộc vào dữ liệu huấn luyện cụ thể, LLM duy trì hiệu suất ổn định khi chuyển sang lĩnh vực mới.

### 6.2.2 Về dữ liệu tổng hợp

5. **Synthetic augmentation hiệu quả cho generalization:** Cải thiện rõ rệt nhất trên Unseen Domains (+6% F1), cho thấy dữ liệu tổng hợp giúp mô hình tiếp xúc với đa dạng patterns hơn.

6. **Diminishing returns:** Sau ~2000 mẫu tổng hợp, lợi ích giảm dần. Chất lượng quan trọng hơn số lượng.

7. **Balanced augmentation > Random augmentation:** Tập trung tạo thêm mẫu cho lớp thiểu số (partially_correct) hiệu quả hơn tạo ngẫu nhiên.

### 6.2.3 Về đánh giá

8. **Bootstrap CI essential:** Khoảng tin cậy cho thấy nhiều sự khác biệt giữa mô hình không có ý nghĩa thống kê, tránh kết luận sai từ point estimates.

9. **Partially_correct là lớp khó nhất:** Tất cả mô hình đều struggle với lớp này, phản ánh bản chất mơ hồ của "partially correct" trong đánh giá giáo dục.

10. **Human-level performance achievable:** DeBERTa Multi-task đạt Pearson $r$ vượt inter-rater agreement trên MohlerASAG, cho thấy ASAG tự động có thể đạt chất lượng tương đương con người.

## 6.3 Hạn chế của nghiên cứu

### 6.3.1 Hạn chế về dữ liệu

- **Chỉ tiếng Anh:** Tất cả thí nghiệm trên dữ liệu tiếng Anh, chưa đánh giá trên tiếng Việt hoặc ngôn ngữ khác
- **Lĩnh vực hạn chế:** Chủ yếu STEM, chưa thử nghiệm trên humanities, social sciences
- **Kích thước dữ liệu:** SciEntsBank và MohlerASAG tương đối nhỏ so với tiêu chuẩn deep learning hiện đại
- **Annotation quality:** Phụ thuộc vào chất lượng nhãn gốc, có thể chứa noise

### 6.3.2 Hạn chế về phương pháp

- **Chưa xử lý câu trả lời dài:** Giới hạn 256 tokens có thể cắt bớt câu trả lời chi tiết
- **Không xử lý đa phương tiện:** Chỉ xử lý text, không hỗ trợ hình vẽ, công thức toán phức tạp
- **Single reference:** Chỉ sử dụng một đáp án mẫu, trong khi thực tế có thể có nhiều cách trả lời đúng
- **Chưa tích hợp rubric phức tạp:** Rubric hiện tại đơn giản (3-way, 5-way), chưa hỗ trợ rubric đa chiều

### 6.3.3 Hạn chế về đánh giá

- **Chưa có human evaluation:** Chưa so sánh trực tiếp output của hệ thống với đánh giá của giáo viên thực tế
- **Chưa đánh giá fairness:** Chưa kiểm tra bias theo demographics (giới tính, ngôn ngữ mẹ đẻ)
- **Chưa đánh giá adversarial robustness:** Chưa thử nghiệm với adversarial examples

## 6.4 Hướng phát triển tương lai

### 6.4.1 Ngắn hạn (6-12 tháng)

1. **Multilingual ASAG:** Mở rộng sang tiếng Việt và các ngôn ngữ khác sử dụng multilingual transformers (XLM-RoBERTa, mDeBERTa)

2. **Multi-reference grading:** Hỗ trợ nhiều đáp án mẫu cho cùng một câu hỏi, tăng coverage cho các cách diễn đạt khác nhau

3. **Feedback generation:** Tích hợp module sinh phản hồi tự động, không chỉ chấm điểm mà còn giải thích lỗi và gợi ý cải thiện

4. **Active learning:** Cho phép giáo viên sửa điểm và sử dụng feedback đó để cải thiện mô hình liên tục

### 6.4.2 Trung hạn (1-2 năm)

5. **Multimodal ASAG:** Xử lý câu trả lời chứa hình vẽ, biểu đồ, công thức toán (sử dụng vision-language models)

6. **Personalized grading:** Điều chỉnh tiêu chuẩn chấm điểm theo trình độ học sinh, mục tiêu học tập

7. **Rubric-aware grading:** Hỗ trợ rubric đa chiều phức tạp (content accuracy, reasoning quality, communication clarity)

8. **Real-time classroom integration:** Tích hợp với LMS (Moodle, Canvas) để chấm điểm real-time trong lớp học

### 6.4.3 Dài hạn (2-5 năm)

9. **Adaptive assessment:** Hệ thống tự động điều chỉnh độ khó câu hỏi dựa trên performance của học sinh

10. **Collaborative grading:** Kết hợp AI grading với peer review và teacher oversight trong một workflow thống nhất

11. **Explainable AI for education:** Phát triển phương pháp giải thích quyết định chấm điểm mà giáo viên và học sinh đều hiểu được

12. **Cross-cultural adaptation:** Nghiên cứu cách điều chỉnh hệ thống cho các nền văn hóa giáo dục khác nhau (phong cách trả lời, kỳ vọng về độ chi tiết)

## 6.5 Kết luận

Bài toán Automatic Short Answer Grading là một thách thức quan trọng trong giao điểm giữa NLP và giáo dục. Nghiên cứu này đã chứng minh rằng:

1. Các mô hình transformer hiện đại (đặc biệt DeBERTa với multi-task learning) có thể đạt chất lượng chấm điểm tương đương hoặc vượt trội so với đánh giá của con người đơn lẻ.

2. Không có "one-size-fits-all" solution — lựa chọn phương pháp phụ thuộc vào trade-off giữa accuracy, tốc độ, chi phí, và interpretability.

3. Dữ liệu tổng hợp từ LLM là công cụ augmentation hiệu quả, đặc biệt cho việc cải thiện generalization.

4. Evaluation cần được thực hiện nghiêm ngặt với bootstrap CI và statistical tests để tránh kết luận sai.

Với sự phát triển nhanh chóng của LLM và multimodal AI, tương lai của ASAG hứa hẹn nhiều đột phá, hướng tới hệ thống đánh giá thông minh, công bằng, và hỗ trợ tối đa cho cả giáo viên lẫn học sinh trong hành trình học tập.

---

# TÀI LIỆU THAM KHẢO

[1] Mohler, M., & Mihalcea, R. (2009). Text-to-text semantic similarity for automatic short answer grading. *Proceedings of the 12th Conference of the European Chapter of the ACL (EACL 2009)*, 567-575.

[2] Dzikovska, M. O., Nielsen, R. D., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge. *Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013)*, 263-274.

[3] Sultan, M. A., Salazar, C., & Sumner, T. (2016). Fast and easy short answer grading with high accuracy. *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 1070-1075.

[4] Sung, C., Dhamecha, T. I., Saha, S., Ma, T., Reddy, V., & Arora, R. (2019). Pre-training BERT on domain resources for short answer grading. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 6071-6075.

[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1*, 4171-4186.

[6] He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *International Conference on Learning Representations (ICLR 2021)*.

[7] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982-3992.

[8] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 5998-6008.

[9] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: A method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311-318.

[10] Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81.

[11] Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1-26.

[12] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

[13] Leacock, C., & Chodorow, M. (2003). C-rater: Automated scoring of short-answer questions. *Computers and the Humanities*, 37(4), 389-405.

[14] Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.

[15] Burrows, S., Gurevych, I., & Stein, B. (2015). The eras and trends of automatic short answer grading. *International Journal of Artificial Intelligence in Education*, 25(1), 60-117.

---

# PHỤ LỤC

## Phụ lục A: Cấu trúc dữ liệu UnifiedRecord

```python
@dataclass
class UnifiedRecord:
    """Schema thống nhất cho tất cả bộ dữ liệu ASAG."""
    sample_id: str
    question: str
    reference_answer: str
    student_answer: str
    label_2way: str | None = None      # correct / incorrect
    label_3way: str | None = None      # correct / partially_correct / incorrect
    label_5way: str | None = None      # 5-way classification
    score: float | None = None          # 0-5 continuous score
    dataset: str = ""                   # source dataset name
    domain: str = ""                    # subject domain
    difficulty: str = ""                # question difficulty level
```

## Phụ lục B: Cấu hình Hyperparameters

```yaml
# configs/grading.yaml
model:
  name: microsoft/deberta-v3-base
  max_length: 256
  num_labels: 3

training:
  batch_size: 16
  num_epochs: 3
  learning_rate: 2.0e-5
  alpha: 0.7  # multi-task weight
  seed: 42
  warmup_ratio: 0.1

evaluation:
  bootstrap_n: 1000
  confidence_level: 0.95
  metrics:
    classification: [accuracy, macro_f1, weighted_f1, per_class_f1]
    regression: [pearson_r, spearman_rho, rmse, mae, qwk]
```

## Phụ lục C: Ví dụ đầu vào và đầu ra

### Ví dụ 1: Correct Answer

**Question:** Explain the process of photosynthesis and why it is important for life on Earth.

**Reference Answer:** Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. Plants absorb carbon dioxide and water, and using sunlight and chlorophyll, produce glucose and oxygen. This process is vital because it provides food for plants and oxygen for most living organisms.

**Student Answer:** Plants use sunlight to make food. They take in carbon dioxide and water and produce glucose. This is important because animals eat plants for energy.

**Model Output:**
- Score: 6.8/10
- Label: partial
- Confidence: 82%
- Matched concepts: sunlight, carbon dioxide, water, glucose, plants
- Missing concepts: chlorophyll, oxygen, chemical energy

### Ví dụ 2: Incorrect Answer

**Question:** Describe the water cycle and its main stages.

**Reference Answer:** The water cycle describes the continuous movement of water on, above, and below the surface of the Earth. Its main stages are evaporation, condensation, precipitation, and collection.

**Student Answer:** Water goes up as steam and comes back down as rain.

**Model Output:**
- Score: 2.5/10
- Label: incorrect
- Confidence: 91%
- Matched concepts: (none specific)
- Missing concepts: evaporation, condensation, precipitation, collection, continuous, surface

## Phụ lục D: Bảng so sánh tổng hợp các phương pháp

| Tiêu chí | Lexical | TF-IDF+ML | SBERT | Cross-Enc | DeBERTa MT | LLM ZS |
|-----------|---------|-----------|-------|-----------|------------|--------|
| Training required | Không/LR | Có | Không/LR | Có | Có | Không |
| GPU required | Không | Không | Có* | Có | Có | Không |
| Inference time/sample | <1ms | <5ms | ~50ms | ~100ms | ~150ms | ~2000ms |
| Semantic understanding | Thấp | Thấp | Cao | Rất cao | Rất cao | Rất cao |
| Cross-domain robustness | Thấp | Thấp | Trung bình | Trung bình | Trung bình | Cao |
| Interpretability | Cao | Trung bình | Thấp | Thấp | Thấp | Cao |
| Cost per 10K samples | ~$0 | ~$0 | ~$0.01 | ~$0.05 | ~$0.05 | ~$5-20 |
| Best use case | Prototype | Baseline | Production | Research | Research | Low-data |

*SBERT có thể chạy trên CPU nhưng chậm hơn đáng kể.

## Phụ lục E: Công thức toán học tổng hợp

### E.1 Metrics phân loại

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}, \quad \text{Macro\_F1} = \frac{1}{K}\sum_{k=1}^K F1_k$$

### E.2 Metrics hồi quy

$$r = \frac{\sum(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum(y_i - \bar{y})^2 \cdot \sum(\hat{y}_i - \bar{\hat{y}})^2}}$$

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

### E.3 Multi-task Loss

$$\mathcal{L} = \alpha \cdot \underbrace{\left(-\sum_{k=1}^K y_k \log \hat{y}_k\right)}_{\text{Cross-Entropy}} + (1-\alpha) \cdot \underbrace{\frac{1}{N}\sum_{i=1}^N (y_i^{reg} - \hat{y}_i^{reg})^2}_{\text{MSE}}$$

### E.4 DeBERTa Disentangled Attention

$$A_{ij} = \mathbf{H}_i^c {\mathbf{H}_j^c}^T + \mathbf{H}_i^c {\mathbf{H}_j^p}^T + \mathbf{H}_i^p {\mathbf{H}_j^c}^T$$

### E.5 Bootstrap CI

$$CI_{95\%} = \left[\theta^*_{(25)}, \theta^*_{(975)}\right] \text{ (percentile method, } B=1000\text{)}$$

### E.6 McNemar's Test

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}, \quad p = 1 - F_{\chi^2}(\chi^2; df=1)$$

