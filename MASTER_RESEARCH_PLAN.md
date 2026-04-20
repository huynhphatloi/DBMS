# UNIFIED MASTER-LEVEL RESEARCH PLAN
## Automatic Short-Answer Grading, Misconception Mining, Feedback Generation, and Robustness Evaluation

---

# ════════════════════════════════════════════════════════════════
# PART A — DATA AUDIT AND INTEGRATION FEASIBILITY
# ════════════════════════════════════════════════════════════════

## 1. Individual Dataset Analysis

### 1.1 SciEntsBank (External, Public Benchmark)

| Property | Value |
|---|---|
| **Source** | SemEval-2013 Task 7 (Dzikovska et al., 2013) |
| **Domain** | Science (15 topics, elementary-level) |
| **Size** | ~10,000 student answers across ~200 questions |
| **Fields** | question, reference_answer, student_answer, label_5way |
| **Label space** | 5-way: correct, partially_correct_incomplete, contradictory, irrelevant, non_domain |
| **Splits** | Predefined: Unseen Answers (UA), Unseen Questions (UQ), Unseen Domains (UD) |
| **Human-labeled** | YES — expert-annotated |
| **Synthetic** | NO |
| **Feedback fields** | NONE |
| **Misconception annotations** | NONE |
| **Usable for supervised grading** | YES — this is the gold standard ASAG benchmark |
| **Usable for feedback** | NO (no feedback annotations) |
| **Usable for misconception mining** | PARTIALLY — incorrect answers exist but no misconception tags |
| **Usable for robustness** | PARTIALLY — cross-split evaluation only, no adversarial perturbations |

**Critical strengths:** Established benchmark with known baselines. Enables direct comparison with published work. Three evaluation settings (UA, UQ, UD) test generalization rigorously.

**Critical weaknesses:** Elementary science only. No feedback. No misconception labels. No adversarial variants. The 5-way label set is the only annotation.

---

### 1.2 MohlerASAG (External, Public Benchmark)

| Property | Value |
|---|---|
| **Source** | Mohler et al. (2011), University of North Texas |
| **Domain** | Computer Science (Data Structures course) |
| **Size** | ~2,273 student answers across 80 questions (10 assignments × 7–10 questions) |
| **Fields** | question, reference_answer (up to multiple), student_answer, score (0–5 continuous, averaged from 2 annotators) |
| **Label space** | Continuous 0–5 score (real-valued, e.g., 3.5) |
| **Splits** | NONE predefined — must create your own |
| **Human-labeled** | YES — dual-annotator with averaged scores |
| **Synthetic** | NO |
| **Feedback fields** | NONE |
| **Misconception annotations** | NONE |
| **Usable for supervised grading** | YES — regression/ordinal task |
| **Usable for feedback** | NO |
| **Usable for misconception mining** | PARTIALLY — low-scoring answers exist but no tags |
| **Usable for robustness** | NO (no perturbations, no adversarial variants) |

**Critical strengths:** Real student answers from a real course. Continuous scores allow regression and ordinal analysis. CS domain complements SciEntsBank's science domain. Dual-annotator scores provide inter-annotator agreement data.

**Critical weaknesses:** Small dataset. No predefined splits (leakage risk if not handled). No feedback. No misconception labels. Single course, single institution.

---

### 1.3 data-generate.csv (Internal, Synthetic/Generated)

| Property | Value |
|---|---|
| **Source** | Your internally generated dataset (likely LLM-generated) |
| **Domain** | 20 domains: biology, chemistry, physics, earth_science, environmental_science, astronomy, health_science, scientific_method, programming_fundamentals, data_structures, algorithms, databases, operating_systems, computer_networks, software_engineering, ai_ml_basics, mathematics_for_science, statistics_and_experiments, cybersecurity_and_networked_systems, digital_logic_and_computing |
| **Size** | 10,000 rows, 800 unique questions, 80 subdomains |
| **Fields (30 columns)** | instance_id, question_id, domain, subdomain, difficulty, split, question, reference_answer, alternative_reference_answers, key_concepts, misconception_inventory, student_answer, student_answer_style, lexical_overlap_level, semantic_correctness_score_0_5, label_5way, label_3way, label_2way, misconception_tags, misconception_span_rationale, missing_concepts, extra_incorrect_claims, feedback_short, feedback_detailed, feedback_type, feedback_tone, adversarial_variant_of, perturbation_type, robustness_notes, annotation_confidence |
| **Label space** | 5-way (correct, partially_correct_incomplete, contradictory, irrelevant, non_domain), 3-way, 2-way, AND continuous 0–5 score |
| **Splits** | Predefined: train (7000), valid (1000), test_unseen_questions (400), test_unseen_answers (500), test_seen (500), test_adversarial (300), test_unseen_domains (300) |
| **Human-labeled** | NO — this is synthetic/generated data |
| **Synthetic** | YES — all 10,000 rows are generated |
| **Feedback fields** | YES — feedback_short (10,000/10,000), feedback_detailed (10,000/10,000) |
| **Misconception annotations** | YES — misconception_tags (1,670 rows), misconception_inventory (all rows), missing_concepts, extra_incorrect_claims |
| **Adversarial variants** | YES — 7,610 rows are adversarial variants with 12 perturbation types |
| **Usable for supervised grading** | WITH CAVEATS — synthetic labels need validation |
| **Usable for feedback** | YES — richest feedback source, but synthetic quality must be verified |
| **Usable for misconception mining** | YES — has misconception_tags, inventory, missing_concepts |
| **Usable for robustness** | YES — purpose-built adversarial variants with 12 perturbation types |

**Label distributions:**
- label_5way: correct=3360, partially_correct_incomplete=2520, contradictory=1670, irrelevant=1630, non_domain=820
- Scores: 5=2400, 4=960, 3=1720, 2=1670, 1=2400, 0=850
- Difficulty: easy=2508, medium=4398, hard=3094
- Answer styles: 10 types (concise, mixed-claim, noisy, paraphrased_low_overlap, overconfident, fragmented, explanatory, hedged, example-driven, topic-drifted)
- Perturbation types: 12 types (high_overlap_wrong_meaning=694, near-contradiction=522, one_correct_plus_fatal_error=708, misleading_fluent_explanation=718, grammar_noise=544, word_order_change=719, hedge_language=540, concept-jumble=695, synonym_swap=713, paraphrase_low_overlap=545, distractor_sentence_added=690, vague_but_plausible=522)
- annotation_confidence: min=0.83, max=0.97, mean=0.908

**Critical strengths:** Extremely rich schema. Covers all 4 research directions. Multi-domain. Has adversarial variants, feedback, misconception annotations, multiple label granularities, and predefined splits with question-level separation.


**CRITICAL WARNINGS FOR data-generate.csv:**

⚠️ **WARNING 1 — Synthetic data bias:** All student answers are LLM-generated, not from real students. This means:
- The distribution of errors may not reflect real student misconceptions
- The linguistic patterns are LLM-typical, not student-typical
- Models trained on this data may overfit to synthetic patterns and fail on real student text
- **Mitigation:** NEVER use this as the sole test set. Always validate on SciEntsBank and MohlerASAG.

⚠️ **WARNING 2 — Circular annotation risk:** The labels, feedback, misconception tags, and student answers were likely all generated by the same LLM pipeline. This creates a risk where:
- The feedback perfectly explains the label because both were generated together
- A model could learn to exploit generation artifacts rather than genuine semantic understanding
- **Mitigation:** Use this data for training/augmentation only. Test on human-annotated data.

⚠️ **WARNING 3 — Feedback quality is unverified:** feedback_short and feedback_detailed exist for all 10,000 rows, but:
- Some feedback_short entries are truncated or incoherent (e.g., "The answer is too unclear to photosynthesis in green plants; 'I would put it like this...")
- feedback_detailed contains template-like patterns with inserted fragments
- **Mitigation:** Sample 200+ rows for human quality audit before using feedback as gold standard.

⚠️ **WARNING 4 — Student answers show generation artifacts:** Inspecting the actual student_answer field reveals:
- Unnatural phrasing: "the oddly key move uses light energy properly so the already outcome reaches water"
- Inserted noise words that follow patterns: "oddly", "plainly", "deeply", "largely", "closely"
- These are clearly perturbation artifacts, not real student language
- **Mitigation:** Acknowledge this limitation. Do not claim models trained on this generalize to real students without validation.

⚠️ **WARNING 5 — annotation_confidence is self-reported:** The confidence scores (0.83–0.97) were generated by the same pipeline, not by human annotators. They cannot be treated as true annotation reliability measures.

---

### 1.4 data-scraping.json (Internal, Scraped)

| Property | Value |
|---|---|
| **Source** | Scraped from OpenStax textbooks |
| **Domain** | Physics (college-physics-2e, university-physics-volume-1), Chemistry (chemistry-2e), Astronomy (astronomy-2e), Biology (biology-2e), Anatomy & Physiology (anatomy-and-physiology-2e) |
| **Size** | 129 entries, 128 unique questions |
| **Fields (5 only)** | id, questions, reference_answer, student_answer, label |
| **Label space** | "label" field = source textbook identifier (e.g., "openstax_college-physics-2e"), NOT a grading label |
| **Splits** | NONE |
| **Human-labeled** | N/A — no grading labels exist |
| **Synthetic** | NO — questions and reference answers are from published textbooks |
| **Student answers** | ALL EMPTY — every single student_answer field is "" |
| **Reference answers** | 28 out of 129 contain "Not found" |
| **Usable for supervised grading** | NO — no student answers, no grading labels |
| **Usable for feedback** | NO |
| **Usable for misconception mining** | NO |
| **Usable for robustness** | NO |

**Label distribution by source:**
- openstax_chemistry-2e: 50
- openstax_university-physics-volume-1: 33
- openstax_college-physics-2e: 18
- openstax_astronomy-2e: 11
- openstax_anatomy-and-physiology-2e: 11
- openstax_biology-2e: 6

**CRITICAL WARNINGS FOR data-scraping.json:**

🚨 **FATAL ISSUE 1 — No student answers:** ALL 129 entries have empty student_answer fields. This dataset CANNOT be used for any grading, feedback, or misconception mining task in its current form.

🚨 **FATAL ISSUE 2 — 28 missing reference answers:** 21.7% of entries have reference_answer = "Not found". These are unusable even as question banks.

🚨 **FATAL ISSUE 3 — "label" is not a grading label:** The "label" field contains the source textbook name, not a correctness label. This is a metadata field, not an annotation.

🚨 **FATAL ISSUE 4 — Many questions are computational/numerical:** A large portion of the physics and chemistry questions require numerical computation (e.g., "What is 100 km/h in m/s?"). These are NOT short-answer conceptual questions suitable for ASAG. They require mathematical reasoning, not semantic understanding.

🚨 **FATAL ISSUE 5 — Reference answers contain formatting artifacts:** Many reference answers have duplicated text from web scraping (e.g., "27 . 8 m/s 27 . 8 m/s" — the answer appears twice due to HTML rendering issues).

**Verdict on data-scraping.json:** This dataset is NOT usable for any of the 4 research directions in its current form. It can ONLY serve as:
1. A question bank for generating synthetic student answers (after cleaning)
2. A source of additional reference answers for domain expansion
3. It must be filtered to remove numerical/computational questions and "Not found" entries first

---

## 2. Critical Issues and Warnings — Cross-Dataset

### 2.1 Label Space Incompatibility

| Dataset | Label Type | Scale |
|---|---|---|
| SciEntsBank | 5-way categorical | correct, partially_correct_incomplete, contradictory, irrelevant, non_domain |
| MohlerASAG | Continuous score | 0.0 – 5.0 (real-valued) |
| data-generate.csv | All three + score | 5-way, 3-way, 2-way, AND 0–5 integer score |
| data-scraping.json | None | N/A |

**Problem:** SciEntsBank uses categorical labels. MohlerASAG uses continuous scores. data-generate.csv has both but they are synthetic. Harmonization requires explicit mapping decisions (see Section 4).

### 2.2 Domain Mismatch

| Dataset | Domains |
|---|---|
| SciEntsBank | Elementary science (15 topics) |
| MohlerASAG | CS / Data Structures (1 course) |
| data-generate.csv | 20 domains (8 science + 8 CS + 4 math/stats/cyber/logic) |
| data-scraping.json | Physics, Chemistry, Astronomy, Biology, Anatomy |

**Problem:** SciEntsBank is elementary-level. MohlerASAG is university CS. data-generate.csv spans both but is synthetic. Cross-domain evaluation is possible but domain shift effects must be measured and reported.

### 2.3 Leakage Risks

| Risk | Source | Severity |
|---|---|---|
| Question overlap between data-generate.csv and SciEntsBank | data-generate.csv covers biology, which overlaps with SciEntsBank science topics | MEDIUM — must verify no identical questions |
| Feedback leaking label information | data-generate.csv feedback_short/feedback_detailed contain explicit grading rationale | HIGH — feedback fields MUST be excluded from grading model inputs |
| misconception_span_rationale leaking labels | Contains explicit statements like "The answer directly reflects the misconception tag X" | HIGH — must be excluded from grading inputs |
| MohlerASAG question-level leakage | No predefined splits; if same question appears in train and test, model memorizes question-specific patterns | HIGH — must split by question_id |
| Adversarial variant linkage | adversarial_variant_of field links perturbed samples to originals; both must be in same split | HIGH — if original is in train and variant in test, model has seen the base answer |

### 2.4 Bias Risks

| Bias | Description | Affected Dataset |
|---|---|---|
| Synthetic language bias | LLM-generated student answers have different linguistic distribution than real students | data-generate.csv |
| Domain coverage bias | Science domains overrepresented vs. CS domains in data-generate.csv | data-generate.csv |
| Difficulty distribution bias | Medium difficulty overrepresented (44%) | data-generate.csv |
| Label imbalance | "correct" class is 33.6% in data-generate.csv; "non_domain" is only 8.2% | data-generate.csv |
| Textbook bias | OpenStax questions skew toward computational/numerical | data-scraping.json |
| Elementary vs. university level | SciEntsBank is elementary; MohlerASAG is university | Cross-dataset |

---

## 3. Unified Research-Ready Schema

```
unified_schema = {
    # === Identity ===
    "sample_id": str,              # Globally unique: e.g., "SEB_UA_0001", "MOH_0001", "GEN_000001", "SCR_001"
    "source_dataset": str,         # "scientsbank" | "mohler" | "data_generate" | "data_scraping"
    "original_id": str,            # Original ID from source dataset
    "question_id": str,            # Standardized question identifier

    # === Domain ===
    "domain": str,                 # Standardized: "biology", "chemistry", "physics", "cs", etc.
    "subdomain": str,              # More specific topic
    "difficulty": str,             # "easy" | "medium" | "hard" | "unknown"

    # === Core Triplet ===
    "question": str,
    "reference_answer": str,
    "alternative_reference_answers": list[str],  # Empty list if none
    "student_answer": str,

    # === Grading Labels ===
    "score_raw": float | None,     # Original score (0–5 for Mohler, 0–5 int for data-generate)
    "score_normalized": float | None,  # Normalized to 0.0–1.0
    "label_2way": str | None,      # "correct" | "incorrect"
    "label_3way": str | None,      # "correct" | "partially_correct" | "incorrect"
    "label_5way": str | None,      # "correct" | "partially_correct_incomplete" | "contradictory" | "irrelevant" | "non_domain"

    # === Concept-Level Annotations ===
    "key_concepts": list[str],
    "misconception_tags": list[str],
    "misconception_inventory": list[dict],  # [{tag, belief}]
    "missing_concepts": list[str],
    "extra_incorrect_claims": list[str],

    # === Feedback ===
    "feedback_short": str | None,
    "feedback_detailed": str | None,
    "feedback_type": str | None,   # "praise" | "hint" | "corrective" | "misconception_refutation" | "encouragement"
    "feedback_tone": str | None,   # "tutor_like" | "supportive" | "neutral" | "direct"
    "rationale": str | None,       # Why this label was assigned

    # === Splits and Metadata ===
    "split": str,                  # "train" | "valid" | "test_ua" | "test_uq" | "test_ud" | "test_adversarial" | etc.
    "is_human_annotated": bool,
    "is_synthetic": bool,
    "is_adversarial": bool,
    "perturbation_type": str | None,
    "adversarial_variant_of": str | None,
    "student_answer_style": str | None,
    "annotation_confidence": float | None,

    # === Usability Flags ===
    "usable_for_grading": bool,
    "usable_for_feedback": bool,
    "usable_for_misconception_mining": bool,
    "usable_for_robustness_eval": bool,
}
```

---

## 4. Label Mapping Strategy

### 4.1 MohlerASAG Score → Classification Labels

MohlerASAG uses continuous 0–5 scores (averaged from 2 annotators). Mapping:

| Score Range | label_2way | label_3way | label_5way |
|---|---|---|---|
| [4.0, 5.0] | correct | correct | correct |
| [2.5, 4.0) | correct* | partially_correct | partially_correct_incomplete |
| [1.0, 2.5) | incorrect | incorrect | contradictory** |
| [0.0, 1.0) | incorrect | incorrect | irrelevant |

*Note: The 2-way boundary at 2.5 is debatable. Report results at multiple thresholds (2.0, 2.5, 3.0) as a sensitivity analysis.

**Note: MohlerASAG does not distinguish "contradictory" from "irrelevant." Mapping low scores to "contradictory" vs. "irrelevant" is unreliable without additional annotation. For 5-way experiments, use MohlerASAG only for regression or 2-way/3-way classification.

**Recommendation:** Keep MohlerASAG as a regression task (primary) and 3-way classification (secondary). Do NOT force it into 5-way classification.

### 4.2 SciEntsBank 5-way → 3-way and 2-way

| label_5way | label_3way | label_2way |
|---|---|---|
| correct | correct | correct |
| partially_correct_incomplete | partially_correct | incorrect* |
| contradictory | incorrect | incorrect |
| irrelevant | incorrect | incorrect |
| non_domain | incorrect | incorrect |

*Note: Mapping "partially_correct_incomplete" to "incorrect" in 2-way is the standard convention in SemEval-2013. However, this is a lossy mapping. Report 2-way results with and without partially_correct in the "correct" class as a sensitivity check.

### 4.3 data-generate.csv → Unified Schema

Direct mapping — the dataset already uses the same 5-way label set as SciEntsBank. The 3-way and 2-way labels are pre-computed. The 0–5 score maps directly.

**However:** Verify that the label_3way mapping in data-generate.csv is consistent with the SciEntsBank convention. Current data shows label_3way has: correct=3360, incorrect=4970, contradictory=1670. This means "contradictory" is kept as a separate class in 3-way, which DIFFERS from the standard SciEntsBank 3-way mapping where contradictory → incorrect. **This must be harmonized.**

### 4.4 When to Use Regression vs. Classification

| Scenario | Task Formulation |
|---|---|
| Comparing with SciEntsBank literature | 5-way, 3-way, 2-way classification |
| Comparing with MohlerASAG literature | Regression (Pearson r, RMSE) or ordinal regression |
| Unified cross-dataset evaluation | 3-way classification (most compatible) |
| Fine-grained error analysis | 5-way classification |
| Production deployment scenario | 3-way + confidence score |

---

## 5. Handling Strategy Per Source

### 5.1 SciEntsBank — CORE SUPERVISED SOURCE

| Role | Details |
|---|---|
| **Primary role** | Core evaluation benchmark for grading (Project 1) |
| **Training** | Use official train split |
| **Testing** | Use UA, UQ, UD splits — report all three |
| **For misconception mining** | Use incorrect/contradictory answers as input corpus (no gold misconception labels) |
| **For feedback** | NOT usable (no feedback annotations) |
| **For robustness** | Cross-split evaluation only |
| **Additional annotation needed** | None for grading. For misconception mining: manual cluster labeling after clustering. |

### 5.2 MohlerASAG — CORE SUPERVISED SOURCE (CS DOMAIN)

| Role | Details |
|---|---|
| **Primary role** | Core evaluation benchmark for grading, especially regression task |
| **Training** | Create question-level splits (60/20/20 by question_id) |
| **Testing** | Hold-out questions only |
| **For misconception mining** | Use low-scoring answers (score < 2.0) as input corpus |
| **For feedback** | NOT usable |
| **For robustness** | NOT usable |
| **Additional annotation needed** | Question-level split creation. Optional: manual misconception labeling on a subset. |

### 5.3 data-generate.csv — AUGMENTATION + FEEDBACK/ROBUSTNESS SOURCE

| Role | Details |
|---|---|
| **Primary role** | Training augmentation for grading. PRIMARY source for feedback (Project 3), misconception mining (Project 2), and robustness evaluation (Project 4) |
| **Training** | Use train split (7000) for augmentation alongside SciEntsBank/MohlerASAG |
| **Testing** | NEVER use as sole test set. Always pair with human-annotated test sets. |
| **For misconception mining** | PRIMARY source — has misconception_tags, inventory, missing_concepts |
| **For feedback** | PRIMARY source — has feedback_short, feedback_detailed, feedback_type, feedback_tone |
| **For robustness** | PRIMARY source — has 7,610 adversarial variants with 12 perturbation types |
| **Additional annotation needed** | Human quality audit of 200+ samples for feedback quality. Validation of label accuracy on 100+ samples. |

### 5.4 data-scraping.json — AUXILIARY QUESTION BANK ONLY

| Role | Details |
|---|---|
| **Primary role** | Question bank for generating additional synthetic student answers |
| **Preprocessing required** | (1) Remove 28 "Not found" reference answers. (2) Remove numerical/computational questions. (3) Clean duplicated text in reference answers. (4) Filter to conceptual short-answer questions only. |
| **Estimated usable entries after cleaning** | ~40–60 conceptual questions (rough estimate) |
| **For grading** | NOT directly usable |
| **For feedback** | NOT directly usable |
| **For misconception mining** | NOT directly usable |
| **For robustness** | NOT directly usable |
| **Can be used to** | Generate synthetic student answers using LLMs, then add to training pool (clearly marked as synthetic) |
| **Additional annotation needed** | Manual filtering of conceptual vs. computational questions. If synthetic answers are generated, they need quality review. |

---


# ════════════════════════════════════════════════════════════════
# PART B — DESIGNING A UNIFIED MASTER-LEVEL RESEARCH PROGRAM
# ════════════════════════════════════════════════════════════════

## 6. Coherent Research Vision

### 6.1 Unified Vision Statement

**"Toward Reliable and Interpretable Automatic Short-Answer Grading: A Unified Framework for Grading, Misconception Diagnosis, Targeted Feedback, and Robustness Evaluation"**

The core insight binding these 4 projects: grading alone is insufficient for educational impact. A complete ASAG system must (1) assign accurate grades, (2) understand WHY an answer is wrong, (3) communicate that understanding back to the student, and (4) resist manipulation. Each project addresses one layer of this stack, and each layer depends on the one below it.

### 6.2 Dependency Chain

```
Project 1: GRADING (foundation)
    ↓ provides trained grading models + label predictions
Project 2: MISCONCEPTION MINING (builds on grading)
    ↓ provides error taxonomies + misconception clusters
Project 3: FEEDBACK GENERATION (builds on grading + misconceptions)
    ↓ provides feedback system that can be stress-tested
Project 4: ROBUSTNESS EVALUATION (tests all above)
```

### 6.3 Shared Infrastructure

1. **Shared data pipeline:** One unified dataset in the schema from Section 3, loaded once, filtered per project
2. **Shared embedding backbone:** One sentence encoder (e.g., all-MiniLM-L6-v2 or all-mpnet-base-v2) used across all projects for consistency
3. **Shared evaluation harness:** Common metrics computation, common reporting format
4. **Shared preprocessing:** Text normalization, tokenization, encoding — done once
5. **Shared model registry:** Grading models from Project 1 are reused in Projects 2, 3, and 4

### 6.4 Core vs. Extension

| Project | Role | Justification |
|---|---|---|
| **Project 1 (Grading)** | CORE THESIS | Foundation for everything. Most established literature. Clearest evaluation. Most defensible. |
| **Project 3 (Feedback)** | CORE THESIS (secondary) | Strongest novelty when grounded in concept-gap analysis. Differentiates from "just another ASAG paper." |
| **Project 2 (Misconception Mining)** | EXTENSION | Interesting but harder to evaluate rigorously. Clustering quality is subjective. |
| **Project 4 (Robustness)** | EXTENSION | High academic value but can be scoped down. Even a partial robustness analysis adds significant value. |

**Minimum defensible thesis:** Projects 1 + 3 (Grading + Feedback) with the unified benchmark as the methodological contribution.

**Full thesis:** All 4 projects, with Projects 2 and 4 as shorter chapters.

---

## 7. Detailed Project Specifications

### PROJECT 1: Automatic Short-Answer Grading

**a) Vietnamese title:** Ứng dụng khai phá dữ liệu trong chấm điểm tự động câu trả lời ngắn của sinh viên

**b) English title:** Multi-Granularity Automatic Short-Answer Grading: A Comparative Study Across Domains, Label Spaces, and Model Families

**c) Research objective:** Develop and evaluate automatic grading models for short student answers at multiple granularity levels (2-way, 3-way, 5-way, regression), across multiple domains (science, CS), using both human-annotated and synthetic training data.

**d) Research questions:**
- RQ1.1: How do transformer-based models compare to traditional ML baselines for ASAG across different label granularities?
- RQ1.2: Does training on synthetic data (data-generate.csv) improve or degrade performance on human-annotated test sets (SciEntsBank, MohlerASAG)?
- RQ1.3: How does cross-domain transfer affect grading accuracy (science → CS, CS → science)?
- RQ1.4: Is multi-task learning (joint classification + regression) beneficial compared to single-task models?

**e) Input / Output:**
- Input: (question, reference_answer, student_answer)
- Output: label_5way OR label_3way OR label_2way OR score (0–5)

**f) Datasets used:**
- SciEntsBank (primary evaluation — UA, UQ, UD)
- MohlerASAG (secondary evaluation — regression)
- data-generate.csv (training augmentation, ablation study)

**g) Required baselines:**
1. Lexical overlap baseline (BLEU, ROUGE-L, word overlap ratio → threshold)
2. TF-IDF + Logistic Regression / SVM / Random Forest
3. SBERT cosine similarity → threshold classifier
4. Cross-encoder (e.g., cross-encoder/stsb-roberta-base) fine-tuned
5. Reference-answer-aware model (concatenate [question; ref_answer; student_answer])
6. LLM zero-shot (GPT-4 / Claude) as upper-bound analysis only

**h) Proposed methodology:**
- Primary model: Fine-tuned cross-encoder with reference-answer-aware input format
- Architecture: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP] → classification head
- Training: SciEntsBank train + optional data-generate.csv augmentation
- Multi-task variant: shared encoder, two heads (classification + regression)

**i) Main experiments:**
1. All baselines on SciEntsBank (UA, UQ, UD) × (2-way, 3-way, 5-way)
2. Best model on MohlerASAG (regression: Pearson r, RMSE; classification: 3-way)
3. Augmentation ablation: SciEntsBank-only vs. SciEntsBank + data-generate.csv
4. Cross-domain: train on science, test on CS (and vice versa)

**j) Secondary experiments / ablations:**
- Input ablation: (q + ref + student) vs. (ref + student) vs. (student only)
- Label granularity comparison: which granularity is most useful?
- Multi-task vs. single-task
- Effect of annotation_confidence filtering on data-generate.csv
- Per-domain performance breakdown

**k) Evaluation metrics:**
- Classification: Accuracy, Macro-F1, Weighted-F1 (primary: Macro-F1)
- Regression: Pearson r, Spearman ρ, RMSE, MAE
- Ordinal: Quadratic Weighted Kappa (QWK)
- All metrics reported with 95% confidence intervals (bootstrap)

**l) Risks and limitations:**
- SciEntsBank is elementary science — results may not generalize to university level
- MohlerASAG is small — high variance in results
- Synthetic augmentation may introduce distribution shift
- Cross-domain transfer may show poor results (this is a finding, not a failure)

**m) Academic contribution:**
- Systematic comparison across label granularities on a unified benchmark
- Empirical analysis of synthetic data augmentation for ASAG
- Cross-domain transfer analysis (science ↔ CS)
- Reference-answer-aware architecture comparison

**n) Potential:** Core thesis chapter (Chapter 4). Publishable as a conference paper at EDM, AIED, or BEA workshop.

---

### PROJECT 2: Misconception Mining

**a) Vietnamese title:** Khai phá các mẫu lỗi phổ biến trong câu trả lời ngắn của sinh viên

**b) English title:** Mining Student Misconception Patterns from Short-Answer Responses Using Embedding-Based Clustering

**c) Research objective:** Discover and categorize common misconception patterns from incorrect and partially correct student answers using unsupervised and semi-supervised clustering methods.

**d) Research questions:**
- RQ2.1: Can embedding-based clustering recover pedagogically meaningful misconception categories from incorrect student answers?
- RQ2.2: Does incorporating the question and reference answer into the embedding improve cluster quality compared to using the student answer alone?
- RQ2.3: How do discovered clusters align with the predefined misconception_inventory in data-generate.csv?
- RQ2.4: Are misconception patterns consistent across domains or domain-specific?

**e) Input / Output:**
- Input: Set of (question, reference_answer, student_answer) triples where label ∈ {partially_correct_incomplete, contradictory, irrelevant}
- Output: Cluster assignments, cluster labels (manual), misconception taxonomy

**f) Datasets used:**
- data-generate.csv (primary — has misconception_tags for validation)
- SciEntsBank (secondary — incorrect answers, no gold misconception labels)
- MohlerASAG (secondary — low-scoring answers)

**g) Required baselines:**
1. Random clustering baseline
2. TF-IDF + KMeans
3. SBERT embeddings + KMeans
4. SBERT embeddings + HDBSCAN
5. BERTopic-style (SBERT + UMAP + HDBSCAN + c-TF-IDF)

**h) Proposed methodology:**
- Embed (question ⊕ reference_answer ⊕ student_answer) using SBERT
- Dimensionality reduction: UMAP
- Clustering: HDBSCAN (allows noise points, variable cluster sizes)
- Cluster labeling: (1) automatic via c-TF-IDF keywords, (2) manual review
- Validation against misconception_tags in data-generate.csv

**i) Main experiments:**
1. Clustering on data-generate.csv incorrect answers, per-question and per-domain
2. Compare embedding strategies: answer-only vs. (question + answer) vs. (question + ref + answer)
3. Compare clustering methods: KMeans vs. HDBSCAN vs. Agglomerative
4. Validate clusters against gold misconception_tags (NMI, ARI, purity)

**j) Secondary experiments / ablations:**
- Effect of including partially_correct vs. only contradictory/irrelevant
- Cross-domain misconception transfer
- Cluster stability analysis (bootstrap resampling)
- Qualitative case studies of discovered misconception patterns

**k) Evaluation metrics:**
- Intrinsic: Silhouette score, Calinski-Harabasz, Davies-Bouldin
- Extrinsic (against gold tags): NMI, ARI, Purity, V-measure
- Qualitative: Human evaluation of cluster coherence (3-point scale)

**l) Risks and limitations:**
- Misconception_tags in data-generate.csv are synthetic — validation is circular if not careful
- Real student misconceptions (SciEntsBank, MohlerASAG) have no gold labels — evaluation is qualitative only
- Clustering quality is inherently subjective
- Small number of misconception types per question limits cluster diversity

**m) Academic contribution:**
- Systematic comparison of embedding + clustering strategies for ASAG misconception mining
- Cross-domain misconception analysis
- Methodology for validating discovered misconceptions against predefined inventories

**n) Potential:** Extension chapter (Chapter 5). Could be part of a workshop paper but unlikely standalone publication without real student data validation.

---

### PROJECT 3: Automatic Feedback Generation

**a) Vietnamese title:** Xây dựng hệ thống phản hồi tự động cho câu trả lời ngắn của sinh viên sử dụng khai phá dữ liệu và xử lý ngôn ngữ tự nhiên

**b) English title:** Concept-Gap-Grounded Automatic Feedback Generation for Short Student Answers

**c) Research objective:** Generate targeted, pedagogically useful feedback for student answers by identifying specific concept gaps (missing concepts, incorrect claims) and producing feedback grounded in those gaps rather than generic praise or criticism.

**d) Research questions:**
- RQ3.1: Does concept-gap-grounded feedback achieve higher semantic coverage and factual consistency than template-based or generic generative feedback?
- RQ3.2: How does the quality of the grading model (Project 1) affect downstream feedback quality?
- RQ3.3: Can retrieval-augmented feedback (retrieving similar past feedback) outperform purely generative approaches?
- RQ3.4: How do human evaluators rate the pedagogical usefulness of generated feedback?

**e) Input / Output:**
- Input: (question, reference_answer, student_answer, predicted_label, predicted_missing_concepts)
- Output: feedback_short (1–2 sentences) + feedback_detailed (paragraph)

**f) Datasets used:**
- data-generate.csv (primary — has feedback_short, feedback_detailed, missing_concepts)
- SciEntsBank + MohlerASAG (for grading model; feedback generated at inference time)

**g) Required baselines:**
1. Template-based feedback (rule-based: if label=correct → "Good job"; if incorrect → "Review [topic]")
2. Retrieval-based feedback (find most similar training example, return its feedback)
3. SBERT similarity + template slot-filling
4. Fine-tuned T5/BART for feedback generation
5. LLM zero-shot feedback (GPT-4) as upper-bound

**h) Proposed methodology:**
- **Hybrid pipeline:**
  1. Grade the answer (Project 1 model)
  2. Identify missing concepts (from key_concepts list, check which are absent in student answer via entailment)
  3. Identify incorrect claims (from extra_incorrect_claims or via contradiction detection)
  4. Generate feedback grounded in identified gaps: "Your answer correctly mentions [X] but misses [Y]. The key connection is [Z]."
- **Training:** Fine-tune T5-base on data-generate.csv (student_answer → feedback_detailed, conditioned on missing_concepts)
- **Grounding constraint:** Feedback must reference specific concepts from the reference answer

**i) Main experiments:**
1. All baselines on data-generate.csv test split
2. Grounded feedback vs. ungrounded feedback (ablation: with/without missing_concepts input)
3. Effect of grading accuracy on feedback quality (use gold labels vs. predicted labels)
4. Human evaluation on 100 samples (3 evaluators, rubric-based)

**j) Secondary experiments / ablations:**
- Feedback_short vs. feedback_detailed generation
- Effect of feedback_tone conditioning
- Hallucination rate analysis (does feedback mention concepts not in reference answer?)
- Cross-domain feedback transfer

**k) Evaluation metrics:**
- **Automatic (secondary):** ROUGE-L, BERTScore, BLEU (against gold feedback)
- **Concept coverage:** % of gold missing_concepts mentioned in generated feedback
- **Factual consistency:** Entailment score between generated feedback and reference_answer
- **Hallucination rate:** % of generated feedback containing claims not supported by reference_answer
- **Human evaluation (primary):** 5-point rubric on (1) accuracy, (2) specificity, (3) pedagogical usefulness, (4) actionability, (5) tone appropriateness

**l) Risks and limitations:**
- Gold feedback in data-generate.csv is synthetic and contains quality issues (truncated sentences, template artifacts)
- Human evaluation is expensive and subjective
- Concept-gap identification depends on key_concepts quality
- Without real student deployment, pedagogical impact cannot be measured

**m) Academic contribution:**
- Concept-gap-grounded feedback generation (not just "your answer is wrong")
- Systematic comparison of feedback strategies for ASAG
- Analysis of grading-feedback pipeline coupling
- Human evaluation protocol for ASAG feedback

**n) Potential:** Core thesis chapter (Chapter 6). Strong publication potential at BEA workshop, AIED, or L@S if human evaluation is rigorous.

---

### PROJECT 4: Robustness and Reliability Evaluation

**a) Vietnamese title:** Phân tích độ tin cậy, tính bền vững và khả năng chống gian lận của các mô hình chấm điểm câu trả lời ngắn

**b) English title:** Adversarial Robustness and Calibration Analysis of Automatic Short-Answer Grading Models

**c) Research objective:** Systematically evaluate how grading models from Project 1 behave under adversarial perturbations, distribution shift, and gaming attempts, and compare vulnerability profiles across model families.

**d) Research questions:**
- RQ4.1: Which perturbation types cause the largest performance drops for each model family?
- RQ4.2: Are transformer-based models more robust than traditional ML models to adversarial inputs?
- RQ4.3: How well-calibrated are grading model confidence scores under distribution shift?
- RQ4.4: Can simple heuristics (keyword stuffing, verbosity) fool grading models?

**e) Input / Output:**
- Input: (question, reference_answer, adversarial_student_answer, perturbation_type)
- Output: Performance metrics under perturbation, calibration curves, vulnerability profiles

**f) Datasets used:**
- data-generate.csv (primary — 7,610 adversarial variants, 12 perturbation types, test_adversarial split of 300)
- SciEntsBank (secondary — apply perturbations to SciEntsBank test answers programmatically)

**g) Required baselines (model families to compare):**
1. **Traditional ML:** TF-IDF + SVM, TF-IDF + Random Forest
2. **Embedding-based:** SBERT similarity, fine-tuned cross-encoder
3. **LLM-based:** GPT-4 zero-shot grading (as upper-bound analysis)

**h) Proposed methodology:**
- Train all model families on clean data (SciEntsBank train + data-generate.csv train)
- Evaluate on:
  1. Clean test sets (baseline performance)
  2. data-generate.csv test_adversarial split (300 samples, 12 perturbation types)
  3. Programmatically perturbed SciEntsBank test answers
- Measure: accuracy drop, F1 drop, calibration shift, confidence-error correlation
- Build vulnerability matrix: model_family × perturbation_type → performance_drop

**i) Main experiments:**
1. Performance on clean vs. adversarial test sets for all model families
2. Per-perturbation-type analysis (which attacks are most effective?)
3. Calibration analysis: reliability diagrams, Expected Calibration Error (ECE)
4. Vulnerability matrix construction

**j) Secondary experiments / ablations:**
- Adversarial training: does including adversarial examples in training improve robustness?
- Confidence thresholding: at what threshold should the system abstain?
- Ensemble robustness: are model ensembles more robust?
- Per-domain robustness variation

**k) Evaluation metrics:**
- Performance drop: Δ(Macro-F1) = F1_clean - F1_adversarial
- Per-perturbation drop: Δ(F1) per perturbation type
- Calibration: ECE, MCE (Maximum Calibration Error), reliability diagrams
- Confidence metrics: AUROC of confidence vs. correctness
- Abstention: accuracy-coverage curves

**l) Risks and limitations:**
- Adversarial variants in data-generate.csv are synthetic — real gaming attempts may differ
- LLM evaluation is expensive (API costs for GPT-4)
- Programmatic perturbations of SciEntsBank may not be realistic
- Calibration analysis requires probability outputs (not all models provide these)

**m) Academic contribution:**
- First systematic adversarial robustness evaluation for ASAG with 12 perturbation types
- Cross-model-family vulnerability comparison
- Calibration analysis for ASAG (rarely studied)
- Practical recommendations for robust ASAG deployment

**n) Potential:** Extension chapter (Chapter 7). High publication potential as a standalone paper at ACL, EMNLP, or NAACL (robustness/evaluation track) if perturbations are validated on real data.

---


# ════════════════════════════════════════════════════════════════
# PART C — EXPERIMENT DESIGN FOR PROJECT 1 (ASAG / GRADING)
# ════════════════════════════════════════════════════════════════

## 8. Complete Technical Pipeline

### 8.1 Data Cleaning
1. **Text normalization:** lowercase, strip extra whitespace, normalize unicode
2. **SciEntsBank:** minimal cleaning needed (already clean)
3. **MohlerASAG:** check for encoding issues, normalize whitespace
4. **data-generate.csv:** flag rows with annotation_confidence < 0.85 for review; remove rows where student_answer is clearly malformed (< 3 tokens)
5. **data-scraping.json:** remove "Not found" entries, remove numerical/computational questions, clean duplicated reference answer text

### 8.2 Label Harmonization
- Apply mappings from Section 4
- For MohlerASAG: create label_3way and label_2way from scores using thresholds
- For data-generate.csv: fix label_3way to match SciEntsBank convention (contradictory → incorrect in 3-way)
- Verify label consistency: score=5 should always map to label_2way=correct, etc.

### 8.3 Split Strategy

**SciEntsBank:** Use official splits (UA, UQ, UD). No modification needed.

**MohlerASAG:** Create question-level splits:
- Group all answers by question_id
- Randomly assign questions (not answers) to train/valid/test (60/20/20)
- Ensure no question appears in multiple splits
- Stratify by score distribution if possible

**data-generate.csv:** Use predefined splits. Verify that:
- adversarial_variant_of links are respected (original and variant in same split)
- No question_id leaks across train/test boundaries
- test_unseen_questions contains only questions not in train
- test_unseen_domains contains only domains not in train

**Cross-dataset evaluation:**
- Train on SciEntsBank train → test on MohlerASAG test (cross-domain)
- Train on SciEntsBank train + data-generate.csv train → test on SciEntsBank UA/UQ/UD (augmentation study)
- Train on MohlerASAG train → test on SciEntsBank UQ (cross-domain, reverse)

### 8.4 In-Domain vs. Cross-Domain Evaluation Matrix

| Train Set | Test Set | Type |
|---|---|---|
| SciEntsBank train | SciEntsBank UA | In-domain, unseen answers |
| SciEntsBank train | SciEntsBank UQ | In-domain, unseen questions |
| SciEntsBank train | SciEntsBank UD | Cross-domain (within science) |
| SciEntsBank train | MohlerASAG test | Cross-domain (science → CS) |
| MohlerASAG train | SciEntsBank UQ | Cross-domain (CS → science) |
| SciEntsBank + data-gen train | SciEntsBank UA/UQ/UD | Augmentation study |
| data-gen train only | SciEntsBank UA/UQ/UD | Synthetic-only training |

---

## 9. Required Baselines (Detailed)

### Baseline 1: Lexical Overlap
- Compute: BLEU-1, BLEU-4, ROUGE-L, Jaccard similarity, word overlap ratio between student_answer and reference_answer
- Use each metric as a single feature → threshold-based classifier
- Also: combine all metrics as features → Logistic Regression
- **Purpose:** Establishes floor performance. Any model that cannot beat this is useless.

### Baseline 2: TF-IDF + Traditional ML
- Represent each (reference_answer, student_answer) pair as TF-IDF vectors
- Features: TF-IDF of student_answer, TF-IDF of reference_answer, cosine similarity, element-wise difference
- Classifiers: Logistic Regression, SVM (linear + RBF), Random Forest, Gradient Boosting
- **Purpose:** Standard ML baseline. Surprisingly competitive on some datasets.

### Baseline 3: SBERT Similarity
- Encode reference_answer and student_answer with SBERT (all-MiniLM-L6-v2)
- Compute cosine similarity
- Use similarity as single feature → threshold classifier or → Logistic Regression
- **Purpose:** Tests whether semantic similarity alone is sufficient.

### Baseline 4: Cross-Encoder (Fine-tuned)
- Input: [CLS] reference_answer [SEP] student_answer [SEP]
- Fine-tune cross-encoder/stsb-roberta-base on training data
- Classification head on [CLS] token
- **Purpose:** Strong transformer baseline without question context.

### Baseline 5: Reference-Answer-Aware Model (Proposed Main Model)
- Input: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
- Fine-tune roberta-base or deberta-v3-base
- Classification head for label prediction; optional regression head for score
- **Purpose:** Full-context model. Expected to be strongest.

### Baseline 6: LLM Zero-Shot (Upper Bound Analysis)
- Prompt GPT-4 / Claude with: "Given the question, reference answer, and student answer, classify the student answer as [correct / partially_correct / incorrect]."
- Include rubric in prompt
- Run on test sets only (no training)
- **Purpose:** Upper bound. NOT the thesis contribution. Analysis only.
- **Cost consideration:** ~$50–100 for full evaluation on SciEntsBank test sets

---

## 10. Task Formulations

### 10.1 2-Way Classification
- Classes: correct, incorrect
- **When to use:** Simplest setting. Use for initial experiments and when comparing with papers that report 2-way only.
- **Limitation:** Loses all nuance. A contradictory answer and an irrelevant answer are treated the same.

### 10.2 3-Way Classification
- Classes: correct, partially_correct, incorrect
- **When to use:** Best balance of granularity and feasibility. RECOMMENDED as the primary classification task for cross-dataset comparison.
- **Note:** This is the most compatible across SciEntsBank (mapped from 5-way), MohlerASAG (mapped from scores), and data-generate.csv.

### 10.3 5-Way Classification
- Classes: correct, partially_correct_incomplete, contradictory, irrelevant, non_domain
- **When to use:** SciEntsBank-specific experiments. Provides finest error analysis.
- **Limitation:** Only SciEntsBank and data-generate.csv have native 5-way labels. MohlerASAG cannot be reliably mapped to 5-way.

### 10.4 Ordinal Regression / Score Regression
- Output: continuous score 0–5
- **When to use:** MohlerASAG experiments (native format). Also useful for data-generate.csv.
- **Models:** Ordinal regression (CORAL loss), standard regression (MSE loss)
- **Metrics:** Pearson r, Spearman ρ, RMSE, MAE, QWK

### 10.5 Multi-Task Learning
- Joint training: classification head + regression head on shared encoder
- **When to use:** When both label and score are available (data-generate.csv). Hypothesis: multi-task learning provides regularization and improves both tasks.
- **Implementation:** Shared DeBERTa encoder → two heads with weighted loss: L = α·L_cls + (1-α)·L_reg

### 10.6 Recommendation
**Primary task:** 3-way classification (most comparable across datasets)
**Secondary tasks:** 5-way on SciEntsBank, regression on MohlerASAG, multi-task on data-generate.csv

---

## 11. Metrics Specification

| Task | Primary Metric | Secondary Metrics |
|---|---|---|
| 2-way classification | Macro-F1 | Accuracy, Weighted-F1 |
| 3-way classification | Macro-F1 | Accuracy, Weighted-F1, per-class F1 |
| 5-way classification | Macro-F1 | Accuracy, Weighted-F1, per-class F1, confusion matrix |
| Regression | Pearson r | Spearman ρ, RMSE, MAE |
| Ordinal | QWK | Pearson r, adjacent accuracy |

**Reporting requirements:**
- All metrics with 95% confidence intervals (1000-iteration bootstrap)
- Statistical significance tests (McNemar's test for classification, paired t-test for regression)
- Confusion matrices for all classification experiments
- Per-class F1 breakdown for imbalanced classes

---

# ════════════════════════════════════════════════════════════════
# PART D — EXPERIMENT DESIGN FOR PROJECT 2 (MISCONCEPTION MINING)
# ════════════════════════════════════════════════════════════════

## 12. Misconception Mining Pipeline

### 12.1 Data Selection
1. From data-generate.csv: select rows where label_5way ∈ {partially_correct_incomplete, contradictory, irrelevant} → ~6,640 rows
2. From SciEntsBank: select rows where label ∈ {partially_correct_incomplete, contradictory, irrelevant, non_domain}
3. From MohlerASAG: select rows where score < 3.0

### 12.2 Embedding Strategies (Compare All Three)

**Strategy A — Answer Only:**
- embed(student_answer)
- Pro: Simple. Clusters answers by surface similarity.
- Con: Ignores what the question asked. Two answers about different topics may cluster together if they use similar words.

**Strategy B — Question + Answer:**
- embed(question ⊕ student_answer)
- Pro: Contextualizes the answer. Same wrong answer means different things for different questions.
- Con: Question text may dominate the embedding.

**Strategy C — Full Triplet:**
- embed(question ⊕ reference_answer ⊕ student_answer)
- Pro: Captures the gap between expected and actual answer.
- Con: Reference answer may dominate. Embedding space may not separate misconception types well.

**Recommended:** Compare all three. Hypothesis: Strategy B or C will outperform A for per-question clustering, but A may work better for cross-question misconception discovery.

### 12.3 Clustering Methods

| Method | Strengths | Weaknesses | When to Use |
|---|---|---|---|
| KMeans | Simple, fast, well-understood | Requires k, assumes spherical clusters | Baseline |
| HDBSCAN | No k required, handles noise, variable density | Sensitive to min_cluster_size | Primary method |
| Agglomerative | Hierarchical, produces dendrogram | Slow for large datasets | Analysis tool |
| BERTopic-style | SBERT + UMAP + HDBSCAN + c-TF-IDF | Complex pipeline, many hyperparameters | Topic-modeling approach |
| Spectral | Handles non-convex clusters | Requires k, expensive | If HDBSCAN fails |

**Recommended pipeline:** SBERT → UMAP (n_components=5) → HDBSCAN (min_cluster_size=5)

### 12.4 Clustering Granularity

**Per-question clustering (RECOMMENDED for primary analysis):**
- Cluster incorrect answers for each question separately
- Advantage: Misconceptions are question-specific
- Disadvantage: Small sample sizes per question

**Per-domain clustering:**
- Cluster all incorrect answers within a domain
- Advantage: Larger sample sizes, discovers cross-question patterns
- Disadvantage: Mixes different question contexts

**Global clustering:**
- Cluster all incorrect answers together
- Advantage: Discovers universal error patterns (e.g., "off-topic" cluster)
- Disadvantage: Loses question-specific meaning

**Recommendation:** Do per-question first, then per-domain, then compare.

### 12.5 Cluster Quality Evaluation

**Intrinsic metrics (no gold labels needed):**
- Silhouette score (higher = better separation)
- Calinski-Harabasz index
- Davies-Bouldin index (lower = better)
- Cluster size distribution (flag if one cluster dominates)

**Extrinsic metrics (against gold misconception_tags in data-generate.csv):**
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Purity
- V-measure (homogeneity + completeness)

**Qualitative evaluation:**
- Sample 5 answers from each cluster
- 2 human evaluators label each cluster with a misconception description
- Inter-annotator agreement (Cohen's κ)
- Pedagogical meaningfulness rating (1–5 scale)

### 12.6 From Clusters to Misconception Patterns

1. For each cluster, extract top-5 keywords via c-TF-IDF
2. Compare with misconception_inventory tags in data-generate.csv
3. Manual labeling: assign a pedagogical description to each cluster
4. Build misconception taxonomy: domain → subdomain → misconception_type → example_answers
5. Validate: do the discovered patterns match known misconceptions in science/CS education literature?

---

## 13. Alternative Approaches Analysis

### Approach 1: Mining Using Answer Only
- **Method:** Embed student_answer → cluster
- **Advantage:** Domain-agnostic. Can discover surface-level error patterns (e.g., "too short", "off-topic language", "keyword stuffing")
- **Disadvantage:** Cannot distinguish conceptual errors from linguistic errors. Two answers about completely different topics may cluster together.
- **Best for:** Discovering answer-style patterns, not conceptual misconceptions.

### Approach 2: Using Full Triplet (Question + Reference + Student)
- **Method:** Embed concatenation → cluster
- **Advantage:** Captures the semantic gap. A student who says "plants absorb food from soil" clusters differently from one who says "photosynthesis produces CO2" because the gap from the reference is different.
- **Disadvantage:** Computationally heavier. Reference answer may dominate embedding.
- **Best for:** Conceptual misconception discovery.

### Approach 3: Using Structured Annotations
- **Method:** Use missing_concepts and extra_incorrect_claims directly as features
- **Advantage:** Most precise. Directly encodes what's wrong.
- **Disadvantage:** Only available in data-generate.csv (synthetic). Not available for real student data.
- **Best for:** Validation of clustering results. If clusters align with structured annotations, the clustering method is working.

**Recommendation:** Use Approach 2 as primary, Approach 3 as validation, Approach 1 as ablation.

---

# ════════════════════════════════════════════════════════════════
# PART E — EXPERIMENT DESIGN FOR PROJECT 3 (AUTOMATIC FEEDBACK)
# ════════════════════════════════════════════════════════════════

## 14. Feedback Generation Pipeline

### 14.1 Strategy Comparison

**Strategy 1: Template-Based Feedback**
- Rules: if label=correct → "Your answer correctly addresses [question_topic]."
- If label=partially_correct → "Your answer mentions [present_concepts] but misses [missing_concepts]."
- If label=incorrect → "Your answer does not address [question_topic]. Review [key_concepts]."
- **Pro:** Deterministic, no hallucination, fast
- **Con:** Generic, not personalized, limited expressiveness

**Strategy 2: Retrieval-Based Feedback**
- Find the most similar (question, student_answer) pair in training set
- Return its feedback_detailed
- Similarity: SBERT cosine similarity on student_answer
- **Pro:** Returns human-quality feedback (if training feedback is good)
- **Con:** May return irrelevant feedback if no close match exists

**Strategy 3: Span-Aware Feedback**
- Identify which spans in student_answer are correct vs. incorrect
- Use token-level attention or entailment to locate errors
- Generate feedback pointing to specific spans
- **Pro:** Most specific and actionable
- **Con:** Technically complex, requires span-level annotations (not available)

**Strategy 4: Generative Feedback (Fine-tuned T5/BART)**
- Input: question [SEP] reference_answer [SEP] student_answer [SEP] label [SEP] missing_concepts
- Output: feedback_detailed
- Fine-tune T5-base or BART-base on data-generate.csv
- **Pro:** Flexible, can generate novel feedback
- **Con:** Hallucination risk, requires quality training data

**Strategy 5: Hybrid (RECOMMENDED)**
- Step 1: Grade the answer (Project 1 model)
- Step 2: Identify missing concepts (entailment check: for each key_concept, does student_answer entail it?)
- Step 3: Identify incorrect claims (contradiction detection against reference_answer)
- Step 4: Generate feedback conditioned on (label, missing_concepts, incorrect_claims)
- **Pro:** Grounded, interpretable, reduces hallucination
- **Con:** Pipeline complexity, error propagation from grading step

### 14.2 Recommended Architecture

```
Input: (question, reference_answer, student_answer)
    ↓
[Grading Model from Project 1] → predicted_label, confidence
    ↓
[Concept Gap Detector] → missing_concepts, present_concepts
    ↓
[Contradiction Detector] → incorrect_claims (if any)
    ↓
[Feedback Generator]
    Input: question + reference_answer + student_answer + label + missing_concepts + incorrect_claims
    Output: feedback_short + feedback_detailed
```

---

## 15. Constructing Gold Feedback

### 15.1 Using Existing Feedback Fields
- data-generate.csv has feedback_short (10,000) and feedback_detailed (10,000)
- **Quality issues observed:**
  - feedback_short sometimes contains truncated sentences
  - feedback_detailed follows a template pattern with inserted fragments
  - Some feedback references internal generation artifacts (e.g., "the wording 'I would put it like this...'")

### 15.2 Quality Audit Protocol
1. Sample 200 rows stratified by label_5way and feedback_type
2. Two human evaluators rate each feedback on:
   - Factual accuracy (1–5): Does the feedback correctly identify what's right/wrong?
   - Specificity (1–5): Does it point to specific concepts, not just "try again"?
   - Coherence (1–5): Is the feedback grammatically correct and readable?
   - Pedagogical value (1–5): Would a student learn from this feedback?
3. Compute inter-annotator agreement (Cohen's κ)
4. If average quality < 3.5/5, the feedback needs human refinement before use as gold standard

### 15.3 When Human Refinement Is Necessary
- If feedback references generation artifacts → must be cleaned
- If feedback is factually incorrect (contradicts reference_answer) → must be corrected
- If feedback is too generic ("review the topic") → must be made specific
- **Estimate:** 20–30% of feedback may need refinement based on initial inspection

### 15.4 Avoiding Generic/Hallucinatory Feedback
- **Grounding constraint:** Every claim in feedback must be traceable to reference_answer or key_concepts
- **Hallucination detection:** Check if generated feedback mentions concepts not in reference_answer or key_concepts
- **Specificity filter:** Reject feedback that doesn't mention at least one specific concept
- **Confidence gating:** Only generate detailed feedback when grading confidence > 0.8; otherwise, generate cautious feedback

---

## 16. Feedback Evaluation Metrics

### 16.1 Automatic Metrics (Secondary — Use With Caution)
- ROUGE-L: overlap with gold feedback (poor proxy for quality)
- BERTScore: semantic similarity with gold feedback (better than ROUGE)
- BLEU: n-gram overlap (poor for feedback evaluation)
- **Warning:** High ROUGE/BLEU does not mean good feedback. Low ROUGE/BLEU does not mean bad feedback. A correct paraphrase scores low on ROUGE but is perfectly good feedback.

### 16.2 Concept Coverage Metric (Proposed)
- For each sample, compute: coverage = |missing_concepts ∩ concepts_mentioned_in_feedback| / |missing_concepts|
- A feedback that mentions all missing concepts has coverage = 1.0
- **This is the most important automatic metric for this project.**

### 16.3 Factual Consistency Metric
- Use NLI model to check: does the generated feedback ENTAIL the reference_answer (or at least not CONTRADICT it)?
- Compute: consistency = fraction of feedback sentences that are entailed by or neutral to reference_answer
- Flag any feedback sentence that contradicts the reference_answer

### 16.4 Human Evaluation Rubric (Primary)

| Criterion | 1 (Poor) | 3 (Adequate) | 5 (Excellent) |
|---|---|---|---|
| **Accuracy** | Feedback contains factual errors | Feedback is mostly correct | Feedback is fully correct |
| **Specificity** | Generic ("try again") | Mentions the topic | Points to specific missing concepts |
| **Actionability** | Student doesn't know what to fix | Student knows the area to review | Student knows exactly what to add/change |
| **Tone** | Harsh or confusing | Neutral | Encouraging and constructive |
| **Pedagogical value** | No learning value | Some learning value | Student would learn from this |

- 3 evaluators, 100 samples, stratified by label and feedback strategy
- Report: mean scores per criterion, inter-annotator agreement (Krippendorff's α)

---

# ════════════════════════════════════════════════════════════════
# PART F — EXPERIMENT DESIGN FOR PROJECT 4 (ROBUSTNESS / RELIABILITY)
# ════════════════════════════════════════════════════════════════

## 17. Robustness Evaluation Framework

### 17.1 Perturbation Taxonomy

The 12 perturbation types in data-generate.csv, organized by attack category:

**Category A: Surface-Level Perturbations (should NOT change the grade)**
| Perturbation | Description | Expected Effect |
|---|---|---|
| grammar_noise | Introduce typos, grammar errors | Grade should stay the same |
| word_order_change | Reorder words/phrases | Grade should stay the same |
| synonym_swap | Replace words with synonyms | Grade should stay the same |
| paraphrase_low_overlap | Rewrite with different words, same meaning | Grade should stay the same |

**Category B: Semantic Perturbations (SHOULD change the grade)**
| Perturbation | Description | Expected Effect |
|---|---|---|
| near-contradiction | Add a near-contradictory claim | Grade should decrease |
| one_correct_plus_fatal_error | Mix correct and fatally wrong claims | Grade should decrease |
| concept-jumble | Mix concepts from nearby topics | Grade should decrease |
| high_overlap_wrong_meaning | High word overlap but wrong meaning | Grade should decrease |

**Category C: Gaming/Deception Perturbations**
| Perturbation | Description | Expected Effect |
|---|---|---|
| misleading_fluent_explanation | Fluent but wrong explanation | Grade should decrease (but models may be fooled) |
| hedge_language | Use hedging to obscure wrongness | Grade should decrease |
| vague_but_plausible | Vague answer that sounds plausible | Grade should decrease |
| distractor_sentence_added | Add irrelevant but topical sentence | Grade should stay the same (but may confuse models) |

### 17.2 Additional Perturbations to Generate Programmatically

For SciEntsBank test answers, generate:
1. **Keyword stuffing:** Append key_concepts as a list to any answer
2. **Verbosity attack:** Pad correct answer with 3× irrelevant filler text
3. **Copy-reference attack:** Copy the reference answer verbatim (should get full marks — tests if model rewards copying)
4. **Empty answer:** Submit empty string (should get 0)
5. **Random text:** Submit random Wikipedia sentence (should get 0 / non_domain)

---

## 18. Measurements

### 18.1 Performance Drop
- Δ(Macro-F1) = F1_clean - F1_perturbed (per perturbation type)
- Δ(Accuracy) = Acc_clean - Acc_perturbed
- Report as heatmap: model × perturbation_type → Δ(F1)

### 18.2 Calibration Under Distribution Shift
- Train on clean data, evaluate calibration on:
  1. Clean test set (baseline calibration)
  2. Each perturbation type separately
- Metrics: ECE (Expected Calibration Error), MCE, reliability diagrams
- Question: Does the model become overconfident or underconfident under perturbation?

### 18.3 Confidence-Error Correlation
- For each prediction, record (confidence, is_correct)
- Compute AUROC: can confidence distinguish correct from incorrect predictions?
- Compare AUROC on clean vs. perturbed data

### 18.4 Uncertainty Estimation
- For transformer models: MC Dropout (run inference 10× with dropout, measure variance)
- For ensemble: disagreement rate across ensemble members
- Question: Does uncertainty increase for adversarial inputs? (It should.)

### 18.5 Abstention / Reject Option
- Define: abstain if confidence < threshold
- Plot accuracy-coverage curve: as threshold increases, accuracy increases but coverage decreases
- Find optimal threshold for 95% accuracy target
- Compare thresholds across model families

### 18.6 Error Taxonomy
- For each model, categorize errors on adversarial data:
  - False positive (incorrect answer graded as correct) — most dangerous
  - False negative (correct answer graded as incorrect) — annoying but less harmful
  - Grade inflation (partially_correct → correct)
  - Grade deflation (correct → partially_correct)
- Report error type distribution per model family

---

## 19. Model Family Comparison

### 19.1 Vulnerability Profiles

| Attack Type | Traditional ML (TF-IDF) | Transformer (Cross-Encoder) | LLM (GPT-4) |
|---|---|---|---|
| grammar_noise | LOW vulnerability (TF-IDF ignores grammar) | LOW (robust to noise) | LOW |
| synonym_swap | HIGH (different words = different features) | LOW (semantic understanding) | LOW |
| keyword_stuffing | HIGH (bag-of-words fooled by keywords) | MEDIUM | LOW |
| misleading_fluent | LOW (doesn't understand fluency) | HIGH (may be fooled by fluent text) | MEDIUM |
| hedge_language | LOW (ignores hedging) | MEDIUM | HIGH (may interpret hedging charitably) |
| verbosity_attack | MEDIUM | MEDIUM (attention dilution) | MEDIUM |
| copy_reference | HIGH (perfect match = high similarity) | MEDIUM | LOW (can detect copying) |

**Hypothesis:** Traditional ML is most vulnerable to synonym_swap and keyword_stuffing. Transformers are most vulnerable to misleading_fluent_explanation. LLMs are most robust overall but may be vulnerable to hedge_language and vague_but_plausible.

### 19.2 Analysis Framework
- For each model family, compute:
  1. Overall robustness score = mean(1 - Δ(F1)) across all perturbation types
  2. Worst-case perturbation = argmax(Δ(F1))
  3. Calibration degradation = ECE_perturbed - ECE_clean
- Present as radar chart: one axis per perturbation type, one line per model family

---


# ════════════════════════════════════════════════════════════════
# PART G — NOVELTY, ACADEMIC CONTRIBUTION, AND POSITIONING
# ════════════════════════════════════════════════════════════════

## 20. Realistic Novelty Assessment

### 20.1 What Is NOT Novel (Do Not Claim These)
- Fine-tuning BERT/RoBERTa for ASAG — done many times
- Using SBERT for semantic similarity — standard approach
- Clustering student answers — done in education research
- Using LLMs for grading — increasingly common
- Beating SciEntsBank SOTA — unlikely and not necessary for a master's thesis

### 20.2 Where Genuine Novelty Can Come From

**Novelty Direction 1: Unified Multi-Granularity Benchmark (MEDIUM novelty, HIGH feasibility)**
- Contribution: A standardized benchmark combining SciEntsBank + MohlerASAG + synthetic data with unified schema, multiple label granularities, and adversarial test sets
- Why it matters: No existing ASAG benchmark provides grading labels + misconception annotations + feedback + adversarial variants in one package
- Risk: The synthetic data component weakens the claim. Mitigate by clearly separating human-annotated evaluation from synthetic training.
- **Verdict: GOOD for thesis. Publishable as a resource paper.**

**Novelty Direction 2: Concept-Gap-Grounded Feedback (HIGH novelty, MEDIUM feasibility)**
- Contribution: Feedback generation that explicitly identifies missing concepts and grounds feedback in those gaps, rather than generating generic text
- Why it matters: Most ASAG work stops at grading. Feedback generation for ASAG is understudied. Grounded feedback is even rarer.
- Risk: Requires good concept-gap detection. Evaluation requires human judges.
- **Verdict: STRONGEST novelty direction. Should be the thesis highlight.**

**Novelty Direction 3: Adversarial Robustness Suite for ASAG (HIGH novelty, MEDIUM feasibility)**
- Contribution: First systematic evaluation of ASAG robustness with 12+ perturbation types across 3 model families
- Why it matters: Robustness evaluation is standard in NLP but almost absent in ASAG literature
- Risk: Perturbations are synthetic. Real gaming attempts may differ.
- **Verdict: STRONG novelty. Publishable as a standalone paper.**

**Novelty Direction 4: Multi-Task Grading + Misconception Mining (MEDIUM novelty, LOW feasibility)**
- Contribution: Joint model that grades AND identifies misconception type
- Why it matters: Connects grading to pedagogical understanding
- Risk: Requires reliable misconception labels for training. Only data-generate.csv has them, and they're synthetic.
- **Verdict: Interesting but risky. Keep as extension.**

**Novelty Direction 5: Calibration-Aware Grading (MEDIUM novelty, HIGH feasibility)**
- Contribution: Analyze and improve calibration of ASAG models; propose confidence-based abstention
- Why it matters: Calibration is critical for deployment but rarely studied in ASAG
- Risk: Low novelty ceiling — calibration analysis is a standard technique applied to a new domain
- **Verdict: GOOD addition to robustness chapter. Not standalone.**

### 20.3 Recommended Novelty Strategy

**Core novelty claim:** "We present a unified framework for automatic short-answer grading that goes beyond score prediction to provide concept-gap-grounded feedback, validated through a multi-granularity benchmark with adversarial robustness evaluation."

This combines Directions 1 + 2 + 3 into one coherent narrative.

---

## 21. Thesis Structure

### Recommended Structure

**Chapter 1: Introduction (15 pages)**
- Problem statement: Why ASAG matters, why grading alone is insufficient
- Research questions (all RQs from all 4 projects)
- Contributions summary
- Thesis outline

**Chapter 2: Related Work (20 pages)**
- 2.1 Automatic Short-Answer Grading (history, methods, benchmarks)
- 2.2 Student Misconception Detection in Education
- 2.3 Automatic Feedback Generation in Educational NLP
- 2.4 Adversarial Robustness in NLP (with focus on educational applications)
- 2.5 Calibration and Uncertainty in Classification Systems

**Chapter 3: Datasets and Unified Benchmark (20 pages)**
- 3.1 SciEntsBank: description, statistics, splits
- 3.2 MohlerASAG: description, statistics, split creation
- 3.3 Internal Generated Dataset: description, generation methodology, quality audit
- 3.4 Internal Scraped Dataset: description, limitations, filtering
- 3.5 Unified Schema Design
- 3.6 Label Harmonization Strategy
- 3.7 Data Quality Analysis and Limitations
- **This chapter IS a contribution.** Treat it as such.

**Chapter 4: Automatic Short-Answer Grading (25 pages) — CORE**
- 4.1 Task Formulations (2-way, 3-way, 5-way, regression)
- 4.2 Baselines and Proposed Models
- 4.3 Experimental Setup
- 4.4 Results on SciEntsBank (UA, UQ, UD)
- 4.5 Results on MohlerASAG
- 4.6 Cross-Domain Transfer Analysis
- 4.7 Synthetic Data Augmentation Analysis
- 4.8 Multi-Task Learning Results
- 4.9 Discussion

**Chapter 5: Misconception Mining (15 pages) — EXTENSION**
- 5.1 Methodology
- 5.2 Clustering Experiments
- 5.3 Validation Against Gold Misconception Tags
- 5.4 Qualitative Analysis of Discovered Patterns
- 5.5 Discussion and Limitations

**Chapter 6: Concept-Gap-Grounded Feedback Generation (25 pages) — CORE**
- 6.1 Feedback Pipeline Architecture
- 6.2 Concept Gap Detection
- 6.3 Feedback Generation Strategies
- 6.4 Automatic Evaluation Results
- 6.5 Human Evaluation Results
- 6.6 Error Analysis: Hallucination and Generic Feedback
- 6.7 Discussion

**Chapter 7: Robustness and Reliability Analysis (20 pages) — EXTENSION**
- 7.1 Perturbation Taxonomy
- 7.2 Experimental Setup
- 7.3 Performance Under Perturbation
- 7.4 Calibration Analysis
- 7.5 Model Family Vulnerability Comparison
- 7.6 Abstention and Confidence Thresholding
- 7.7 Discussion and Recommendations

**Chapter 8: Conclusion and Future Work (10 pages)**
- 8.1 Summary of Contributions
- 8.2 Limitations
- 8.3 Future Work (real student deployment, multilingual ASAG, LLM-as-judge scaling)
- 8.4 Broader Impact

**Total estimated length:** ~150 pages

### Structural Notes
- Chapters 4 and 6 are the core. They must be the strongest.
- Chapters 5 and 7 can be shorter. If time is limited, reduce scope but do not remove entirely.
- Chapter 3 (benchmark) is a contribution in itself. Do not treat it as "just data description."

---

# ════════════════════════════════════════════════════════════════
# PART H — PRACTICAL IMPLEMENTATION ROADMAP
# ════════════════════════════════════════════════════════════════

## 22. Staged Roadmap

### Phase 0: Data Audit (Week 1–2) — PRIORITY: CRITICAL

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Download and verify SciEntsBank | Local copy with verified splits | None | Low |
| Download and verify MohlerASAG | Local copy with score distributions | None | Low |
| Audit data-generate.csv quality | Quality report: 200-sample manual review | None | Medium — may reveal serious quality issues |
| Audit data-scraping.json usability | Filtered question bank (conceptual questions only) | None | Low |
| Verify no question overlap between datasets | Overlap report | All datasets loaded | Medium |

**Deliverable:** Data audit report documenting all issues found.

---

### Phase 1: Dataset Standardization + Benchmark Construction (Week 3–5) — PRIORITY: HIGH

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Implement unified schema | Python script: raw → unified format | Phase 0 | Low |
| Implement label harmonization | Mapping functions with unit tests | Phase 0 | Medium — edge cases in score→label mapping |
| Create MohlerASAG question-level splits | Split files | Phase 0 | Low |
| Verify adversarial variant linkage in splits | Validation script | Phase 1 schema | Medium |
| Build data loading library | Python package with train/test loaders | All above | Low |
| Write Chapter 3 draft | 15-page draft | All above | Low |

**Deliverable:** Unified dataset package + Chapter 3 draft.

---

### Phase 2: Grading Baselines + Strong Model (Week 6–12) — PRIORITY: HIGH

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Implement lexical overlap baseline | Results on SciEntsBank | Phase 1 | Low |
| Implement TF-IDF + ML baselines | Results on SciEntsBank + MohlerASAG | Phase 1 | Low |
| Implement SBERT similarity baseline | Results on all datasets | Phase 1 | Low |
| Fine-tune cross-encoder | Results on all datasets | Phase 1 | Medium — hyperparameter tuning |
| Fine-tune reference-answer-aware model | Results on all datasets | Phase 1 | Medium |
| Run LLM zero-shot evaluation | Results on test sets | Phase 1 | Low (cost: ~$50–100) |
| Augmentation ablation study | Comparison table | All baselines | Low |
| Cross-domain transfer experiments | Transfer matrix | All baselines | Low |
| Multi-task learning experiments | Comparison with single-task | Phase 2 models | Medium |
| Write Chapter 4 draft | 20-page draft | All above | Medium |

**Deliverable:** Complete grading results + Chapter 4 draft.

---

### Phase 3: Misconception Mining (Week 13–16) — PRIORITY: MEDIUM

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Implement embedding + clustering pipeline | Working pipeline | Phase 1 | Low |
| Run clustering experiments (3 embedding strategies × 3 methods) | Results table | Pipeline | Medium |
| Validate against gold misconception_tags | NMI, ARI scores | Clustering results | Low |
| Qualitative analysis of top clusters | Case study document | Clustering results | Medium — subjective |
| Write Chapter 5 draft | 12-page draft | All above | Low |

**Deliverable:** Misconception taxonomy + Chapter 5 draft.

---

### Phase 4: Feedback Generation (Week 13–20) — PRIORITY: HIGH

*Note: Can overlap with Phase 3 — different people or parallel work streams.*

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Implement concept-gap detector | Working module | Phase 2 grading model | Medium |
| Implement template-based feedback baseline | Results | Phase 1 | Low |
| Implement retrieval-based feedback baseline | Results | Phase 1 | Low |
| Fine-tune T5 for feedback generation | Working model | Phase 1 + concept-gap detector | Medium — training stability |
| Implement hybrid feedback pipeline | Working pipeline | All above | Medium |
| Automatic evaluation (ROUGE, BERTScore, concept coverage) | Results table | All pipelines | Low |
| Human evaluation (100 samples, 3 evaluators) | Evaluation report | All pipelines | HIGH — finding evaluators, time-consuming |
| Write Chapter 6 draft | 20-page draft | All above | Medium |

**Deliverable:** Feedback system + human evaluation report + Chapter 6 draft.

---

### Phase 5: Robustness Suite (Week 17–22) — PRIORITY: MEDIUM

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Evaluate all Phase 2 models on adversarial test set | Performance drop table | Phase 2 | Low |
| Generate programmatic perturbations for SciEntsBank | Perturbed test sets | Phase 1 | Medium |
| Compute calibration metrics | Reliability diagrams, ECE | Phase 2 models | Low |
| Build vulnerability matrix | Heatmap visualization | All above | Low |
| Abstention analysis | Accuracy-coverage curves | Phase 2 models | Low |
| Adversarial training ablation | Comparison table | Phase 2 + adversarial data | Medium |
| Write Chapter 7 draft | 15-page draft | All above | Low |

**Deliverable:** Robustness evaluation report + Chapter 7 draft.

---

### Phase 6: Thesis Writing + Paper Writing (Week 20–28) — PRIORITY: CRITICAL

| Task | Deliverable | Dependency | Risk |
|---|---|---|---|
| Write Chapter 1 (Introduction) | 15-page draft | All experiments | Low |
| Write Chapter 2 (Related Work) | 20-page draft | Literature review | Medium — time-consuming |
| Revise Chapters 3–7 based on results | Final drafts | All experiments | Medium |
| Write Chapter 8 (Conclusion) | 10-page draft | All chapters | Low |
| Internal review and revision | Revised thesis | Full draft | Medium |
| Prepare defense presentation | Slides | Final thesis | Low |

**Deliverable:** Complete thesis + defense presentation.

---

## 23. Minimum Viable Thesis vs. Extensions

### Minimum Publishable / Defensible Core (If Time Is Limited)

**Must do:**
1. ✅ Phase 0 + Phase 1: Data audit + unified benchmark (Chapter 3)
2. ✅ Phase 2: Grading experiments with all baselines (Chapter 4)
3. ✅ Phase 4 (partial): At least template + retrieval feedback baselines + concept-gap analysis (Chapter 6, reduced)
4. ✅ Phase 5 (partial): At least adversarial evaluation on data-generate.csv test_adversarial (Chapter 7, reduced)

**This gives you:** A thesis with 4 substantive chapters (benchmark + grading + feedback + robustness) that is defensible.

### Extensions (If Time Allows)
1. Full misconception mining with qualitative analysis (Chapter 5)
2. Full human evaluation for feedback (expensive but high impact)
3. Full robustness suite with programmatic perturbations on SciEntsBank
4. Multi-task learning experiments
5. LLM-as-judge analysis
6. Adversarial training experiments

### What to Cut First If Behind Schedule
1. Cut: Multi-task learning (nice-to-have, not essential)
2. Cut: LLM zero-shot evaluation (expensive, not the contribution)
3. Reduce: Misconception mining to a brief analysis section within Chapter 4
4. Reduce: Robustness to adversarial test set evaluation only (no programmatic perturbations)
5. NEVER cut: Grading baselines, feedback pipeline, unified benchmark

---

## 24. Summary Table

| # | Project Title (EN) | Objective | Datasets | Main Model | Baselines | Primary Metric | Novelty | Difficulty | Priority | Thesis Chapter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Multi-Granularity ASAG | Grade student answers at multiple granularities across domains | SciEntsBank, MohlerASAG, data-generate.csv | Reference-answer-aware cross-encoder (DeBERTa) | Lexical overlap, TF-IDF+ML, SBERT similarity, Cross-encoder, LLM zero-shot | Macro-F1 (classification), Pearson r (regression) | Cross-domain transfer + synthetic augmentation analysis | MEDIUM | HIGH (core) | Chapter 4 |
| 2 | Misconception Mining | Discover misconception patterns from incorrect answers | data-generate.csv (primary), SciEntsBank, MohlerASAG | SBERT + UMAP + HDBSCAN | TF-IDF+KMeans, SBERT+KMeans | NMI, ARI (extrinsic), Silhouette (intrinsic) | Embedding strategy comparison for ASAG misconceptions | MEDIUM | MEDIUM (extension) | Chapter 5 |
| 3 | Concept-Gap-Grounded Feedback | Generate targeted feedback based on identified concept gaps | data-generate.csv (primary), SciEntsBank (inference) | Hybrid: grading + concept-gap detector + T5 generator | Template-based, Retrieval-based, T5 ungrounded | Concept coverage, Human evaluation (5-point rubric) | Grounded feedback for ASAG (novel pipeline) | HIGH | HIGH (core) | Chapter 6 |
| 4 | Adversarial Robustness for ASAG | Evaluate model robustness under 12+ perturbation types | data-generate.csv (adversarial), SciEntsBank (programmatic) | All models from Project 1 | TF-IDF+SVM, Cross-encoder, LLM | Δ(Macro-F1), ECE, AUROC | First systematic ASAG robustness evaluation | MEDIUM-HIGH | MEDIUM (extension) | Chapter 7 |

---

# ════════════════════════════════════════════════════════════════
# FINAL SECTION — ADVISOR'S VERDICT
# ════════════════════════════════════════════════════════════════

## 1. Recommended Core Thesis Direction

**"Concept-Gap-Grounded Automatic Short-Answer Grading and Feedback Generation with Adversarial Robustness Evaluation"**

Core = Grading (Chapter 4) + Feedback (Chapter 6) + Unified Benchmark (Chapter 3).
Extensions = Misconception Mining (Chapter 5) + Full Robustness Suite (Chapter 7).

The strongest novelty is in the feedback pipeline. The grading chapter provides the foundation. The benchmark chapter provides the methodological contribution. The robustness chapter provides the critical evaluation. Together, this is a coherent, defensible master's thesis.

## 2. What to Do First This Week

1. **Download SciEntsBank and MohlerASAG.** Verify you can load them. Count rows. Check splits.
2. **Run the 200-sample quality audit on data-generate.csv.** Randomly sample 200 rows, manually check: Is the student_answer coherent? Does the label match? Is the feedback accurate? Document findings.
3. **Filter data-scraping.json.** Remove "Not found" entries and numerical questions. Count how many conceptual questions remain. Decide if it's worth keeping.
4. **Set up the project repository.** Create the unified schema loader. Write unit tests for label mapping.
5. **Implement the lexical overlap baseline on SciEntsBank.** This takes 2 hours and gives you your first result.

## 3. What Absolutely Should NOT Be Done

🚫 **Do NOT start with LLM experiments.** They are expensive, slow, and not your contribution. Do them last as analysis.

🚫 **Do NOT use data-generate.csv as your only test set.** It is synthetic. Any result on synthetic-only evaluation is academically weak.

🚫 **Do NOT skip the data quality audit.** The generated data has visible artifacts. If you build on unaudited data, your entire thesis is on shaky ground.

🚫 **Do NOT try to claim SOTA on SciEntsBank.** That is not the point. The point is the unified framework, the feedback pipeline, and the robustness analysis.

🚫 **Do NOT treat data-scraping.json as a usable dataset.** It has zero student answers. It is a question bank at best.

🚫 **Do NOT attempt all 4 projects at full depth simultaneously.** Do Projects 1 and 3 thoroughly. Do Projects 2 and 4 at reduced scope if time is limited.

🚫 **Do NOT generate more synthetic data before validating what you have.** Quality over quantity.

## 4. Final Verdict

**Is this a coherent master-level research program?**

**YES — conditionally.** The 4 projects form a logical stack (grade → understand errors → give feedback → test robustness). The dependency chain is sound. The datasets, despite their issues, are sufficient if handled correctly.

**The conditions:**
1. You must treat data-generate.csv as what it is: synthetic training/augmentation data, NOT ground truth. All final evaluations must include human-annotated benchmarks (SciEntsBank, MohlerASAG).
2. You must conduct the quality audit before building on the generated data.
3. You must scope Projects 2 and 4 as extensions, not equal-weight chapters. Trying to do all 4 at full depth will result in 4 shallow chapters instead of 2 strong ones.
4. The feedback pipeline (Project 3) is where your thesis lives or dies. Invest the most effort there.

**If you follow this plan:** You will have a defensible thesis with genuine academic contribution, a clear narrative, and realistic scope. The unified benchmark + grounded feedback + robustness evaluation combination is novel enough for a master's thesis and potentially yields 1–2 workshop/conference papers.

**If you ignore the warnings about synthetic data quality:** You will have a thesis that looks impressive on paper but collapses under reviewer scrutiny. Do the audit.

---
*Research plan prepared with strict but realistic academic standards. April 2026.*
