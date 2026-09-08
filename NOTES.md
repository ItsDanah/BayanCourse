# Lab Notes
.venv\Scripts\Activate.ps1
## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: PII
- Example: لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة 😡 <br> 0551234567 1023456789   
- Why it matters: Personal information should not be exposed in text used for NLP processing.
- Decision: mask sensitive values while preserving their general type.

### Defect 2
- Class: Code-switching Arabic ↔ English
- Example: يوجد تسرب مياه في الدمام والبلاغ رقم BYN-2026-000035   
- Why it matters: Mixed-language content affects how text is tokenized and represented.
- Decision: Preserve, English content may carry important information.

### Defect 3
- Class: Emoji
- Example: الحاوية ممتلئة في الرياض ولم تُفرغ منذ 6 أيام 😡
- Why it matters: Emoji can carry useful information about sentiment and user intent.
- Decision: Preserve could be useful for sentiment analysis.

### Defect 4
- Class: HTML remnants
- Example: الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة <br>
- Why it matters: HTML markup tags are for formatting rather than part of the natural-language feedback.
- Decision: Clean and remove HTML remnants.

### Defect 5
- Class: Unicode forms
- Example: طلب ترخيص موأعيد الصيانة متوقف عند المراجعة منذ 2 أيام
- Why it matters: Different representations can cause visually similar text to be processed as different forms.
- Decision: Clean and apply consistent Unicode normalization while preserving meaningful text.

### Defect 6
- Class: Repeated letters
- Example: لووووسمحت دفعت الفاتورة لكن الحالة ما زالت غير مسددة
- Why it matters: Character repetition can create inconsistent forms of the word and affect tokenization.
- Decision: Task-dependent, reduce excessive repeated characters while keeping the original word recognizable.

## Lab 1 — Sentence Segmentation

I manually spot-checked five examples using the spaCy segmentation pipeline.

1. An Arabic complaint was separated using periods, exclamation marks, and
   Arabic question marks.
2. The abbreviation `Dr.` was preserved and did not create an incorrect
   sentence boundary.
3. Mixed Arabic and English text was segmented successfully.
4. Emoji were preserved during preprocessing and segmentation.
5. In the numbered-list complaint, spaCy returned the numbered markers as
   separate segments.

The pipeline applied the shared preprocessing function and returned only
non-empty sentence strings.

## Lab 2 — Parameter audit
Attention
Custom scaled dot-product attention matches PyTorch at atol=1e-6.
4-token toy sequence produces a 4x4 attention matrix.
Attention weights from random Q/K/V are not linguistically meaningful.
Multi-Head Attention
d_model = 768
n_heads = 12
d_k = 64
MHA parameters = 2,362,368
Input shape = [1, 4, 768]
Output shape = [1, 4, 768]
Parameter Audit
mBERT total = 177.85M
mBERT embeddings = 92.21M (51.8%)
CAMeLBERT total = 109.08M
CAMeLBERT embeddings = 23.43M (21.5%)
Attention and FFN sizes are nearly identical across both models.
Main size difference comes from vocabulary / embedding table: multilingual tax.
Causal Mask
Lower-triangular attention mask passed.
Each token can attend only to itself and previous positions.
This is decoder-style causal attention.
Attention Diagnostics
Candidate adjacency heads were observed across the three Arabic sentences.
Candidate [SEP]-sink heads were observed in deeper layers.
Attention maps are diagnostics, not causal explanations.
PAD Leak
PAD attention mass WITH mask: 0.000000
PAD attention mass WITHOUT mask: 0.253806
Regression check passed: PAD mass with mask < 0.01.
Missing attention masks can create silent numerical errors without crashing.
Interesting tokenizer observation
"بطيئًا" produced [UNK] in one example.
Tokenisation problems happen before attention and cannot be repaired downstream.
Lab 4 — Dialect audit

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
