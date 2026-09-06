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
- Decision: Clean and remove HTML remnants..

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

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
