# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base`
- Arabic fertility evidence:XLM-R achieved an Arabic fertility of 1.672, lower than mBERT (2.153) and DistilBERT (4.527), although CAMeLBERT performed best at 1.405.
- English fertility evidence:XLM-R achieved an English fertility of 1.434, providing strong English tokenization while maintaining good Arabic performance.
- p95 length evidence: Its p95 sequence lengths were 21 tokens for Arabic and 23 tokens for English, showing short and balanced sequences across both languages.
- Operational trade-off / rationale: XLM-R provides the best overall balance for Bayan’s bilingual Arabic and English data. CAMeLBERT performs better on Arabic but produces much longer English sequences, while DistilBERT performs poorly on Arabic. Using one multilingual checkpoint also simplifies training and deployment.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
