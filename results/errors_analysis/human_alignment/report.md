# Human-as-Judge 对齐实验报告

有效人工标注样本数：40

### 整体与按模型指标

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| overall | 40 | 0.8500 | 0.7000 | 1.0000 | 0.8235 | 0.7000 |
| Qwen3-VL-235B-A22B-Instruct-cot | 10 | 0.8000 | 0.6000 | 1.0000 | 0.7500 | 0.6000 |
| Qwen3.5-35B-A3B-cot | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot | 10 | 0.6000 | 0.2000 | 1.0000 | 0.3333 | 0.2000 |
| gpt-5.4-mini-cot | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 按模型 × Human Type

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| Qwen3-VL-235B-A22B-Instruct-cot / faithfulness | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3-VL-235B-A22B-Instruct-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / faithfulness | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / logical | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / faithfulness | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 按模型 × GPT Type

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| Qwen3-VL-235B-A22B-Instruct-cot / faithfulness | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3-VL-235B-A22B-Instruct-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3-VL-235B-A22B-Instruct-cot / logical | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-35B-A3B-cot / faithfulness | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / faithfulness | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / logical | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gpt-5.4-mini-cot / faithfulness | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 幻觉类型一致性

Type agreement: 0.9286 （仅限人和 GPT 均判定有幻觉的样本）

| Human \ GPT | faithfulness | factuality | logical |
|-------------|-------------:|----------:|--------:|
| faithfulness | 7 | 0 | 0 |
| factuality | 0 | 4 | 0 |
| logical | 1 | 0 | 2 |

### Mismatch 清单

Total mismatches: 6

| sample_id | model | pid | human_label | gpt_has_h | human_type | gpt_type | notes |
|-----------|-------|-----|------------:|----------:|------------|----------|-------|
| Qwen3_VL_235B_A22B_Instruct_cot__115 | Qwen3-VL-235B-A22B-Instruct-cot | 115 | 0 | 1 | none | logical |  |
| Qwen3_VL_235B_A22B_Instruct_cot__208 | Qwen3-VL-235B-A22B-Instruct-cot | 208 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__438 | gemini-2.5-flash-cot | 438 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__711 | gemini-2.5-flash-cot | 711 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__796 | gemini-2.5-flash-cot | 796 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__859 | gemini-2.5-flash-cot | 859 | 0 | 1 | none | faithfulness |  |

### 错误模式归因表

| 模式 | 样本数 | 说明 | 典型样例 ID |
|------|-------:|------|------------|
| MLLM Judge 过判 | 6 | 模型合理拒答年龄估算问题，Judge 判为 logical 幻觉 | Qwen3_VL_235B_A22B_Instruct_cot__115, gemini_2_5_flash_cot__438 |
| MLLM Judge 漏检 | 0 | — | — |
| 类型错位 | 1 | 人判 logical，GPT 判 faithfulness（模型误认人物身份） | Qwen3_5_35B_A3B_cot__53 |
