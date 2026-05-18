# Human-as-Judge 对齐实验报告

有效人工标注样本数：40

### 整体与按模型指标

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| overall | 40 | 0.6500 | 0.3500 | 0.8750 | 0.5000 | 0.3000 |
| Qwen3-VL-235B-A22B-Instruct-cot | 10 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | -0.2000 |
| Qwen3.5-35B-A3B-cot | 10 | 0.7000 | 0.4000 | 1.0000 | 0.5714 | 0.4000 |
| gemini-2.5-flash-cot | 10 | 0.6000 | 0.2000 | 1.0000 | 0.3333 | 0.2000 |
| gpt-5.4-mini-cot | 10 | 0.9000 | 0.8000 | 1.0000 | 0.8889 | 0.8000 |

### 按模型 × Human Type

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| Qwen3-VL-235B-A22B-Instruct-cot / logical | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-35B-A3B-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-35B-A3B-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / faithfulness | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 按模型 × GPT Type

| 子集 | N | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|------|---:|---------:|----------:|-------:|---:|--------------:|
| Qwen3-VL-235B-A22B-Instruct-cot / faithfulness | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3-VL-235B-A22B-Instruct-cot / factuality | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3-VL-235B-A22B-Instruct-cot / logical | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-35B-A3B-cot / faithfulness | 3 | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0.0000 |
| Qwen3.5-35B-A3B-cot / factuality | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-35B-A3B-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / faithfulness | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gemini-2.5-flash-cot / logical | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gpt-5.4-mini-cot / faithfulness | 3 | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.0000 |
| gpt-5.4-mini-cot / factuality | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-5.4-mini-cot / logical | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 幻觉类型一致性

Type agreement: 0.8571 （仅限人和 GPT 均判定有幻觉的样本）

| Human \ GPT | faithfulness | factuality | logical |
|-------------|-------------:|----------:|--------:|
| faithfulness | 2 | 0 | 0 |
| factuality | 1 | 2 | 0 |
| logical | 0 | 0 | 2 |

### Mismatch 清单

Total mismatches: 14

| sample_id | model | pid | human_label | gpt_has_h | human_type | gpt_type | notes |
|-----------|-------|-----|------------:|----------:|------------|----------|-------|
| Qwen3_VL_235B_A22B_Instruct_cot__871 | Qwen3-VL-235B-A22B-Instruct-cot | 871 | 1 | 0 | logical | none |  |
| Qwen3_VL_235B_A22B_Instruct_cot__115 | Qwen3-VL-235B-A22B-Instruct-cot | 115 | 0 | 1 | none | logical |  |
| Qwen3_VL_235B_A22B_Instruct_cot__208 | Qwen3-VL-235B-A22B-Instruct-cot | 208 | 0 | 1 | none | logical |  |
| Qwen3_VL_235B_A22B_Instruct_cot__395 | Qwen3-VL-235B-A22B-Instruct-cot | 395 | 0 | 1 | none | faithfulness |  |
| Qwen3_VL_235B_A22B_Instruct_cot__652 | Qwen3-VL-235B-A22B-Instruct-cot | 652 | 0 | 1 | none | factuality |  |
| Qwen3_VL_235B_A22B_Instruct_cot__815 | Qwen3-VL-235B-A22B-Instruct-cot | 815 | 0 | 1 | none | faithfulness |  |
| Qwen3_5_35B_A3B_cot__130 | Qwen3.5-35B-A3B-cot | 130 | 0 | 1 | none | faithfulness |  |
| Qwen3_5_35B_A3B_cot__366 | Qwen3.5-35B-A3B-cot | 366 | 0 | 1 | none | faithfulness |  |
| Qwen3_5_35B_A3B_cot__745 | Qwen3.5-35B-A3B-cot | 745 | 0 | 1 | none | factuality |  |
| gemini_2_5_flash_cot__438 | gemini-2.5-flash-cot | 438 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__711 | gemini-2.5-flash-cot | 711 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__796 | gemini-2.5-flash-cot | 796 | 0 | 1 | none | logical |  |
| gemini_2_5_flash_cot__859 | gemini-2.5-flash-cot | 859 | 0 | 1 | none | faithfulness |  |
| gpt_5_4_mini_cot__325 | gpt-5.4-mini-cot | 325 | 0 | 1 | none | faithfulness |  |

### 错误模式归因表

该表用于人工复核 mismatch 后填写。

| 模式 | 样本数 | 说明 | 典型样例 ID |
|------|-------:|------|------------|
| MLLM Judge 漏检 |  |  |  |
| MLLM Judge 过判 |  |  |  |
| 类型错位 |  |  |  |
| 图像/问题歧义 |  |  |  |
| 其他 |  |  |  |
