# OCRBench 幻觉评估设计文档

## 1. 背景与动机

### 1.1 为什么评估 OCR 幻觉？

OCR（光学字符识别）是多模态大模型（MLLM）的基础能力之一。在 VQA 场景中，模型需要：

1. **识别**图像中的文字（如路牌、菜单、文档、公式）
2. **理解**文字与图像上下文的关系
3. **推理**基于识别结果回答问题

OCR 环节的微小错误（如将 "10:00 AM" 识别为 "10:00 PM"）会导致下游推理结果完全错误，即**OCR 级联幻觉**。评估 OCR 场景下的幻觉率能反映模型在文本密集型视觉任务中的可靠性。

### 1.2 与现有任务的差异

| 维度 | POPE | MathVista | OCR |
|------|------|-----------|-----|
| 任务类型 | Yes/No 二分类 | 数学推理 | 文本识别 + VQA |
| 检测方法 | 规则法 | GPT Judge | GPT Judge |
| 幻觉触发点 | 对象存在性 | 推理过程 | **文字误读/捏造** |
| 答案形式 | yes/no | 数值/选项 | 自由文本/识别结果 |

OCR 的特殊性在于：文本是图像中的**精确信息载体**。模型可能"读出"不存在于图中的文字（视觉忠实性错误），或正确读文但推理出错——比纯视觉 VQA 多了一层"文本准确度"维度。

---

## 2. 数据集：OCRBench v1

### 2.1 数据来源

**OCRBench** (Liu et al., 2023) 是 MLLM OCR 能力评估的标准基准，包含 **1000 条**人工标注样本。

- 论文：*OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models* (Science China Information Sciences, 2024)
- 代码：https://github.com/Yuliang-Liu/MultimodalOCR
- 许可：MIT

### 2.2 任务组成

| 任务组 | question_type | 样本数 | 评估价值 |
|-------|---------------|--------|---------|
| **Text Recognition** | Regular / Irregular / Artistic / Handwriting / Digit String / Non-Semantic | 300 | 基础 OCR 能力，纯识别准确率 |
| **Scene Text VQA** | Scene Text-centric VQA | 200 | **高**：模型需结合场景理解与文字识别 |
| **Document VQA** | Doc-oriented VQA | 200 | **高**：文档级理解中的幻觉检测 |
| **KIE** | Key Information Extraction | 200 | **高**：结构化信息提取中的幻觉 |
| **HMER** | Handwritten Mathematical Expression Recognition | 100 | 专业公式识别，LaTeX 输出 |

### 2.3 数据格式

OCRBench 数据为 JSON 格式，每条包含：

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `id` | int | `1` | 唯一标识 |
| `image_path` | str | `images/iiit5k/word_123.png` | 图像相对路径 |
| `question` | str | `"what is written in the image?"` | 问题 |
| `answers` | str/list[str] | `["CENTRE"]` | 标准答案（可能有多个变体） |
| `type` | str | `"Regular Text Recognition"` | 任务类型 |
| `dataset_name` | str | `"IIIT5K"` | 来源数据集 |

### 2.4 答案格式与规范化

OCRBench 的 `answers` 字段可能是字符串或列表（多个有效答案变体）。在加载器 `load_ocrbench` 中：

- 若 `answers` 为列表：取 `answers[0]` 作为标准答案
- 始终 `str()` 转换确保类型一致
- 保留 `task_type` 和 `dataset_name` 字段用于后续按子任务分层分析

---

## 3. 幻觉类型适配

**核心原则：不新增幻觉类型，沿用现有三分类体系**（Faithfulness / Factuality / Logical），在 GPT Judge prompt 中做 OCR 特化。

### 3.1 OCR 场景 → 现有类型映射

| OCR 场景 | 示例 | 映射类型 | 判定依据 |
|----------|------|----------|----------|
| 模型"读出"图中不存在的文字 | 图中路牌只有 "STOP"，模型输出 "SLOW" | **忠实性**（Faithfulness） | 视觉内容不一致——文本是图像的一部分 |
| 模型正确读文但替换错字 | "OPEN 24 HOURS" 读成 "OPEN 25 HOURS" | **忠实性**（Faithfulness） | 属性级错误（文本内容） |
| 模型正确读文但与知识矛盾 | 读出厂商标 "CoCa-Cola"，回答 "Pepsi" | **事实性**（Factuality） | 与世界知识矛盾 |
| 模型正确读文但推理出错 | 读取 "全场五折"，推论 "半价出售，买一送一" | **逻辑性**（Logical） | 推理过程有误 |
| 模型正确读文并正确推理 | 读取 "A=30, B=50"，回答 "差值为20" | **无幻觉** | — |

### 3.2 OCR 幻觉的核心特征

OCR 幻觉主要集中在**忠实性（Faithfulness）**子类的**属性级**错误中，文本作为图像的精确视觉信息元素，模型的错误输出表现为：

1. **文字替代**：将图中文字 A 识别为文字 B（字母级错误）
2. **文字捏造**：输出完全不存在的文字内容
3. **文字遗漏**：遗漏图像中的部分文字
4. **数字误读**：特别易发生在数字串、价格、日期等场景

---

## 4. 检测方法

### 4.1 主检测：GPT Judge

复用现有 **MMHal-Bench 0-6 评分协议**（`evaluation/detectors/gpt_judge.py`），针对 OCR 场景做 prompt 特化：

- **任务描述**：提示 Judge 当前是 OCR 任务，要求关注文字级准确性
- **OCR 提醒**：要求 Judge 检查模型是否正确读取了图像中的文字
- **幻觉类型分类**：启用三分类（faithfulness / factuality / logical）

### 4.2 评估指标

| 指标 | 来源 | 说明 |
|------|------|------|
| **Hallucination Rate** | GPT Judge (score < 3) | 幻觉率，同 MathVista |
| **Average Score** | GPT Judge | 0-6 平均分 |
| **Type breakdown** | GPT Judge | 三类幻觉的分布比例 |
| **子任务分层指标** | 按 task_type 聚合 | 不同 OCR 任务类型的幻觉率差异 |

### 4.3 与 MathVista 的差异

| 方面 | MathVista | OCR |
|------|-----------|-----|
| 幻觉类型分类 | 仅 COT 开启 | **始终开启**（OCR 场景自动启用分类） |
| Judge prompt | 数学推理描述 | OCR 文本识别描述 + 文字级准确性提醒 |
| 答案格式 | 数值/选项 | 自由文本（识别结果） |

---

## 5. 架构设计

### 5.1 文件结构

```
hallucination-of-mllm/
├── data/
│   ├── ocr_loader.py          # OCRBench 数据加载器 (新增)
│   └── ocr/
│       ├── OCRBench.json       # 数据标注文件（需下载）
│       └── images/             # 图像文件（需下载）
│
├── evaluation/
│   ├── detectors/
│   │   └── gpt_judge.py        # 修改：增加 OCR prompt 分支
│   └── runners/
│       └── ocr.py              # OCR 评估 runner (新增)
│
├── configs/
│   └── config.py               # 修改：添加 OCR 路径
│
└── main.py                     # 修改：增加 OCR 数据集支持
```

### 5.2 数据流

```
data/ocr_loader.py
    ↓ load_ocrbench()
list[dict]  # id, image, question, answer, task_type, dataset_name
    ↓
evaluation/runners/ocr.py
    ├─ load_response_subset()  → 过滤有模型回答的样本
    ├─ GPTJudge.judge()        → 逐条评估（OCR 特化 prompt）
    └─ run_resumable_batch()   → 批处理 + 断点续跑
    ↓
results/{model}_ocr.json      # 评估结果
```

### 5.3 与现有流程的关系

```
generate_responses.py  →  OCRBench 模型回答（与现有流程相同）
                               ↓
main.py --dataset ocr  →  run_ocr()  →  GPT Judge 评估
                               ↓
                        results/{model}_ocr.json
```

---

## 6. 使用方式

### 6.1 数据准备

```bash
# 从 GitHub 下载 OCRBench 数据和图像
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git
cp MultimodalOCR/OCRBench/OCRBench.json data/ocr/
cp -r MultimodalOCR/OCRBench/images data/ocr/
```

### 6.2 生成模型回答

```bash
python scripts/generate_responses.py \
    --dataset mathvista \
    --model gpt-5.4-mini \
    --output responses/gpt5.4-mini_ocr.json
```

> OCRBench 的 question 格式是通用的 VQA 形式，现有 prompt 可直接使用。

### 6.3 运行 GPT Judge 评估

```bash
python main.py \
    --dataset ocr \
    --model gpt-5.4-mini \
    --response-files gpt-5.4-mini:ocr=responses/gpt5.4-mini_ocr.json \
    --workers 4
```

### 6.4 多模型同时评估

```bash
python main.py \
    --dataset all \
    --response-files \
    gpt-5.4-mini:ocr=responses/gpt5.4-mini_ocr.json \
    gemini-2.5-flash:ocr=responses/gemini-2.5-flash_ocr.json \
    Qwen3.5-35B-A3B:ocr=responses/Qwen3.5-35B-A3B_ocr.json \
    Qwen3-VL-235B-A22B-Instruct:ocr=responses/Qwen3-VL-235B-A22B-Instruct_ocr.json \
    --workers 4
```

---

## 7. 扩展方向

### 7.1 按子任务分层分析

OCRBench 包含 10 个 question_type，可分析模型在不同 OCR 子任务上的幻觉率差异：
- Text Recognition 任务：反映基础 OCR 能力
- Scene Text VQA：反映场景理解 + OCR 结合能力
- Document VQA：反映文档级 OCR 能力
- KIE：反映结构化信息提取能力

### 7.2 错误归因分类

参考 MathVista human-as-judge 实验设计，对 OCR 幻觉的 mismatch 案例按以下模式归因：

| 错误模式 | 说明 |
|----------|------|
| 文字误读 | 模型错误识别文字字符 |
| 文字捏造 | 模型输出不存在的文字内容 |
| 上下文误解 | 模型正确读文但未结合视觉上下文 |
| 推理跳步 | 模型正确读文但跳过必要的推理步骤 |

### 7.3 文本精确度补充指标

对 Text Recognition 和 KIE 任务，可引入字符错误率（CER）作为 GPT Judge 的补充指标：

```
CER = edit_distance(predicted_text, ground_truth) / len(ground_truth)
```

这能精确量化模型的文字识别精确度，与 GPT Judge 的主观评分形成互补。

---

## 8. 参考文献

1. Liu, Y., Li, Z., Huang, M., et al. (2024). OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models. *Science China Information Sciences*. arXiv:2305.07895.
2. Sun, Z., Shen, S., Cao, S., et al. (2024). Aligning Large Multimodal Models with Factually Augmented RLHF. *ACL Findings*. arXiv:2309.14525.