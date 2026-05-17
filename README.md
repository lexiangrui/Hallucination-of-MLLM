# MLLM Hallucination Detection

自动化多模态大模型幻觉检测与评估工具。

## 目录结构

```
.
├── main.py                   # 主入口，命令行运行
├── configs/
│   └── config.py             # 全局配置 (API Keys, 路径, 模型列表)
├── data/
│   ├── __init__.py
│   ├── pope_loader.py        # POPE 数据集加载
│   ├── mathvista_loader.py   # MathVista 数据集加载
│   └── vqarad_loader.py      # VQA-RAD 数据集加载
├── evaluation/
│   ├── __init__.py              # 导出 judge / run_* 接口
│   ├── judge.py                 # GPT Judge 裁判逻辑
│   ├── metrics.py               # 检测指标计算
│   ├── rule_based.py            # POPE 规则判断法
│   ├── run_mathvista.py         # MathVista 评测运行
│   ├── run_pope.py              # POPE 评测运行
│   └── run_vqarad.py            # VQA-RAD 评测运行
├── scripts/
│   ├── generate_responses.py # 调用测试模型 API 生成回答 JSON
│   ├── export_human_eval.py
│   └── analyze_human_alignment.py
├── utils/
│   ├── api.py                # 模型 API 调用 (OpenAI Chat / Responses / Anthropic Messages)
│   └── batch.py              # 断点续跑、并发、JSON 与图片工具
├── responses/                # 模型回答文件
└── results/                  # 评估结果和日志
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai anthropic Pillow pyarrow pandas python-dotenv
```

### 2. 配置 API Key

模型与 API key、base URL、调用方式的对应关系集中配置在 `configs/config.py` 的 `MODEL_API_CONFIGS`。每个模型条目直接填写自己的 `api_key` 和 `base_url`，`api_method` 支持四种协议：

| api_method | 协议 | SDK | 适用模型 |
|------------|------|-----|----------|
| `chat` | OpenAI Chat Completions | `openai` | Qwen3.5-35B-A3B / Qwen3-VL-235B-A22B-Instruct / gemini-2.5-flash |
| `responses` | OpenAI Responses | `openai` | gpt-5.4-mini |
| `messages` | Anthropic Messages | `anthropic` | （保留接口，当前未配置模型） |

`utils/api.py` 通过 `ModelClient` 统一封装四种协议，调用方只需传模型名，无需关心底层差异。图片格式通过 PIL 从文件内容自动检测，兼容扩展名与实际格式不一致的情况。

### 3. 准备数据

下载数据集并放置于对应目录：

- **POPE**: 从 https://github.com/RUCAIBox/POPE 下载，放置于 `data/POPE/`
- **MathVista**: 从 https://mathvista.github.io/ 下载，放置于 `data/MathVista/`
- **VQA-RAD**: 从 https://huggingface.co/datasets/flaviagiammarino/vqa-rad 下载，放置于 `data/VQA-RAD/`

### 4. 生成模型回答

先调用测试模型 API 生成回答文件。输出格式为 `{样本ID: 模型回答}`，可直接传给 `main.py`。

POPE 默认按每个 split 随机抽样 1000 条（seed=42，方便复现），下面 POPE 命令显式写出 `--max-samples 1000`。MathVista 默认使用 testmini split（1000 条样本）。VQA-RAD 使用 Hugging Face Parquet 的 test split（451 条样本），无需额外采样。

#### POPE random

```bash
python scripts/generate_responses.py --dataset pope --pope-split random \
  --model gpt-5.4-mini \
  --output responses/gpt5.4-mini_pope_random.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split random \
  --model gemini-2.5-flash \
  --output responses/gemini-2.5-flash_pope_random.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split random \
  --model Qwen3.5-35B-A3B \
  --output responses/Qwen3.5-35B-A3B_pope_random.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split random \
  --model Qwen3-VL-235B-A22B-Instruct \
  --output responses/Qwen3-VL-235B-A22B-Instruct_pope_random.json \
  --max-samples 1000 \
  --workers 4
```

#### POPE popular

```bash
python scripts/generate_responses.py --dataset pope --pope-split popular \
  --model gpt-5.4-mini \
  --output responses/gpt5.4-mini_pope_popular.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split popular \
  --model gemini-2.5-flash \
  --output responses/gemini-2.5-flash_pope_popular.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split popular \
  --model Qwen3.5-35B-A3B \
  --output responses/Qwen3.5-35B-A3B_pope_popular.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split popular \
  --model Qwen3-VL-235B-A22B-Instruct \
  --output responses/Qwen3-VL-235B-A22B-Instruct_pope_popular.json \
  --max-samples 1000 \
  --workers 4
```

#### POPE adversarial

```bash
python scripts/generate_responses.py --dataset pope --pope-split adversarial \
  --model gpt-5.4-mini \
  --output responses/gpt5.4-mini_pope_adversarial.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split adversarial \
  --model gemini-2.5-flash \
  --output responses/gemini-2.5-flash_pope_adversarial.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split adversarial \
  --model Qwen3.5-35B-A3B \
  --output responses/Qwen3.5-35B-A3B_pope_adversarial.json \
  --max-samples 1000 \
  --workers 4

python scripts/generate_responses.py --dataset pope --pope-split adversarial \
  --model Qwen3-VL-235B-A22B-Instruct \
  --output responses/Qwen3-VL-235B-A22B-Instruct_pope_adversarial.json \
  --max-samples 1000 \
  --workers 4
```

#### MathVista direct

```bash
python scripts/generate_responses.py --dataset mathvista \
  --model gpt-5.4-mini \
  --output responses/gpt5.4-mini_mathvista.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model gemini-2.5-flash \
  --output responses/gemini-2.5-flash_mathvista.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model Qwen3.5-35B-A3B \
  --output responses/Qwen3.5-35B-A3B_mathvista.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model Qwen3-VL-235B-A22B-Instruct \
  --output responses/Qwen3-VL-235B-A22B-Instruct_mathvista.json \
  --workers 10
```

#### MathVista CoT

```bash
python scripts/generate_responses.py --dataset mathvista \
  --model gpt-5.4-mini \
  --prompt-mode cot \
  --output responses/gpt5.4-mini_mathvista_cot.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model gemini-2.5-flash \
  --prompt-mode cot \
  --output responses/gemini-2.5-flash_mathvista_cot.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model Qwen3.5-35B-A3B \
  --prompt-mode cot \
  --output responses/Qwen3.5-35B-A3B_mathvista_cot.json \
  --workers 4

python scripts/generate_responses.py --dataset mathvista \
  --model Qwen3-VL-235B-A22B-Instruct \
  --prompt-mode cot \
  --output responses/Qwen3-VL-235B-A22B-Instruct_mathvista_cot.json \
  --workers 10
```

#### VQA-RAD

```bash
python scripts/generate_responses.py --dataset vqarad \
  --model gpt-5.4-mini \
  --output responses/gpt5.4-mini_vqarad.json \
  --workers 10

python scripts/generate_responses.py --dataset vqarad \
  --model gemini-2.5-flash \
  --output responses/gemini-2.5-flash_vqarad.json \
  --workers 10

python scripts/generate_responses.py --dataset vqarad \
  --model Qwen3.5-35B-A3B \
  --output responses/Qwen3.5-35B-A3B_vqarad.json \
  --workers 10

python scripts/generate_responses.py --dataset vqarad \
  --model Qwen3-VL-235B-A22B-Instruct \
  --output responses/Qwen3-VL-235B-A22B-Instruct_vqarad.json \
  --workers 10
```

生成回答默认会从已有输出文件断点续跑，并每 10 条自动保存一次；命令行只需要按 API 限流情况调整 `--workers`。MathVista 默认 prompt 只要求输出最终答案，不输出推理过程；需要推理链时再手动加 `--prompt-mode cot`。脚本不再设置输出 token 上限，主要通过 prompt 控制回答长度。

回答文件格式如下：

```json
{
    "0": "Yes, there is a chair in the image.",
    "1": "No, I don't see a cat.",
    "2": "Yes, the person is wearing a red shirt."
}
```

### 5. 运行检测

POPE 评测同样使用每个 split 1000 条随机样本（seed=42）。评测会自动只使用回答文件中已有的样本；MathVista 和 VQA-RAD 会调用 GPT Judge，POPE 不需要裁判模型 API。

#### POPE random

```bash
python main.py --dataset pope --model Qwen3.5-35B-A3B --pope-split random \
  --response-files Qwen3.5-35B-A3B:pope=responses/Qwen3.5-35B-A3B_pope_random.json \
  --max-samples 1000

python main.py --dataset pope --model Qwen3-VL-235B-A22B-Instruct --pope-split random \
  --response-files Qwen3-VL-235B-A22B-Instruct:pope=responses/Qwen3-VL-235B-A22B-Instruct_pope_random.json \
  --max-samples 1000

python main.py --dataset pope --model gpt-5.4-mini --pope-split random \
  --response-files gpt-5.4-mini:pope=responses/gpt5.4-mini_pope_random.json \
  --max-samples 1000

python main.py --dataset pope --model gemini-2.5-flash --pope-split random \
  --response-files gemini-2.5-flash:pope=responses/gemini-2.5-flash_pope_random.json \
  --max-samples 1000
```

#### POPE popular

```bash
python main.py --dataset pope --model Qwen3.5-35B-A3B --pope-split popular \
  --response-files Qwen3.5-35B-A3B:pope=responses/Qwen3.5-35B-A3B_pope_popular.json \
  --max-samples 1000

python main.py --dataset pope --model Qwen3-VL-235B-A22B-Instruct --pope-split popular \
  --response-files Qwen3-VL-235B-A22B-Instruct:pope=responses/Qwen3-VL-235B-A22B-Instruct_pope_popular.json \
  --max-samples 1000

python main.py --dataset pope --model gpt-5.4-mini --pope-split popular \
  --response-files gpt-5.4-mini:pope=responses/gpt5.4-mini_pope_popular.json \
  --max-samples 1000

python main.py --dataset pope --model gemini-2.5-flash --pope-split popular \
  --response-files gemini-2.5-flash:pope=responses/gemini-2.5-flash_pope_popular.json \
  --max-samples 1000
```

#### POPE adversarial

```bash
python main.py --dataset pope --model Qwen3.5-35B-A3B --pope-split adversarial \
  --response-files Qwen3.5-35B-A3B:pope=responses/Qwen3.5-35B-A3B_pope_adversarial.json \
  --max-samples 1000

python main.py --dataset pope --model Qwen3-VL-235B-A22B-Instruct --pope-split adversarial \
  --response-files Qwen3-VL-235B-A22B-Instruct:pope=responses/Qwen3-VL-235B-A22B-Instruct_pope_adversarial.json \
  --max-samples 1000

python main.py --dataset pope --model gpt-5.4-mini --pope-split adversarial \
  --response-files gpt-5.4-mini:pope=responses/gpt5.4-mini_pope_adversarial.json \
  --max-samples 1000

python main.py --dataset pope --model gemini-2.5-flash --pope-split adversarial \
  --response-files gemini-2.5-flash:pope=responses/gemini-2.5-flash_pope_adversarial.json \
  --max-samples 1000
```

#### MathVista direct

```bash
python main.py --dataset mathvista --model Qwen3.5-35B-A3B \
  --response-files Qwen3.5-35B-A3B:mathvista=responses/Qwen3.5-35B-A3B_mathvista.json \
  --workers 4

python main.py --dataset mathvista --model Qwen3-VL-235B-A22B-Instruct \
  --response-files Qwen3-VL-235B-A22B-Instruct:mathvista=responses/Qwen3-VL-235B-A22B-Instruct_mathvista.json \
  --workers 4

python main.py --dataset mathvista --model gpt-5.4-mini \
  --response-files gpt-5.4-mini:mathvista=responses/gpt5.4-mini_mathvista.json \
  --workers 4

python main.py --dataset mathvista --model gemini-2.5-flash \
  --response-files gemini-2.5-flash:mathvista=responses/gemini-2.5-flash_mathvista.json \
  --workers 4
```

#### MathVista CoT

```bash
python main.py --dataset mathvista --model gpt-5.4-mini-cot \
  --response-files gpt-5.4-mini-cot:mathvista=responses/gpt5.4-mini_mathvista_cot.json \
  --workers 4

python main.py --dataset mathvista --model gemini-2.5-flash-cot \
  --response-files gemini-2.5-flash-cot:mathvista=responses/gemini-2.5-flash_mathvista_cot.json \
  --workers 4

python main.py --dataset mathvista --model Qwen3.5-35B-A3B-cot \
  --response-files Qwen3.5-35B-A3B-cot:mathvista=responses/Qwen3.5-35B-A3B_mathvista_cot.json \
  --workers 4

python main.py --dataset mathvista --model Qwen3-VL-235B-A22B-Instruct-cot \
  --response-files Qwen3-VL-235B-A22B-Instruct-cot:mathvista=responses/Qwen3-VL-235B-A22B-Instruct_mathvista_cot.json \
  --workers 5
```

#### VQA-RAD

```bash
python main.py --dataset vqarad --model Qwen3.5-35B-A3B \
  --response-files Qwen3.5-35B-A3B:vqarad=responses/Qwen3.5-35B-A3B_vqarad.json \
  --workers 10

python main.py --dataset vqarad --model Qwen3-VL-235B-A22B-Instruct \
  --response-files Qwen3-VL-235B-A22B-Instruct:vqarad=responses/Qwen3-VL-235B-A22B-Instruct_vqarad.json \
  --workers 10

python main.py --dataset vqarad --model gpt-5.4-mini \
  --response-files gpt-5.4-mini:vqarad=responses/gpt5.4-mini_vqarad.json \
  --workers 10

python main.py --dataset vqarad --model gemini-2.5-flash \
  --response-files gemini-2.5-flash:vqarad=responses/gemini-2.5-flash_vqarad.json \
  --workers 10
```

生成回答和 GPT Judge 检测都通过 `utils/api.py` 的 `create_model_client()` 统一解析模型配置；命令行只需指定模型名，脚本自动从 `configs/config.py` 获取 API key、base URL 和调用协议。评测只使用回答文件中已有的样本；GPT Judge 默认断点续跑，每 10 条自动保存一次；并发度通过 `--workers` 控制。

## 数据集与检测方法

| 数据集 | 检测方法 | 原因 |
|--------|----------|------|
| **POPE** | 规则判断法 | 二元 Yes/No 对象存在性问题，参考 POPE 官方评估代码归一化回答并计算二分类指标 |
| **MathVista** | GPT Judge | 数学视觉推理题型多样，使用 GPT-5.5 作为自动化裁判综合判断 |
| **VQA-RAD** | GPT Judge | 医学放射科 VQA，使用 GPT-5.5 作为自动化裁判，额外按 closed/open 答案类型分层统计幻觉率 |

### POPE (Polling-based Object Probing Evaluation)

- **用途**：评估对象幻觉 —— 检测 MLLM 是否捏造不存在的对象
- **题型**：二元 Yes/No 问题（"图中有椅子吗？"）
- **采样策略**：random / popular / adversarial 三种
- **检测方式**：使用规则判断法将回答归一化为 `yes`/`no`，再与 ground truth 比对；该规则参考 POPE 官方 `evaluate.py`；`label=no` 但模型回答 `yes` 记为对象幻觉

### MathVista

- **用途**：评估数学视觉推理中的幻觉 —— 图表理解、几何推理、公式解读等
- **题型**：自由回答 (free_form) / 选择题 (multiple_choice)
- **数据集**：默认使用 testmini split（1000 条样本），也可选择完整的 test split（6141 条样本）
- **检测方式**：以图片 + 问题 + 模型回答 + ground truth 为输入，输出幻觉判定与类型分类（faithfulness / factuality / logical）

### VQA-RAD

- **用途**：评估医学放射科 VQA 中的幻觉 —— 检测 MLLM 在 X-ray、CT、MRI 图像问答中的视觉误判与知识错误
- **题型**：closed（yes/no）和 open-ended 两类，共 451 条 QA 对（使用 Hugging Face test split），覆盖放射科图像
- **检测方式**：以图片 + 问题 + 模型回答 + ground truth 为输入，使用 GPT Judge 判断幻觉；额外按 closed/open 分层统计幻觉率

## 检测方法详情

### 规则判断法 (evaluation/rule_based.py) — POPE 专用

POPE 部分使用规则判断法。回答归一化规则参考 RUCAIBox/POPE 官方 `evaluate.py`，随后进行二分类评估：

| 步骤 | 说明 |
|------|------|
| Yes/No 归一化 | 参考 POPE 官方 `evaluate.py`：只保留第一个句号前的第一句，移除逗号；如果词中包含 `No` / `not` / `no`，判为 `no`，否则判为 `yes` |
| 二分类评估 | `yes` 为正类，计算 TP / FP / TN / FN、Accuracy、Precision、Recall、F1、Yes Ratio |
| 对象幻觉判定 | `label=no` 但模型回答 `yes`，即 FP，记为对象幻觉 |

### GPT Judge (evaluation/judge.py) — MathVista 和 VQA-RAD 专用

使用 GPT-5.5 作为自动化裁判，以图片 + 问题 + 模型回答 + ground truth 作为输入，
判断回答是否包含幻觉。所有 VQA-RAD 结果和 MathVista CoT 结果均会进一步分类至忠实性/事实性/逻辑性三种类型。

**direct 输出格式**：
```json
{
    "score": 1,
    "has_hallucination": true,
    "reason": "The model describes a cat, but the image shows a dog."
}
```

**cot 输出格式**：
```json
{
    "score": 1,
    "has_hallucination": true,
    "hallucination_type": "faithfulness",
    "reason": "The model describes a cat, but the image shows a dog."
}
```

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Accuracy** | (TP + TN) / Total | 整体正确率 |
| **Precision** | TP / (TP + FP) | 检测为幻觉中真正是幻觉的比例 |
| **Recall** | TP / (TP + FN) | 真实幻觉中被成功检出的比例 |
| **F1 Score** | 2PR / (P + R) | Precision 与 Recall 的调和平均 |
| **Yes Ratio** | N_{pred=yes} / N_total | POPE 官方报告指标，用于观察模型是否倾向回答 yes |
| **Object Hallucination Rate** | FP / N_total | POPE 中 `label=no` 但模型回答 `yes` 的比例 |
| **Hallucination Rate** | N_{score<3} / N_total | MathVista / VQA-RAD GPT Judge 指标。参考 MMHal-Bench (Sun et al., ACL Findings 2024, Table 6) 的 0-6 评分协议，使用 GPT-5.5 评分；score < 3 为含幻觉，score >= 3 为无幻觉，值越低越好 |

> 注：幻觉检测视为二分类问题 —— Positive = 存在幻觉。

## 输出结果

检测结果保存于 `results/` 目录，JSON 格式：

```
results/
├── gpt-5.4-mini_pope_random.json  # POPE 规则检测结果
├── gpt-5.4-mini_mathvista.json    # MathVista GPT Judge 结果
└── gpt-5.4-mini_vqarad.json       # VQA-RAD GPT Judge 结果（含 closed/open 分层）
```

## 扩展指南

### 添加新数据集

1. 在 `data/` 下新建 loader
2. 在 `evaluation/detectors/` 中添加对应的检测器或 prompt 模板
3. 在 `evaluation/runners/` 中添加运行逻辑
4. 在 `main.py` 中注册分发

### 添加新模型

在 `configs/config.py` 的 `MODEL_API_CONFIGS` 和 `MODELS` 中添加条目：

```python
MODEL_API_CONFIGS = {
    ...
    "new-model": {
        "api_key": "sk-...",
        "base_url": "https://api.example.com/v1",
        "api_method": "chat",       # "chat" / "responses" / "messages"
        "api_model": "model-id",    # 可选，默认取模型名
        "max_tokens": 4096,         # 可选
    },
}
MODELS = [
    "gpt-5.4-mini",
    "gemini-2.5-flash",
    "Qwen3.5-35B-A3B",
    "new-model",
]
```

### 添加新检测方法

1. 在 `evaluation/detectors/` 下新建检测器
2. 在 `evaluation/runners/` 中接入数据加载、检测和指标保存
3. 在 `main.py` 中添加分发入口
