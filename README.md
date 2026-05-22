# MLLM Hallucination Detection

多模态大模型幻觉自动检测与评估。支持三个数据集（POPE、MathVista、VQA-RAD），两种检测方法（规则判断法、MLLM Judge）。

## 目录结构

```
.
├── main.py                       # 主入口
├── configs/
│   └── config.py                 # 全局配置（API Keys、路径、模型列表）
├── data/
│   ├── pope_loader.py            # POPE 数据集加载（HF Parquet → 样本列表）
│   ├── mathvista_loader.py       # MathVista 数据集加载
│   └── vqarad_loader.py          # VQA-RAD 数据集加载
├── evaluation/
│   ├── judge.py                  # MLLM Judge（MMHal-Bench 0–6 分协议）
│   ├── metrics.py                # 指标计算：F1、HR、Kappa、McNemar、置信区间
│   ├── rule_based.py             # POPE 规则判断法
│   ├── run_pope.py               # POPE 评测运行器
│   ├── run_mathvista.py          # MathVista 评测运行器
│   └── run_vqarad.py             # VQA-RAD 评测运行器
├── scripts/
│   ├── generate_responses.py     # 调用测试模型 API 生成回答
│   ├── error_analysis.py         # 阈值分析 + 长度偏差 + 跨模型一致性
│   ├── judge_consistency.py      # Judge 模型一致性（GPT-5.5 vs Claude）
│   ├── cot_significance.py       # Direct vs CoT 配对显著性检验
│   ├── export_human_eval.py      # 导出分层人工标注样本
│   ├── annotate.py               # 人类标注 Web 服务器
│   └── analyze_human_alignment.py# 人机对齐分析
├── utils/
│   ├── api.py                    # API 调用（OpenAI Chat/Responses、Anthropic Messages）
│   └── batch.py                  # 断点续跑、JSON/图片工具
├── responses/                    # 模型回答 JSON
├── results/                      # 检测结果与误差分析
├── report/                       # LaTeX 实验报告
├── doc/                          # 中文实验报告
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install openai anthropic Pillow pyarrow pandas python-dotenv scipy scikit-learn statsmodels
```

### 配置 API Key

API key、base URL、调用方式统一配置在 `configs/config.py` 的 `MODEL_API_CONFIGS` 中。`api_method` 支持三种协议：

| api_method | 协议 | 适用模型 |
|------------|------|----------|
| `chat` | OpenAI Chat Completions | Qwen3.5 / Qwen3-VL / Gemini 2.5 Flash |
| `responses` | OpenAI Responses | GPT-5.4-mini / GPT-5.5（Judge） |
| `messages` | Anthropic Messages | Claude Opus 4.7（Judge 一致性实验） |

### 流程

**Step 1：生成模型回答**

```bash
# POPE random split
python scripts/generate_responses.py --dataset pope --pope-split random --model gpt-5.4-mini --output responses/gpt-5.4-mini_pope_random.json --workers 10

# MathVista testmini
python scripts/generate_responses.py --dataset mathvista --model gpt-5.4-mini --output responses/gpt-5.4-mini_mathvista.json --workers 10

# MathVista CoT
python scripts/generate_responses.py --dataset mathvista --model gpt-5.4-mini --prompt-mode cot --output responses/gpt-5.4-mini_mathvista_cot.json --workers 10

# VQA-RAD
python scripts/generate_responses.py --dataset vqarad --model gpt-5.4-mini --output responses/gpt-5.4-mini_vqarad.json --workers 10
```

输出格式：`{样本ID: "回答文本", ...}`，支持断点续跑。

**Step 2：运行检测**

```bash
# POPE（规则判断法，无需 Judge API）
python main.py --dataset pope --model gpt-5.4-mini --pope-split random \
  --response-files gpt-5.4-mini:pope=responses/gpt-5.4-mini_pope_random.json

# MathVista / VQA-RAD（MLLM Judge）
python main.py --dataset mathvista --model gpt-5.4-mini \
  --response-files gpt-5.4-mini:mathvista=responses/gpt-5.4-mini_mathvista.json --workers 10

# 全部数据集 + 全部模型
python main.py --dataset all \
  --response-files gpt-5.4-mini:pope=responses/gpt-5.4-mini_pope_random.json \
  gpt-5.4-mini:mathvista=responses/gpt-5.4-mini_mathvista.json \
  gpt-5.4-mini:vqarad=responses/gpt-5.4-mini_vqarad.json
```

### 误差分析

```bash
# 阈值分析 + 长度偏差 + 跨模型一致性（零 API 成本）
python scripts/error_analysis.py

# Direct vs CoT 显著性检验
python scripts/cot_significance.py

# Judge 一致性检验（需 Claude API）
python scripts/judge_consistency.py --workers 4

# 人类对齐实验
python scripts/export_human_eval.py       # 导出 40 条分层样本
python scripts/annotate.py                # 启动标注服务器 localhost:8765
python scripts/analyze_human_alignment.py # 生成对齐分析报告
```

## 数据集与检测方法

| 数据集 | 检测方法 | 说明 |
|--------|----------|------|
| **POPE** | 规则判断法 | 二元 Yes/No 物体存在性探测（3 splits × 1000 条）。归一化回答后计算二分类指标。 |
| **MathVista** | MLLM Judge（0–6 分） | 数学视觉推理（testmini，1000 条）。score < 3 判定为幻觉。 |
| **VQA-RAD** | MLLM Judge（0–6 分） | 医学放射科 VQA（test，451 条）。按 closed/open 答案类型分层统计。 |

### 幻觉分类体系

MLLM Judge 在 score < 3 时自动标注类型：
- **Faithfulness**（忠实性）：视觉不一致 — 描述图中不存在的物体/属性/关系
- **Factuality**（事实性）：与既定世界知识相矛盾
- **Logical**（逻辑性）：推理错误 — 结论不源于证据或自相矛盾

完整报告见 [doc/实验报告.md](doc/实验报告.md)（中文）或 [report/](report/)（LaTeX 英文）。

## 扩展指南

### 添加新模型

在 `configs/config.py` 中配置：
```python
MODEL_API_CONFIGS["new-model"] = {
    "api_key": "...",
    "base_url": "https://api.example.com/v1",
    "api_method": "chat",       # "chat" | "responses" | "messages"
    "api_model": "model-id",    # 可选，默认取 key 名
}
```

### 添加新数据集

1. 在 `data/` 下新建 loader
2. 在 `evaluation/` 中添加检测运行器（参考 `run_pope.py`）
3. 在 `main.py` 中注册分发
