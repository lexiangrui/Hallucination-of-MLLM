 直接结论：CoT 在多模态模型中会加剧幻觉

    Core statistics:
  - Direct 正确 → CoT 幻觉: 116 例
  - Direct 幻觉 → CoT 正确: 39 例
  - CoT 净增幻觉: +77 例 (16.3% → 24.0%)

  三个根本原因：

  1. CoT 暴露推理过程 → GPT Judge 更容易检测到错误

  Direct 模式下模型只需输出 "B"，GPT Judge 给 4-5 分；CoT
  模式下模型展开详细推理，如果推理中任何一步出错，GPT Judge
  一眼就能看到。即便最终答案正确，推理中的错误描述也会被扣到 score 2。典型例子：
  - PID 34：两个模型都回答 B（正确），但 CoT 推理中说 "overstating the smoothness" → 被 judge 判为
  score 2
  
  2. 承诺错误 (Commitment Error)

  CoT 让模型在推理早期锁定一个假设，然后围绕错误前提展开自信但错误的推理。类型迁移数据证实了这一点：
  - 81 例直接从 none→faithfulness：CoT 强迫模型描述图像细节，描述错了就被判幻觉
  - 23 例从 faithfulness→logical：视觉读取错误 → 后续推理全部基于错误前提 → 逻辑错误叠加

  3. 过度思考导致幻觉

  最典型的 PID 29：图片问两个人物的年龄差，GT=4。Direct 模式正确使用历史知识计算出生年份差。CoT
  模式却花大量篇幅论证"图片里看不到出生日期，所以无法判断"，最后胡诌答案 0。CoT
  让模型变得过于谨慎，反而开始虚构不存在的"推理困难"。

  分数分布也证实了极化效应：
          Direct    CoT
  Score 6:   452 →  723  (更多详细正确的回答)
  Score 5:   291 →   30  (大部分被推到了6或2)
  Score 2:   142 →  194  (详细但错误的推理)
  Score 0:     0 →    8  (完全错误的CoT)
  CoT 把模型推向两极——要么更好（6 暴增），要么更差（大量落入
  0-2），平均分微升（4.847→4.941）但不掩盖幻觉率大幅上升。
  
  1. "More Thinking, Less Seeing?" — 最贴合的论文

  Liu et al. (2025) "More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal 
  Reasoning Models", arXiv:2505.21523

  ▎ 论文直接研究了"推理链越长 → 幻觉越多"的现象。注意力分析显示：longer reasoning chains lead to 
  ▎ reduced focus on visual inputs——模型在生成 CoT 
  ▎ 时注意力从图像内容漂移，越来越依赖语言先验。作者提出 RH-AUC 指标来量化"推理长度 vs 
  ▎ 视觉感知精度"之间的权衡，发现更多思考反而意味着更少看见。

  这与你观察到的 Qwen-VL CoT 分数极化现象完全一致：CoT 让模型更依赖文本推理而非视觉证据。

  2. "Chain-of-Thought Degrades Visual Spatial Reasoning"

  Kancheti et al. (2025) "Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of 
  Multimodal LLMs", arXiv:2504.16060

  ▎ 评估 17 个模型 × 13 个空间推理基准，发现 CoT consistently degrades visual spatial reasoning。使用
  ▎  No-Image++ 消融方法证明模型从文本先验中"幻觉出视觉细节"——论文原文：hallucinate visual details 
  ▎ from textual priors even when the image is absent。结论是 text-only CoT 
  ▎ 不适合需要精确视觉定位的任务。

  3. "Thinking Before Looking"

  Zheng et al. (2024) "Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating 
  Visual Hallucination", arXiv:2411.12591

  ▎ 指出 CoT 在 MLLM 中面临 heightened risks of hallucination in cross-modality comprehension。提出 
  ▎ Visual Inference Chain 
  ▎ (VIC)——先做文本推理再看图——通过延迟视觉输入来降低跨模态幻觉风险。这间接印证了"CoT + 
  ▎ 同时看图"的组合会引入额外幻觉。

  ---
  对应机制：Snowball Hallucination (雪球式幻觉)
  
  4. ACL 2024 — Multimodal Hallucination Snowballing

  Zhong et al. (2024) "Investigating and Mitigating the Multimodal Hallucination Snowballing in Large
   Vision-Language Models", ACL 2024, arXiv:2407.00569

  ▎ 提出 MMHalSnowball 框架，发现 LVLM 在生成过程中一旦产生一个幻觉，就会在后续推理中接受该幻觉并基于
  ▎ 它做出更多错误断言（snowballing）。开放式 LVLM 在遇到自己的幻觉后性能下降 ≥31%。这对应你分析中的 
  ▎ Commitment Error：模型在 CoT 早期锁定错误前提，后续围绕它展开看似合理但错误的推理。

  5. ICML 2024 — 语言模型的雪球效应

  Zhang et al. (2024) "How Language Model Hallucinations Can Snowball", ICML 2024

  ▎ 虽然针对文本 LLM，但机制完全通用：模型倾向于承诺早期错误答案，然后生成虚假理由来"证明"它（即使模
  ▎ 型在孤立提示下能识别这些理由是错的——GPT-4 检测率 87%）。零样本 CoT 
  ▎ 对简单任务有帮助，但对复杂多步推理任务反而失败。

**分析**：

Qwen3-VL-235B-A22B-Instruct 表现出了一个看似矛盾的结论：**开启 CoT 后评分均值上升（4.85 → 4.94），但幻觉率也上升了（16.3% → 24.0%）。**

这一现象与近期多篇研究的结果一致。Liu 等（2025）在 "More Thinking, Less Seeing?" 中通过注意力分析发现，**推理链越长，模型对视觉输入的注意力越弱**（attention drift），模型逐渐从图像内容漂移到语言先验，导致"更多思考反而更少看见"\cite{liuMoreThinkingLess2025}。Kancheti 等（2025）在 "Chain-of-Thought Degrades Visual Spatial Reasoning" 中评估了 17 个模型，发现 CoT **一致性地损害了视觉空间推理**，模型会"从文本先验中幻觉出视觉细节"\cite{kanchetiChainThoughtDegrades2025}。这与我们观察到的 faithfulness 幻觉从 113 增至 154 相吻合。

Zhong 等（2024）在 ACL 2024 上提出的 **Multimodal Hallucination Snowballing** 框架直接对应我们观察到的 Commitment Error 机制\cite{zhongInvestigatingMitigatingMultimodal2024}：模型在推理早期产生一个幻觉后，会围绕该错误前提展开后续推理，使错误在推理链中传播放大。我们的数据中 31 例 "none→logical"、81 例 "none→faithfulness" 的跃迁正是这种雪球效应的体现。

这一现象的根源在于 CoT 导致的分数分布极化效应。

- **分数极化**：Direct 模式下分数在 4–6 之间相对均匀分布（6 分 452 例，5 分 291 例，4 分 93 例）。CoT 模式下，大量样本被推向了两个极端——6 分从 452 暴增至 723（模型给出详尽正确推理），但 2 分也从 142 增至 194、0 分从 0 增至 8（详尽但错误的推理被 Judge 严格惩罚）。中间的 4–5 分几乎消失（93→6, 291→30），说明 CoT 将"有些信息但不详细"的回答要么提升为"详细且正确"（6 分），要么降级为"详细但错误"（2 分）。

- **Commitment Error（承诺错误）**：CoT 迫使模型在推理早期做出明确的视觉断言和中间结论。一旦某个中间步骤错误（例如误读图表数值、错误分类物体），模型会围绕错误前提继续展开看似合理的推理，导致最终状态是"信息丰富但错误"（score 2）。Zheng 等（2024）在 "Thinking Before Looking" 中也指出，CoT 在多模态模型中面临 **heightened risks of hallucination in cross-modality comprehension**，并提出 Visual Inference Chain（先文本推理再引入视觉输入）来缓解此问题\cite{zhengThinkingBeforeLooking2024}。

- **样本级转换分析**：逐样本比较 Direct 和 CoT 的判定结果，116 例在 Direct 下正确（score≥3）但在 CoT 下被判幻觉（score<3），而仅有 39 例出现反向改善。净增 77 例幻觉，与 HR 从 16.3% 升至 24.0% 一致。

- **CoT 暴露 vs 诱发**：CoT 的 faithfulness 幻觉从 113 增至 154（+41），logical 幻觉从 44 增至 80（+36）。Faithfulness 的增加部分来自"暴露效应"——Direct 模式下模型隐藏的视觉感知错误在 CoT 推理中被显式描述出来，被 Judge 捕获；Logical 的增加则主要是"诱发效应"——CoT 的逐步推理本应减少 logical 错误（对小模型效果显著），但 Qwen-VL 的 Direct 模式本身 logical 错误已很少（44 例），CoT 反而在某些情况下引入了不必要的推理步骤，创造了新的逻辑错误可能。

- **与小模型的对比**：GPT-5.4-mini 和 Qwen3.5-35B-A3B 的 CoT 均显著降低 HR（-15.7pp 和 -8.4pp），而 Qwen-VL 的 CoT 反而升高 HR（+7.7pp）。这一差异的关键在于**基线水平不同**：小模型 Direct 的 logical 幻觉分别占 68.7% 和 63.1%，CoT 大幅修正了这些明显推理错误；而 Qwen-VL Direct 的 logical 仅占 27.0%，CoT 的"修正空间"很小，其引入的副作用（暴露视觉错误、承诺错误）反而超过了其收益。这也符合 Liu 等（2025）的发现：更大模型在平衡推理长度与视觉感知精度方面表现更好，但依然无法避免 attention drift 问题\cite{liuMoreThinkingLess2025}。