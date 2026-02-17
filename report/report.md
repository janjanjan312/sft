# AIMS5740 Homework1 — SFT Practice 报告（2–4页）

> 模型：`Qwen/Qwen2.5-0.5B`  
> 数据：`Open-Orca/OpenOrca`（固定 revision：`e9c87b4`，先取 `train[:1_000_000]`）

## 1. 数据清洗策略（Data Cleaning）

### 1.1 数据来源与子集构造
- **数据集**：Open-Orca/OpenOrca（无官方 split）
- **可复现子集**：`train[:1_000_000]`，`revision="e9c87b4"`
- **最终训练样本量**：N = ______（建议 200k–350k）

### 1.2 清洗目标与规则
你需要说明你如何处理以下问题，并给出阈值/理由：
- **Formatting**：缺字段、空字段、异常 role（本作业数据为表格字段，重点是缺失/空文本）
- **Quality**：空/乱码、异常短/异常长、明显模板/重复
- **Consistency**：去重（exact/近似），避免重复语义对训练造成偏置

（示例写法，可改）
- 空文本过滤：`question/response` 为空或全空白直接丢弃  
- 长度过滤：`question` < __ 字符 或 `response` < __ 字符丢弃；超长异常也丢弃  
- 重复过滤：对长回答计算最大词频占比，超过阈值则丢弃  
- 去重：对标准化后的 `(question, response)` 做 hash 去重  

### 1.3 清洗统计
把脚本输出的统计表贴在这里（例如 `data_cleaning_stats.json` 的摘要）：
- 1M 原始样本：1,000,000  
- 丢弃（空/长度/重复/去重/其它）：_____  
- 最终保留：_____  

## 2. 训练设置与调参（Training & Tuning）

### 2.1 训练框架与方法
- **框架**：LLaMA-Factory
- **训练阶段**：SFT
- **方法**：Full finetuning（全参数微调）

### 2.2 关键超参数（务必列出来）
建议用表格：

| 超参 | 值 |
|---|---|
| `finetuning_type` | `full` |
| `cutoff_len` | ____ |
| `per_device_train_batch_size` | ____ |
| `gradient_accumulation_steps` | ____ |
| Effective batch size | ____ |
| `learning_rate` | ____ |
| `num_train_epochs` | ____ |
| `warmup_ratio/steps` | ____ |
| `weight_decay` | ____ |
| `lr_scheduler_type` | ____ |
| precision | bf16/fp16 |
| seed | ____ |

### 2.3 训练日志与收敛（Loss / Perplexity）
- 训练 loss 是否稳定下降？  
- 是否出现过拟合迹象（例如训练 loss 降但泛化无提升）？  
- 结合 `plot_loss` 或日志截图简单讨论  

## 3. 评测结果（Evaluation）

### 3.1 评测设置
使用 `lm_eval`，并确保：
- **apply_chat_template=True**
- 任务与 few-shot 固定如下：
  - mmlu 5-shot
  - arc_easy 0-shot
  - arc_challenge 25-shot
  - hellaswag 10-shot
  - winogrande 5-shot
  - truthfulqa_mc2 0-shot
  - piqa 0-shot
  - boolq 0-shot

### 3.2 自动评测分数（Base vs SFT）
把 `summary.csv` 或 `summary.md` 的表贴在这里，并简要解释变化原因（即使分数下降也能解释）。

（粘贴表格位置）

### 3.3 质性分析（Manual / Behavior）
挑 3–5 个你关心的 prompt，展示 **Base** 与 **SFT** 的输出对比，并分析：
- 是否更遵循指令/更详细/更少胡说
- 是否更像训练分布（OpenOrca风格的推理链/解释）

## 4. 结论与反思（Conclusions）
- 最有效的清洗/调参是什么？为什么？  
- 你认为 SFT 对模型“学到数据分布”与“输出行为改善”分别有哪些证据？  
- 下一步若要提升分数，你会怎么做（更强清洗、更长训练、更合理超参、更大/更小数据等）？

