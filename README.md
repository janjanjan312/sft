# AIMS5740 Homework1 — SFT Practice (Qwen2.5-0.5B)

本目录用于完成作业要求的 **data → training → evaluation** 闭环，并产出可提交的压缩包（<10MB，不含模型权重）。

## 你需要提交的内容（压缩包内）

- **一个完整训练脚本**（`.py` 或 `.ipynb`）
  - SFT 训练脚本 + 日志（含关键超参）
  - Base vs SFT 的评测分数对比
- **一份 2–4 页简短报告**（数据清洗策略、训练调参、评测结果、结论分析）

建议最终把要提交的文件都放到 `submission/`，然后用 `scripts/make_submission_zip.py` 生成 zip。

## 推荐执行流程

### 0) 环境准备（建议用带 NVIDIA GPU 的 Linux/Colab）

本机 macOS 通常无法进行 CUDA 训练/评测（`lm_eval` 也会很慢）。建议使用：
- **Google Colab**（T4/A100 等）
- 或学校/实验室 GPU 服务器

最低依赖（后续脚本会用到）：
- `torch`（CUDA 版）
- `transformers`
- `datasets`
- `accelerate`
- `lm_eval[hf]`
- `sentencepiece`、`tiktoken`
- **LLaMA-Factory**（用于 full finetune）

> 注意：不同环境的 CUDA/torch 组合不同，这里不强行写死安装命令。你可以先把 LLaMA-Factory 按官方说明安装好，确保命令 `llamafactory-cli` 可用。

### 1) 数据准备与清洗（OpenOrca 1M 子集 → 200k~350k）

脚本：`scripts/prepare_openorca_sft.py`

它会：
- 按作业指定 revision (`e9c87b4`) 读取 `train[:1_000_000]`
- 做基础清洗（空样本/异常长度/简单重复/去重）
- 采样到目标规模（默认 250k，可改）
- 导出为 LLaMA-Factory 支持的 **Alpaca 格式**（jsonl）
- 输出清洗统计到 `artifacts/training_logs/data_cleaning_stats.json`

示例：

```bash
python scripts/prepare_openorca_sft.py \
  --out artifacts/data/openorca_sft_clean.jsonl \
  --stats artifacts/training_logs/data_cleaning_stats.json \
  --max_samples 250000 \
  --seed 42
```

### 2) SFT 训练（LLaMA-Factory / full finetuning）

配置：`configs/qwen2_5_0_5b_sft_full.yaml`

训练（示例）：

```bash
llamafactory-cli train configs/qwen2_5_0_5b_sft_full.yaml
```

训练输出默认到：`artifacts/saves/qwen2.5-0.5b/full/sft/`

你需要保留：
- `artifacts/training_logs/train.log`（关键超参 + loss 曲线信息）
- （可选）`plot_loss.png`（若开启 `plot_loss`）

### 3) 评测（lm_eval，Base vs SFT）

脚本：`eval/run_lm_eval.py`

会按作业要求跑 8 个任务，并固定 few-shot：
- mmlu 5-shot
- arc_easy 0-shot
- arc_challenge 25-shot
- hellaswag 10-shot
- winogrande 5-shot
- truthfulqa_mc2 0-shot
- piqa 0-shot
- boolq 0-shot

示例：

```bash
# Base
python eval/run_lm_eval.py --model_pretrained "Qwen/Qwen2.5-0.5B" --out_dir artifacts/eval/base

# SFT（改成你训练输出目录）
python eval/run_lm_eval.py --model_pretrained "artifacts/saves/qwen2.5-0.5b/full/sft" --out_dir artifacts/eval/sft

# 汇总
python eval/summarize_lm_eval.py --base_dir artifacts/eval/base --sft_dir artifacts/eval/sft --out_csv artifacts/eval/summary.csv
```

### 4) 写报告（2–4 页）

模板：`report/report.md`  
把数据清洗、训练超参、评测对比表、结论分析填进去即可（建议导出 PDF）。

### 5) 生成最终提交 zip（<10MB）

脚本：`scripts/make_submission_zip.py`

你需要提供：
- 你的 **StudentID** 和 **Name**（用于文件命名）
- 训练脚本/日志/评测结果/报告的位置（默认从 `submission/` 取）

```bash
python scripts/make_submission_zip.py --student_id 12345678 --name ZhangSan
```

会生成：`submission/12345678_ZhangSan_AIMS5740_Homework1.zip`

## 你现在需要给我的信息（两项就够）

1) **你打算在哪跑训练与评测**：Colab / GPU服务器 / 本机（如果是 GPU 服务器，CUDA 版本大概是多少也行）  
2) **StudentID 与 Name**（用于自动生成提交 zip 的文件名）

