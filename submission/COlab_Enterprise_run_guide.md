# Colab Enterprise 跑作业指南（按你邮件里的 GPU 规则）

## 结论：我能不能“直接 access 这些 GPU”？

**不行。** 我在你本地的 Cursor 里只能帮你把代码/配置都准备好，但 **无法替你登录 Google Cloud / Colab Enterprise，也无法直接占用或启动你的 T4/L4 runtime**（这些都需要你的账号权限与浏览器交互）。  
你需要在 Colab 里把项目跑起来；我负责把步骤写到“复制粘贴就能跑”。

## 1) 先在 Colab Enterprise 创建 GPU Runtime（邮件要求）

按邮件步骤做（你已经有指引图）：
- 登录 Google Cloud Console（用学校给你的账号）
- 选择正确的 Project（邮件里写的那个）
- 搜索并进入 **Colab Enterprise**
- 左侧 **My notebooks**
- **Important**：Region 选 **`us-east4 (Northern Virginia)`**
- Connect → Create new runtime
- Runtime template：
  - 用 **T4**：选择 `...-T4-20M`
  - 用 **L4**：选择 `...-L4-20M`
- 在 notebook 里运行：

```bash
!nvidia-smi
```

看到 **Tesla T4** 或 **NVIDIA L4** 就对了。

## 2) 把本地 `genai_hw1/` 放到 Colab（两种方式，二选一）

### 方式 A：上传到 Google Drive（最简单）
- 把整个文件夹 `genai_hw1/` 拖到 Drive（或先 zip 上传再解压）
- 在 Colab notebook 里挂载 Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

然后进入目录（按你的 Drive 实际路径改）：

```bash
%cd /content/drive/MyDrive/genai_hw1
!ls -la
```

### 方式 B：GitHub（如果你愿意把作业放 repo）
在 Colab：

```bash
!git clone <你的repo地址>
%cd <repo>/genai_hw1
```

## 3) 安装依赖（Colab 里执行）

```bash
!pip -q install -U pip
!pip -q install -r requirements.txt
!pip -q install -U deepspeed
```

安装 LLaMA-Factory（推荐用官方 repo 的方式，保证 `llamafactory-cli` 可用）：

```bash
!git clone https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
!pip -q install -e .
%cd /content/drive/MyDrive/genai_hw1  # 改回你的作业目录

!llamafactory-cli -h | head -n 5
```

> 如果你用的是 L4/T4，**bf16** 不一定都可用；若训练时报 dtype/amp 错，把 `configs/qwen2_5_0_5b_sft_full.yaml` 里的 `bf16: true` 改成 `false` 并把 `fp16: true`。

## 4) 一键跑通（建议按顺序）

### 4.0 长时间任务：计时 + 日志 + 后台进度（推荐用法）

Colab 同一个 runtime 通常一次只能跑一个 cell。为了避免“看起来卡住”、以及方便中途查看进度，建议：

- **前台跑（推荐）**：带计时，并把输出同步保存到日志文件

```bash
# 训练（示例）
!mkdir -p artifacts/training_logs
!time python scripts/run_llamafactory_train.py --config configs/qwen2_5_0_5b_sft_full.yaml 2>&1 | tee artifacts/training_logs/train_console.log

# base 评测（示例）
!mkdir -p artifacts/eval/base
!time python eval/run_lm_eval.py --model_pretrained "Qwen/Qwen2.5-0.5B" --out_dir artifacts/eval/base --device cuda:0 --batch_size 4 2>&1 | tee artifacts/eval/base/eval_console.log

# sft 评测（示例）
!mkdir -p artifacts/eval/sft
!time python eval/run_lm_eval.py --model_pretrained "artifacts/saves/qwen2.5-0.5b/full/sft" --out_dir artifacts/eval/sft --device cuda:0 --batch_size 4 2>&1 | tee artifacts/eval/sft/eval_console.log
```

- **后台跑（可选）**：不占住 cell，你可以用另一个 cell 查看日志/文件大小

```bash
# 后台启动（示例：base eval）
!nohup python eval/run_lm_eval.py --model_pretrained "Qwen/Qwen2.5-0.5B" --out_dir artifacts/eval/base --device cuda:0 --batch_size 4 > artifacts/eval/base/eval_console.log 2>&1 &

# 查看进度
!tail -n 50 artifacts/eval/base/eval_console.log
!ls -lh artifacts/eval/base | head
```

> 如果评测 OOM：把 `--batch_size 4` 改成 `1` 或 `2`。

### 4.1 数据清洗导出（1M → 250k，默认）

```bash
!python scripts/prepare_openorca_sft.py --max_samples 250000 --seed 42
!ls -la artifacts/llamafactory_data | head
```

这一步会生成：
- `artifacts/llamafactory_data/openorca_sft_clean.jsonl`
- `artifacts/llamafactory_data/dataset_info.json`
- `artifacts/training_logs/data_cleaning_stats.json`

### 4.2 训练（full finetune）

```bash
!mkdir -p artifacts/training_logs
!time python scripts/run_llamafactory_train.py --config configs/qwen2_5_0_5b_sft_full.yaml 2>&1 | tee artifacts/training_logs/train_console.log
!ls -la artifacts/training_logs | head
```

### 4.3 评测（Base vs SFT）

```bash
# base
!mkdir -p artifacts/eval/base
!time python eval/run_lm_eval.py --model_pretrained "Qwen/Qwen2.5-0.5B" --out_dir artifacts/eval/base --device cuda:0 --batch_size 4 2>&1 | tee artifacts/eval/base/eval_console.log

# sft（默认输出目录）
!mkdir -p artifacts/eval/sft
!time python eval/run_lm_eval.py --model_pretrained "artifacts/saves/qwen2.5-0.5b/full/sft" --out_dir artifacts/eval/sft --device cuda:0 --batch_size 4 2>&1 | tee artifacts/eval/sft/eval_console.log

# 汇总
!python eval/summarize_lm_eval.py --base_dir artifacts/eval/base --sft_dir artifacts/eval/sft --out_csv artifacts/eval/summary.csv
!sed -n '1,20p' artifacts/eval/summary.csv
```

> 评测如果 OOM：先把 `--batch_size` 从 4 降到 1 或 2。

## 5) 生成最终提交 zip（<10MB，不含权重）

你的命名是：**`1155245855_ShiliangChen_AIMS5740_Homework1.zip`**  
执行：

```bash
!python scripts/make_submission_zip.py --student_id 1155245855 --name ShiliangChen
!ls -la submission | grep AIMS5740
```

## 你还需要做的一件事：导出报告 PDF（2–4 页）

编辑 `report/report.md`，把：
- 清洗统计（`data_cleaning_stats.json` 摘要）
- 训练超参（yaml）
- 评测汇总表（`artifacts/eval/summary.md` 或 `summary.csv`）

补齐后导出成 `report/report.pdf`（可在本地转 PDF 再上传回 Drive），然后再重新跑一次打包脚本即可。

