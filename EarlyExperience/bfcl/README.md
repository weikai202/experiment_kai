# Early Experience on BFCL — Qwen3-32B non-thinking

用于实验对照的独立实现，调用本机 Gorilla/BFCL v4 的模拟器与评分器。

| 组别 | 初始化 | 训练方式 |
|---|---|---|
| IL | Qwen3-32B | expert SFT |
| IWM → IL | Qwen3-32B | 先预测候选动作的真实执行结果，再从该 checkpoint 继续 expert SFT |
| SR + IL | Qwen3-32B | expert 和环境反馈驱动的 reflection 样本混合 SFT |

所有模型调用显式设置 `chat_template_kwargs.enable_thinking=false`。SR 的 `<reflection>…</reflection>` 是可见训练目标，不是 Qwen 内置 thinking 模式；执行器只解析其后的 Python 函数调用。

参考：[EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience)、[BFCL 复现说明](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/envs/bfcl_v4/README.md)、[原论文](https://arxiv.org/abs/2510.08558)。实现的范围是 **BFCL v4 multi-turn**，不包含完整排行榜的 single-turn、web search、memory 等任务。

## 当前产物与验证边界

- 已下载公开 Opus 专家轨迹和官方评分，校验 SHA256。
- 已从成功的 162 个 Base case 划分 121 train / 41 heldout，seed=42。
- 已生成 1,196 条 expert 训练样本及 398 条 heldout 样本。三个 OOD 类别不进入训练数据。
- 162 个成功 case 的全部工具观察已在当前模拟器回放，观察不一致数为 0。
- 测试中的五状态数据生成使用测试替身生成文本、真实模拟器执行动作；**这些不是 Qwen3 生成的实验数据**。
- 尚未调用真实 Qwen3 服务，未训练 32B 模型，也未产生模型准确率。训练需要独立 GPU 环境；本机现有 BFCL 环境的 PyTorch 为 CPU 版本。

## 安装

以下路径是原开发机器的运行示例；其他机器请替换项目路径、`BFCL_PY` 和 Gorilla 路径。安装 BFCL 后可直接运行数据准备和测试：

```bash
cd /path/to/experiment_kai/EarlyExperience/bfcl
BFCL_PY=/home/weik/CoMAP/.venv-bfcl/bin/python
"$BFCL_PY" -m ee_bfcl --help
"$BFCL_PY" -m pytest -q
```

另建训练环境，避免改变 CoMAP 的依赖。建议 Python 3.10；根据集群安装匹配 CUDA 的 PyTorch：

```bash
python3.10 -m venv .venv-train
.venv-train/bin/pip install -e /home/weik/gorilla/berkeley-function-call-leaderboard
.venv-train/bin/pip install -e '.[train,test]'
```

本实现验证的 BFCL commit 为 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`。新机器应 checkout 这个 commit 后安装。参考仓库要求的 2025-12 版本与这里不同，因此本项目属于该方法的 Qwen3 适配，不能直接声称复现其 Qwen2.5 表格数值。运行时核验准备阶段保存的数据、模拟器及评分器哈希。

## 1. 准备专家数据

原开发机器的 `data/prepared` 已准备好；数据、缓存和模型权重不随 Git 上传。新 clone 请运行下面命令下载公开专家输入并准备数据。重新准备时使用新的输出目录：

```bash
"$BFCL_PY" -m ee_bfcl.download --output data/source
"$BFCL_PY" -m ee_bfcl prepare \
  --results data/source/opus_base_result.jsonl \
  --scores data/source/opus_base_score.jsonl \
  --output data/prepared --seed 42
```

`manifest.json` 保存 case 划分、输入哈希、BFCL commit 和源文件哈希。官方 score 文件只有汇总和失败 case；解析器检查数量完整性后计算成功集合，不会把缺失的评分文件默认当成全成功。

`expert_records.jsonl` 保留 case/turn/action/observation，`expert_sft_text.jsonl` 是标准 `{"messages": [...]}`。heldout 文件单独保存，训练命令不使用它。每个用户回合增加 `[]` 停止样本，并将专家的自然语言结束回复统一为 `[]`；这项格式适配对三组完全一致。原始专家并行调用仍作为一个 action。

## 2. 配置 Qwen3 服务并生成 IWM/SR 数据

编辑 `configs/qwen3_32b.json` 的 `base_url`、`model`，必要时设置 `revision` 标识实际模型版本。鉴权使用 `EE_API_KEY`；配置不保存密钥。服务需支持 vLLM 的 `chat_template_kwargs` 参数。

vLLM 示例（GPU 数量和上下文长度应按机器调整；本项目不会自动启动服务）：

```bash
vllm serve /path/to/Qwen3-32B \
  --served-model-name Qwen/Qwen3-32B \
  --tensor-parallel-size 4 --max-model-len 32768 --port 8000
```

先 dry run，不调用模型：

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/smoke --config configs/qwen3_32b.json --limit 5
```

真实五状态 smoke（65 个逻辑请求，另加失败重试；实际 token 使用记录于 API 缓存）：

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/smoke --config configs/qwen3_32b.json --limit 5 --execute
```

检查 `outputs/smoke/states/*.json` 的候选动作、真实返回、摘要和反思，再生成全量：

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/full --config configs/qwen3_32b.json --filter-leaks --execute
```

每个训练 action state：从模拟器公开且有文档的方法中抽取 K=10 个不同候选名称，排除该专家 action 中的函数名称；一次模型调用为全部候选填写参数。每个候选在独立 `deepcopy` 上执行。工具错误也保留，绝不由模型虚构 next state。随后生成描述性摘要，选 K=3 个候选供 SR 比较。

IWM 用独立世界模型 system prompt，以 `Observation:` 开头输出观察摘要；IL/SR 使用动作格式，防止训练时把预测结果与选择动作混淆。SR 最终动作由原始 expert action 固定拼接。默认只审计反思中的标签泄漏；`--filter-leaks` 对应参考仓库的词汇筛选，丢弃数量记录在 `reflection_audit.jsonl` 和 `report.json`，不会筛除 IWM 的错误反馈。

状态结果与 API 响应均可缓存续跑。模型、数据划分、筛选选项或 limit 改变时须使用新的输出目录；更换同名服务 checkpoint 时务必更新配置中的 `revision`。API 超时或输出截断会报错并保留进度，不会把截断结果当作训练样本。

## 3. 训练三组 baseline

先生成带输入哈希和阶段依赖的训练计划；默认拒绝将部分 smoke 数据用作正式实验：

```bash
"$BFCL_PY" -m ee_bfcl plan-training --prepared data/prepared \
  --collected outputs/full --output checkpoints
```

下面展示训练的四次调用；在训练环境中执行。32B 全参数训练通常需要分片，使用集群的 Accelerate/FSDP/DeepSpeed 启动配置，将 `python` 替换为对应的 `accelerate launch -m` 调用方式。**默认每设备 batch=1、gradient accumulation=32，有效 batch 还要乘进程数**；比较实验需显式调整成一致的有效 batch。

```bash
# IL
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/il

# IWM stage 1
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data outputs/full/iwm_sft_text.jsonl --output checkpoints/iwm_stage1

# IWM stage 2: 必须继承 stage 1 权重
python -m ee_bfcl.train --model checkpoints/iwm_stage1 \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/iwm_il

# SR + IL: 从同一个基础模型开始
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl outputs/full/reflection_sft_text.jsonl \
  --output checkpoints/sr_il
```

训练默认每阶段 1 epoch、lr=1e-5、cosine、warmup=0.03，仅是可修改的起点，不代表原文超参数。仅对最后一条 assistant target 计算 loss，历史消息与 padding 均 mask。原生 non-thinking 前缀进入 prompt mask；SR 显式反思和动作均计算 loss。序列超长直接报错，不静默截断或丢弃样本。

先检查真实 token 长度，完全不加载模型权重：

```bash
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/il --validate-only
```

内存有限时可给从基础模型开始的三个命令加 `--lora --lora-rank 32`。IWM 第二阶段自动识别并继续训练第一阶段 adapter。为公平比较，三组应使用相同训练方式。LoRA 是额外实验选择，应单独注明；需要完整 checkpoint 时：

```bash
python -m ee_bfcl.merge_adapter --adapter checkpoints/iwm_il --output checkpoints/iwm_il_merged
```

合并 32B adapter 需要足够 CPU RAM。训练恢复使用 `--resume-from-checkpoint /path/to/checkpoint-N`。

## 4. BFCL 评测

将每组最终 checkpoint 分别部署到服务，复制一份 JSON 配置，设置该组的 `model`、`base_url` 与唯一 `revision`。训练与评测都采用 BFCL Python 文本调用模式，**不是 Qwen 原生 XML/JSON FC 模式**。

```bash
"$BFCL_PY" -m ee_bfcl evaluate --prepared data/prepared \
  --output outputs/eval_il/base --config configs/il_eval.json \
  --category multi_turn_base --execute

for category in multi_turn_long_context multi_turn_miss_func multi_turn_miss_param; do
  "$BFCL_PY" -m ee_bfcl evaluate --prepared data/prepared \
    --output "outputs/eval_il/$category" --config configs/il_eval.json \
    --category "$category" --execute
done
```

`configs/il_eval.json` 需由 `configs/qwen3_32b.json` 复制后修改；IWM/SR 使用各自配置和输出目录。去掉 `--execute` 可预览样本数量与请求上限。先加 `--limit 1` 可做模型服务 smoke，请使用单独输出目录。

Base 只评 41 个 heldout case；另外三个类别分别评完整 OOD。缺失函数在指定回合才加入提示。调用官方 `multi_turn_checker` 和 `multi_turn_irrelevance_checker`，以真实状态与返回值评判，不采用字符串 exact-match。为了避免执行生成的任意 Python，模型和参考动作都通过同一 AST literal-only executor 执行。

`results.jsonl` 保存逐 case 原始输出、工具反馈、解析结果与官方评分详情；`metrics.json` 保存该类别 accuracy。不同类别应分别报告；这不是完整 BFCL leaderboard overall score。长上下文 OOD 可能需要更大的服务上下文窗口；不足时服务报错，应调整配置后在新输出目录重跑，不能截断用户任务后仍称完整评测。

## 代码位置

- `prepare.py`：专家轨迹提取、按 case 切分、真实观察回放检查。
- `environment.py`：BFCL 模拟器副本、动作解析、官方评分器适配。
- `collect.py`：候选动作、观察摘要、IWM/SR 数据、标签泄漏审计。
- `client.py`：non-thinking 请求、缓存与 Qwen 工具观察模板转换。
- `train.py`：IL / 两阶段 IWM / SR 混合训练与最终 assistant loss mask。
- `evaluate.py`：多轮交互、延迟函数提示、官方执行评分、断点续跑。
- `tests/test_baseline.py`：真实模拟器集成、五状态测试替身 smoke、tokenizer 和隔离检查。

公平比较时，应让你们的方法与这三组共享 `manifest.json` 的 case 划分、工具提示、终止条件、推理预算与评分接口；不要训练 Base 后再报告整个 Base 的分数。


## Hugging Face → vLLM 启动入口

模型固定为 `Qwen/Qwen3-32B`，HF revision 为 `9216db5781bf21249d130ec9da846c4624c16137`。
在已分配且空闲的 GPU 节点，创建独立 serving 环境，按 GPU 驱动/CUDA 兼容性安装 vLLM
（Qwen 官方要求 vLLM >=0.8.5；不要覆盖 BFCL 或训练环境的 torch）。

```bash
# 在装有 vLLM 的 Python 环境、项目根目录执行
python -m ee_bfcl.serve --dry-run
python -m ee_bfcl.serve
```

入口使用 HF 模型 ID 自动下载权重，默认 BF16、TP=4、32768 上下文、端口 8000，
并从同一 HF revision 生成强制 non-thinking 模板。现有配置
`configs/qwen3_32b.json` 可直接访问 `http://localhost:8000/v1`。
请求端也继续传 `enable_thinking=false`。不启用原生 FC parser，保持既有 BFCL 文本调用协议。
可用 `--tensor-parallel-size` 调整分卡；显存不足会在下载模型前报错。
在其他节点运行时，通过 SSH 端口转发到 localhost:8000，或明确设置 host 和客户端 base_url。

服务就绪后运行真实五状态 smoke：

```bash
/home/weik/CoMAP/.venv-bfcl/bin/python -m ee_bfcl collect \
  --prepared data/prepared --output outputs/qwen3_hf_smoke \
  --config configs/qwen3_32b.json --limit 5 --execute
```

启动方式依据 [Qwen 官方 vLLM 部署文档](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)。
本次检查四张 L40S 均被已有任务占用，未启动服务或下载 32B 权重。

验证产物中提及的本机路径、已生成数据和测试报告仅记录开发验证状态，不作为仓库随附文件。尚未下载公开数据时，相关集成测试会显式 skip。
