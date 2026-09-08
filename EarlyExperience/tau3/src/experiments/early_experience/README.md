# Early Experience on τ³-bench

本模块实现 [EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience)
的 IL、IWM → IL、SR baseline，用于**训练后在未见任务上评测**。
适配的 τ³ 源码版本：`17e07b1da2bbc0cadfddeea36412686e0604127b`。

## 方法与范围

- **IL**：专家状态 → 专家动作。
- **IWM**：在专家状态执行 K 个不同的候选动作，用真实下一观察做预测训练；
  随后在**同一模型权重**上进行 IL。IWM 和 IL 使用不同的系统提示及目标格式。
- **SR**：基座模型读取专家动作和候选动作的真实结果，每个候选生成一条反思；
  将反思和专家动作作为监督目标，与专家样本混合训练。
- 支持 `retail`、`airline`、`telecom` 和离线测试用 `mock`，仅文本半双工。
  **尚不支持 `banking_knowledge`、语音/全双工和 solo/GT agent**。
  知识检索环境包含外部资源，不能假定 `deepcopy` 能隔离状态。
- 上游 EE 的 tau-bench 目录仍未发布实现。这里是根据其方法说明编写的 τ³ 适配，
  不是作者发布的 τ³ 实验结果，也不保证复现原论文数值。

实现决策：默认 K=3；一次请求生成多个候选并去重，属于批量候选提议设置，
不是 K 次独立策略采样。专家原始下一观察来自演示轨迹，工具结果额外重放验证；
候选对话动作重新调用同一配置的用户模拟器，因此用户响应存在随机性。
所有错误工具返回均保留；不按 reward 自动筛选演示。导入轨迹应由实验者预先确认质量。
如使用强模型生成候选/反思，应报告为 teacher-assisted EE；论文式 self 设置需使用
将被训练的同一基座模型。反思长度提示为 200–400 词，软上限 500 词。

## 安装

从仓库根目录运行。模块路径仍为 `tau2`，这是 τ³ 上游的包名。

```bash
uv sync --extra dev --extra voice --extra knowledge
# PyAudio 编译需要系统 PortAudio 开发包（例如 portaudio-devel / portaudio19-dev）。
# 训练依赖；PyTorch 的 CUDA 版本按训练机器安装：
uv pip install 'transformers==4.57.1' 'accelerate==1.11.0' torch
.venv/bin/python -m experiments.early_experience --help
```

此版本上游在文本入口中也加载部分 voice/knowledge 模块，因此仅 `uv sync`
不足以运行。使用 `.venv/bin/python` 可避免后续 `uv run` 重新同步移除额外训练依赖。
API key 通过环境变量配置，不要写进配置文件或 `--generator-args`。
本模块不会自动启动训练服务器；使用 LiteLLM 支持的模型名与 `api_base` 接入模型。

## 1. 固定任务划分

```bash
.venv/bin/python -m experiments.early_experience split \
  --domain retail --train-split train --eval-split test --output retail_split.json
```

也可手动提供 `{"domain":"retail","train_ids":["..."],"eval_ids":["..."]}`。
两组 ID 必须非空、无重复且不重叠。已经用 `train` 采样/训练时，不能再将含训练任务的
`base` 全集作为无泄漏测试集。跨方法固定 split、模型、随机种子、用户模拟器和 trials。

## 2. 采集或导入专家演示

准备 `teacher.json`（模型名称仅为示例，应替换为实验所用模型）：

```json
{
  "domain": "retail",
  "agent": "llm_agent",
  "llm_agent": "openai/teacher",
  "llm_args_agent": {"api_base": "http://localhost:8000/v1", "temperature": 0},
  "llm_user": "openai/user-model",
  "llm_args_user": {"api_base": "http://localhost:8001/v1", "temperature": 0},
  "num_trials": 1,
  "max_steps": 100,
  "seed": 42
}
```

```bash
.venv/bin/python -m experiments.early_experience collect \
  --config teacher.json --split retail_split.json --output teacher_results.json
```

该命令调用真实模型及原生评估器，会产生费用。已有原生 `Results` JSON 可直接进入下一步，
但必须只含训练任务，并使用普通 `llm_agent` / `ee_agent` 和 `user_simulator`。
不接受将 task 的 evaluation_criteria 当作可见输入的 GT agent。
专家轨迹中初始脚本历史和默认问候仅作为上下文；其后的每个 assistant 决策均需实际下一观察，
缺失结果或不能重放的轨迹会报错，不能被静默忽略。

## 3. 真实分支采样、反思与导出

先用五个状态冒烟，检查 `transitions.jsonl` 的真实动作、返回值和反思内容：

```bash
.venv/bin/python -m experiments.early_experience generate \
  --results teacher_results.json --split retail_split.json --output ee_retail \
  --generator-model openai/base-policy \
  --generator-args '{"api_base":"http://localhost:8002/v1","temperature":0.7}' \
  --k 3 --max-states 5
.venv/bin/python -m experiments.early_experience export --data ee_retail
```

删除 `--max-states` 再运行即可继续。每个完整状态写入后立即 flush；重启时跳过已完成状态。
配置与来源文件哈希不匹配会拒绝续跑。中断于状态内部时该状态需要重做；若文件存在半行，
JSON 读取会报错，请保留备份后修复未完成的尾行。

每个状态约一次候选生成、K 次反思生成，另有候选对话动作带来的用户模拟调用。
`--max-states` 限制的是本次新增状态数。完整运行前应基于冒烟 usage 估算成本。
不足 K 个不同候选、生成被截断、用户工具循环超限都会明确报错。

输出：`expert_sft.jsonl`、`iwm_sft.jsonl`、`reflection_sft.jsonl`，以及
用于审计的 `transitions.jsonl`、`manifest.json`。SFT 保留原生多轮 messages 和结构化
`tool_calls`；工具 schema 位于顶层 `tools`。IWM 只预测下一 agent 可见观察，不预测隐藏 DB。

## 4. 训练三组 baseline

每组均从**同一个原始基座 checkpoint**开始。IWM 命令内部执行两阶段：

```bash
for method in il iwm sr; do
  .venv/bin/python -m experiments.early_experience.train \
    --method "$method" --model /path/to/base-model --data ee_retail \
    --output "checkpoints/$method" --epochs 1 --iwm-epochs 1 \
    --learning-rate 2e-5 --batch-size 1 --gradient-accumulation 32 \
    --max-length 16384 --bf16 --seed 42
done
```

这是全参数 SFT。模型必须自带兼容工具调用的 Hugging Face chat template。
只计算每条样本最后一个 assistant completion 的 loss；历史、用户和工具返回均 mask。
过长样本直接报错，避免截断动作标签。IWM 第一阶段模型保存至 `iwm_warmup`，第二阶段
继续相同模型参数并重建 optimizer。最终可部署模型为各目录下的 `final`。
默认 epoch/LR 是可配置起点，不是论文 τ³ 推荐超参数。SR 默认每状态一条 IL 加 K 条 SR；
IWM 多一次 warm-up，两者额外训练量应在实验报告中列出。

## 5. 部署并使用原生 τ³ 评测

把 `checkpoints/<method>/final` 部署到支持相应模型工具解析的 OpenAI-compatible 服务。
在与 `teacher.json` 相同结构的 `eval.json` 中将 `llm_agent` 和 `api_base` 指向该服务，
固定 `llm_user`、用户参数和 seed，并设置 `num_trials`（例如 4）。

```bash
.venv/bin/python -m experiments.early_experience evaluate \
  --config eval.json --split retail_split.json \
  --manifest ee_retail/manifest.json --output il_results.json
```

对 IWM/SR 更换 endpoint 和输出文件，保留其他条件。命令强制使用测试 ID 并校验数据 manifest，
通过原生 runner/evaluator 输出原生 Results 和指标。τ-bench 的 `pass^k` 表示 k 次均成功的
一致性指标，不是至少成功一次的 `pass@k`。不在此模块重写奖励或成功判断。

三组均使用同一 `ee_agent` prompt。SR 的 `<think>...</think>` 在工具执行/发给用户前剥离，
后续 history 也不保留私有反思，与数据构建时的可见历史一致。输出标签未闭合则报错。
推理服务器必须保留模型生成的标签或按模型约定解析 reasoning，并正确解析工具调用。

## 离线验证

```bash
.venv/bin/python -m pytest src/experiments/early_experience/tests -q
.venv/bin/ruff check src/experiments/early_experience
.venv/bin/ruff format --check src/experiments/early_experience
```

测试使用真实 mock 环境工具，覆盖隔离、同批调用顺序、错误保留、私有信息不可见、
反思与动作格式、训练 mask、任务划分与三组 curriculum。测试不调用付费 API。

参考：[方法定义](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/skill/METHOD.md)、
[实现注意事项](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/skill/method_recap.md)、
[论文](https://arxiv.org/abs/2510.08558)。

### 本次验证记录

- Python 3.12.14；14 项数据/环境/原生评估器测试通过。
- retail、airline、telecom 的真实查询工具均已在隔离副本中执行并验证。
- 3 项随机小模型 CPU 训练测试通过，覆盖 IL、IWM 两阶段权重更新、SR 混合训练。
- Ruff lint/format 通过；retail 原生 train/test 为 74/40 个任务，无重叠。
- 尚未调用真实生成模型进行数据采集，也未训练正式基座模型或产生正式 benchmark 分数。
