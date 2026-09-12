# 当前 GPU 实例的运行代码快照

此版本收录本机已经验证的修复与 full 启动代码。最新 full 于 2026-09-12 21:42 UTC 在在线校准阶段停止：完成 29 条，第 30 条 Policy 输出截断。不能把此快照当作完整实验已成功或最终性能已达标的证明。

## 内容与边界

- `src/`、`tests/`、`configs/`、`prompts/`、`uv.lock` 与运行目录中的代码逐字节一致。
- `scripts/runtime/` 收录原本位于仓库外的固定命令启动器、预检、smoke、数据构建入口、监测和 supervisor 部署模板。
- 模型原始响应只保存在本机运行账本中；本仓库保存审计实现，不上传真实轨迹或响应。
- 不包含凭据、模型权重、虚拟环境、数据 bundle、SQLite、日志、缓存和运行结果。

## 部署路径

这些是当前 Vast Linux 实例的部署快照，保留已验证的绝对路径，不是任意目录下的一键安装器：

| 路径 | 内容 |
|---|---|
| `/root/toolsandbox` | 本目录代码 |
| `/root/toolsandbox-runtime` | 启动器脚本、运行配置及生成的实验产物 |
| `/root/toolsandbox-runtime/models/Qwen3-32B` | 单独准备的模型权重 |
| `/root/toolsandbox-runtime/vllm-env` | 独立 Python 3.12 vLLM 环境 |
| `/root/.config/toolsandbox-runtime/credentials.json` | 本机凭据；永不复制进仓库 |

项目依赖使用 Python 3.10 和 `uv sync --frozen`。GPU 服务使用独立 Python 3.12 环境；已安装的关键版本记录在 `scripts/runtime/vllm-requirements.txt`，它不是整个 GPU 环境的依赖锁。Qwen 的完整启动参数见 `supervisor-scripts/toolsandbox-qwen.sh`。

在新实例安装运行脚本时，将 `scripts/runtime/` 顶层 `.py` 复制到 `/root/toolsandbox-runtime/`，将 `config/*.json` 复制到同目录，将 `toolsandbox-with-secrets` 安装到 `/root/.local/bin/`。保留 owner-only 凭据目录 0700、文件 0600；使用启动器的 `configure-openai` 和 `configure-rapidapi` 在终端隐藏输入密钥。

supervisor 模板分别放在 `/opt/supervisor-scripts/` 和 `/etc/supervisor/conf.d/`；它们依赖 Vast 已安装的 logging/environment helpers。不要在实验运行中覆盖其代码、配置或重启服务。

## 数据与启动前置条件

`configs/run/live_full_v2.json` 和 `live_calibration_v1.json` 绑定当前本机数据目录与哈希。数据 bundle、执行环境 attestation、image provenance、外部预检结果和校准证据均是本地生成的运行材料，不随源码发布。新机器必须重新构建并核验这些材料；不能复制旧哈希伪装验证通过。相关实现位于 `reproducibility/dataset_manifest.py`、`orchestration/live_environment_attestation.py` 和 `orchestration/live_full_bootstrap.py`。

准备完毕后先检查：

```bash
cd /root/toolsandbox
uv run --frozen python -m toolsandbox_pipeline.orchestration.live_full_bootstrap --validate-only
/root/.local/bin/toolsandbox-with-secrets external-preflight
/root/.local/bin/toolsandbox-with-secrets preflight
```

运行入口：`toolsandbox-with-secrets reflection-smoke`、`live-calibration`、`full`。正式长任务使用 supervisor 的 `toolsandbox-full` 服务。测试集不属于当前 full 入口。

## 实时监测

监测先无锁复制数据库，验证副本稳定性，再查询副本；绝不通过 SQLite 连接活库。源数据库使用 DELETE journal 和 timeout=0，普通只读连接也可能阻塞实验写入。

每次启动新 campaign，需要替换 `toolsandbox-progress.sh` 中的 `--campaign` 与 `--online` 两个路径。发布模板中的路径只是打包时的实例值，不应在新机器原样启动。监测每 15 秒写 JSON、Markdown 和 HTML；遇并发写入保留上一份快照并标记 stale；实验停止后保存最后快照退出。

## 已做验证

- 调用 ID、provider/online、native 执行与恢复，以及 full 校准/训练接线：382 项相关测试通过。
- runtime 无锁监测：9 项合成测试通过，包括并发 writer 连续 18 次提交。
- 原始重复调用 ID 的失败响应离线重新解析成功，参数、顺序、原始字节均保持不变。
- 当前 full 已通过先前重复 ID 导致失败的位置，但在第 30 条出现 Policy 输出截断；最终三轮尚未开始，后续仍需解决长度上限问题。

发布副本的 runtime 单元测试可在 `scripts/runtime` 内运行：

```bash
python3 -m unittest -q test_secret_launcher.py test_progress_monitor.py
```
