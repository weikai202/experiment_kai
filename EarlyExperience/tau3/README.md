# Early Experience baseline for τ³-bench

包含 IL、IWM → IL、SR 的数据生成、训练和原生 τ³ 评测适配。
支持 retail、airline、telecom 文本半双工任务。

## 在固定版本的 τ³ 环境中使用

本目录保存实验模块，目录结构与 τ³ 仓库一致。先将模块复制到固定版本的上游仓库，
然后在该仓库根目录按照完整指南安装依赖、生成数据、训练和评测：

```bash
# 在 experiment_kai 仓库根目录执行；为上游选择一个新的目录。
git clone https://github.com/sierra-research/tau2-bench.git ../tau3-early-experience
git -C ../tau3-early-experience checkout 17e07b1da2bbc0cadfddeea36412686e0604127b
cp -R EarlyExperience/tau3/src/experiments/early_experience \
  ../tau3-early-experience/src/experiments/
cd ../tau3-early-experience
```

接着阅读 [完整运行指南](src/experiments/early_experience/README.md)。
原生 Python 包名及 CLI 仍使用 `tau2`，这是 τ³ 上游的命名。

## 文件

- `src/experiments/early_experience/core.py`：环境分支隔离、训练样本格式、任务划分校验。
- `pipeline.py`：真实分支采样、反思生成、断点续跑和 SFT 导出。
- `train.py`：IL、IWM 两阶段、SR 混合训练。
- `agent.py`、`__main__.py`：推理适配和采集/评测命令行。
- `tests/`：数据、真实领域工具、原生评估器以及小模型 CPU 训练测试。

验证：14 项数据/环境/评估器测试及 3 项 CPU 训练测试通过，Ruff 检查通过。
没有运行正式基座模型训练或生成正式 benchmark 分数。

方法参考：[OSU-NLP-Group/EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience)。
