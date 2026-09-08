# 实验协议与复现记录

日期：2026-09-07。

- 用户指定 policy 和数据生成模型：Qwen3-32B，non-thinking。
- 独立项目 `/home/weik/early-experience-bfcl`，复用已有 Gorilla，未修改上游代码。
- 当前 BFCL commit：`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，2026-03-23。
- 参考 EarlyExperience 文档指定 2025-12 BFCL 版本；这里保留并记录当前安装版本。当前检查结果支持数据兼容性，不意味着逐项复现论文实验。
- 专家输入为 `HuanzhiMao/BFCL-Result/main/2025-12-16` 下的 `claude-opus-4-5-20251101-FC` Base result 和 score；精确 SHA256 见 `data/prepared/manifest.json`。
- 仅使用官方评分成功的 162 个 Base case。按数值 case ID 排序后用独立 `random.Random(42)` shuffle，前 121 个训练，余下 41 个 heldout。保存具体 ID；上游文档未公开完整 split ID 列表，不能断言与其原始划分逐 ID 相同。
- 三个 OOD 类别各 200 个 case，完全隔离于训练。38 个失败专家 Base case 不用于这个协议的训练或 heldout 报告。
- 观察来自真实模拟器。候选方法结构采样 K=10，一次生成参数，逐候选 deepcopy 执行；SR 使用其中 K=3。
- IWM 预测纯描述性观察摘要。原始执行返回及 post-state 保存在中间数据，不直接暴露为 policy 的额外输入。
- SR 使用显式 `<reflection>` 文本及固定专家动作，生成目标约 200 words、提示软上限约 500 words；Qwen thinking 始终关闭。
- API 输出长度上限用于服务资源控制；若 finish_reason 不是 stop，则样本被判为未完成并报错，不接受截断样本。
- IWM 保留执行错误；SR 默认审计词汇泄漏，显式 `--filter-leaks` 启用参考仓库的词汇筛选，并记录影响。
- 专家自然语言回合结束回复统一为 `[]`，增加相应停止训练样本；三组格式一致。
- 使用官方 prompting 模式 system prompt 与工具文档；Qwen 工具观察转为等价 `<tool_response>` user 文本，已通过官方 tokenizer 等价性测试。
- 评分调用官方 state/response checker 和 irrelevance checker。执行器以直接方法调用和 AST literal 参数替代上游 eval；已检查四类数据全部 4,625 个参考调用的语法兼容性。
- 训练默认全参数 SFT，1 epoch/stage、lr=1e-5；这些是显式可配置的适配起点。LoRA 可选，实验中须保持三组一致并注明。

离线验证：22 个测试通过。全部成功专家轨迹的工具观察回放不一致数为 0。expert 训练共 1,196 个样本、8,461,337 tokens，其中监督 token 数 15,929，最大序列长度 10,197。tokenizer 验证使用官方 Qwen3 文件与当前环境的 Transformers 5.16.1；训练额外依赖单独约束于 Transformers 4.51–4.x，完整 GPU 训练尚未执行。

验证文件：`outputs/validation/tests.xml`、`outputs/validation/data_and_tokenizer.json`。所有测试替身和 oracle 仅用于验证代码，不构成模型测评结果。真实模型 API 调用数和 GPU 训练运行数均为 0。
