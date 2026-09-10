# AGG_FWC

这是 `DG-for-RMFD` 中 AGG 基线的独立配置副本，不是新方法。原项目的七种方法及其测试、模型文件均不修改；本目录只提供 AGG 实验的配置与说明。

## 边界与实验纪律

- `data_root` 默认是原 PU 数据路径 `D:/datasets/PU`。raw data 位于外部，只读；本配置不会读取数据、创建模型或启动训练。
- 初始 identity 路径是控制组，用于确认封装不会改变原始 extractor/classifier 的行为。
- 后续实验每次只改一个因素，并记录对应的配置、随机种子和产物目录。
- 每个运行使用 `results/<run_id>`、`checkpoints/<run_id>`、`logs/<run_id>` 三个独立目录；已有任一目录时不会覆盖结果。归档时结果与 checkpoint 需要合计这两个目录：`final.pt` 位于 `checkpoints/<run_id>/final.pt`，不位于 `result_dir`。
- `config.json` 会记录 `smoke` 和 `require_cuda`。`status.json` 只接受 `success`、`blocked`、`incomplete`：success 表示链路完成，blocked 表示未开始训练，incomplete 表示运行失败；`complete` 和 `formal_result` 与状态一致，smoke 永远不是 formal result。
- `inner_lr`、`trade_off`、`momentum` 为兼容原 AGG CLI 保留；AGG-only 训练路径不使用这些参数。

## 配置测试

在 `D:\pycharm\AEI\DG-for-RMFD-main` 下运行：

```powershell
python -m pytest AGG_FWC/tests/test_config.py -q
python -m pytest AGG_FWC/tests/test_baseline_equivalence.py AGG_FWC/tests/test_feature_combination.py -q
```

## 后续运行模板

配置 smoke 检查（不读取数据）：

```powershell
python -c "from AGG_FWC.config import build_config; print(build_config(run_id='smoke'))"
```

AGG-only 入口命令如下；`--smoke` 只缩小本次运行的 epoch 和样本数，并在 Config/`config.json` 中记录 `smoke: true`，不改变 full 默认值：

```powershell
python -m AGG_FWC.run --run_id smoke --smoke --output_root AGG_FWC/runs
python -m AGG_FWC.run --run_id full --output_root AGG_FWC/runs
```

如果实验协议必须使用 GPU，可增加 `--require-cuda`。CUDA 不可用时该 run 会写入
`status.json: blocked` 和明确的日志，不会启动训练或生成指标；未要求 GPU 时保留
默认 CPU fallback，并在 metadata 和 `status.json` 中记录 `cuda_fallback: true`。

启动日志会记录 `torch.cuda.is_available`、实际 `device`、`torch.__version__`
和 `torch.version.cuda`；最终 target 评估结束后，指标和最终 checkpoint 分别写入该
run 的 `results/<run_id>` 与 `checkpoints/<run_id>`，并在
`results/<run_id>/config.json` 保存完整配置快照。最终评估完成后按稳定顺序写出
`config.json`、`final.pt`、`predictions.npz`、`epoch_metrics.json`，最后写
`metrics.json`；因此 `metrics.json` 声明的 checkpoint 已先存在。`metrics.json` 至少
包含 `run_id`、`config_path`、`checkpoint_path`，路径以 JSON 字符串保存。环境 metadata
同时写入训练日志。本 Task 未执行真实数据训练，因此不对训练完成或性能作任何声明。

训练结果还包括 `results/<run_id>/epoch_metrics.json`（逐 epoch 损失与准确率）、
`results/<run_id>/predictions.npz`（最终 target labels/predictions）和
`results/<run_id>/metrics.json`（最终 target 的 accuracy、macro/weighted F1、
逐类指标及 confusion matrix）。

`--smoke` 产物只用于检查配置、数据加载和训练/保存链路，不得作为正式实验结果、
性能比较或论文证据；正式结果必须使用非 smoke 配置并按相同的结果目录与
checkpoint 目录合计结构归档。

## 源域特征贡献审计

`models/feature_contribution_audit.py` 提供冻结分类器的源域后验审计。它将每个窗口的
概率先按 `(condition_id, record_id)` 聚合为记录级概率，再逐维把特征替换为源域中位数，
输出 NLL 增量和 Macro-F1 降幅。该报告只用于检验“特征价值是否对应分类贡献”，不改变
模型权重、训练损失、颜色/价格映射或 3×3 打包，也不接收或读取目标域数据。

```python
from AGG_FWC.models import audit_source_feature_contributions

report = audit_source_feature_contributions(
    frozen_classifier,
    source_features,
    source_labels,
    source_record_ids,
    source_condition_ids,
    device="cuda",
    seed=2026,
)
```

报告可用 `save_feature_contribution_audit` 保存为同名 `.npz` 数值文件和 `.json` 元数据；
保存函数拒绝覆盖已有产物。只有源训练/源验证数据可以进入审计，目标集不得用于特征排序、
阈值选择或结果筛选。

## Task8：严格 PU manifest 适配

`datasets/strict_pu_adapter.py` 不会把旧版 `PU_bearing.data_load()` 产生的工况/故障类
窗口顺序伪装成轴承记录。严格构建需要外部提供一行一个原始测量记录的 CSV/JSON manifest，
至少包含 `record_id`、`condition_id`、`bearing_id`、`fault_label`、
`label_provenance`、`path` 和 `split`；`split` 必须恰好包含
`source_train`、`source_val`、`target`。

默认使用轴承独立协议：

```python
from AGG_FWC.datasets.strict_pu_adapter import build_strict_pu_loaders

bundle = build_strict_pu_loaders(
    data_root="D:/datasets/PU",
    manifest_path="D:/datasets/PU/manifest.csv",
    audit_path="D:/datasets/PU/protocol_audit.json",
)
```

若严格复现原论文按工况划分、允许同一轴承跨工况重复，必须显式声明：

```python
bundle = build_strict_pu_loaders(
    data_root="D:/datasets/PU",
    manifest_path="D:/datasets/PU/manifest.csv",
    audit_path="D:/datasets/PU/paper-condition-audit.json",
    protocol_mode="paper_condition",
)
# 等价写法：bearing_disjoint=False
```

两种显式写法都会在 `protocol_audit.json` 中记录 `protocol_mode`、
`bearing_disjoint`、`condition_split` 和 `bearing_intersections`。未声明时仍拒绝
跨 split 轴承重复；缺少真实轴承级 metadata 时直接阻断，不能生成 `record_id`。
当前适配层返回的是可审计的记录级 loader metadata，不是把旧窗口读取器包装成可训练
loader；真实训练前仍需实现保持 `record_id` 的逐记录数据物化。

评估器 CLI 要求 `--metadata` 是 JSON object，且必须包含非空 `run_id` 和
`config_path`；labels/predictions 必须是严格一维、非空且长度相同的整数类别索引。
NaN/Infinity 会被拒绝，已有 `metrics.json` 也不会覆盖。旧的
`results/task6-smoke-20260908` 是历史不完整快照，其 `status.json` 标记
`legacy: true`，不能因旧 `config.json` 缺少 `smoke` 字段而被伪造为新的 smoke 或正式结果。

`--cuda_device` 会在 seed 设置前写入 `CUDA_VISIBLE_DEVICES`，并在环境
metadata 中记录请求值与可见 GPU 数量。训练、数据加载或结果保存发生异常时，
入口会写入 `status.json: incomplete` 和失败日志，并从 root logger 移除并关闭本次
run 的日志 handler，避免后续运行重复写入。当前真实环境阻塞探针记录在
`logs/environment_blocker.log`；未执行正式训练。
