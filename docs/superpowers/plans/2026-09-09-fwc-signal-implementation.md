# FWC-Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改上游 AGG 的前提下，构建严格源域训练、记录级推理的 FWC-Signal 特征效用组合故障分类模块，并只在冻结后评估目标域。

**Architecture:** 原 AGG 先在来源数据训练并冻结特征提取器。来源域校准器产生每个深度特征的贡献、稳定性和冗余度；低容量属性预测器在未知工况预测这些属性。记录级 FWC 将属性变为颜色、合法形状和数据驱动效用，精确装入 3×3 网格，并通过两阶段藏品规则形成软门控；分类仍在窗口级完成，再在记录级汇聚概率。

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, pandas, scikit-learn, pytest；CUDA 只用于训练、窗口前向和批量遮蔽，元数据审计与来源校准允许离线 CPU 执行。

---

## 固定实施边界

- 新代码只能位于 `D:\pycharm\AEI\DG-for-RMFD-main\AGG_FWC`，不得修改父项目源文件。
- 所有源/验证/测试分组必须以原始记录和轴承编号为单位，在切窗前固定；不得从文件名自动猜测故障标签。
- 目标域只在最终冻结评估时读取；不得参与 epoch 选择、属性校准、归一化分位数、单特征 Logistic Regression、外置特征筛选或任何阈值选择。
- 不使用人工价格表或随机价格扰动。效用固定为 `u=C*S*(1-R)`，总效用为 `P=s*u`。
- 当前目录不是 Git 仓库；各任务完成后写入 `results/<run_id>/` 的版本、命令和测试摘要，不执行不存在的 Git 提交。

## 文件结构

- Create: `AGG_FWC/protocol.py` — 记录清单、组划分验证、目标域访问守卫。
- Create: `AGG_FWC/models/fwc_types.py` — 不可变物品、布局、校准统计数据类型。
- Create: `AGG_FWC/models/fwc_calibration.py` — 来源域贡献/稳定性/冗余度、单特征外置筛选与校准文件 I/O。
- Create: `AGG_FWC/models/fwc_attributes.py` — 1536 输出的属性预测器及其训练/保存接口。
- Create: `AGG_FWC/models/fwc_packing.py` — 3×3 形状放置与精确位掩码装箱。
- Modify: `AGG_FWC/models/feature_combination.py` — 从恒等层替换为记录级 FWC 门控和两阶段藏品流程；保留恒等层。
- Modify: `AGG_FWC/models/agg_fwc_model.py` — 暴露原始 512 维特征、按给定门控分类，保持原 AGG 等价路径。
- Modify: `AGG_FWC/config.py`, `AGG_FWC/run.py` — 增加显式协议、阶段、阈值和只读目标域开关；保存完整配置。
- Modify: `AGG_FWC/train.py` — 分阶段训练、来源验证早停/选择、最终一次目标测试、记录级预测与审计产物。
- Modify: `AGG_FWC/evaluate.py` — 记录级指标、分组预测保存和 FWC 诊断汇总。
- Create tests: `AGG_FWC/tests/test_protocol.py`, `test_fwc_calibration.py`, `test_fwc_attributes.py`, `test_fwc_packing.py`, `test_fwc_inference.py`；扩充现有 `test_feature_combination.py`, `test_runner.py`, `test_baseline_equivalence.py`。

### Task 1: 建立数据协议硬门槛

**Files:**
- Create: `AGG_FWC/protocol.py`
- Create: `AGG_FWC/tests/test_protocol.py`
- Modify: `AGG_FWC/datasets/PU_bearing.py`

- [ ] **Step 1: 写入失败测试，禁止轴承 ID 被当作故障类别**

```python
def test_manifest_rejects_bearing_id_as_fault_label(tmp_path):
    manifest = pd.DataFrame({
        "record_id": ["r1"], "condition_id": ["N15"],
        "bearing_id": ["KA04"], "fault_label": ["KA04"], "path": ["x.mat"],
    })
    with pytest.raises(ValueError, match="bearing_id cannot be fault_label"):
        validate_manifest(manifest)
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_protocol.py::test_manifest_rejects_bearing_id_as_fault_label -v`  
Expected: FAIL because `validate_manifest` does not exist.

- [ ] **Step 3: 实现清单与泄露检查**

```python
class ProtocolBlockedError(RuntimeError):
    """Raised when a requested experiment lacks auditable fault labels or grouped splits."""

REQUIRED_COLUMNS = {"record_id", "condition_id", "bearing_id", "fault_label", "path"}

def validate_manifest(frame: pd.DataFrame) -> None:
    if REQUIRED_COLUMNS.difference(frame.columns) or frame.empty:
        raise ValueError("manifest is missing required record metadata")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id must be unique before windowing")
    if (frame["bearing_id"].astype(str) == frame["fault_label"].astype(str)).any():
        raise ValueError("bearing_id cannot be fault_label")

def assert_disjoint_bearings(*frames: pd.DataFrame) -> None:
    groups = [set(frame["bearing_id"]) for frame in frames]
    if any(left & right for i, left in enumerate(groups) for right in groups[i + 1:]):
        raise ValueError("bearing leakage across splits")
```

Extend the PU loader to build the manifest from verified Paderborn metadata, preserving a distinct `record_id` for every `.mat` measurement before windows are generated. If verified metadata cannot supply a genuine fault label distinct from bearing identity, raise `ProtocolBlockedError`, write `protocol_audit.json`, and stop before any training.

- [ ] **Step 4: 追加协议测试并运行**

Add tests for duplicate records, overlapping train/validation/test bearings, missing metadata and a valid three-way split.  
Run: `pytest AGG_FWC/tests/test_protocol.py -v`  
Expected: PASS.

- [ ] **Step 5: 写入来源可追溯审计产物**

Implement `write_protocol_audit(manifest, split, path)` to save record counts, fault-class counts, bearing intersections and hashes of each split.  
Run: `pytest AGG_FWC/tests/test_protocol.py::test_audit_contains_empty_pairwise_intersections -v`  
Expected: PASS.

### Task 2: 实现来源域属性校准与外置特征筛选

**Files:**
- Create: `AGG_FWC/models/fwc_types.py`
- Create: `AGG_FWC/models/fwc_calibration.py`
- Create: `AGG_FWC/tests/test_fwc_calibration.py`

- [ ] **Step 1: 写入贡献、稳定性、冗余度的失败测试**

```python
def test_attribute_calibration_prefers_stable_nonredundant_contributor():
    features = np.array([[2., 0.], [2.1, 0.], [-2., 0.], [-2.1, 0.]])
    labels = np.array([1, 1, 0, 0])
    record_ids = np.array(["a", "a", "b", "b"])
    report = calibrate_condition(features, labels, record_ids, repeats=5, seed=2026)
    assert report.contribution[0] > report.contribution[1]
    assert report.redundancy[0] == pytest.approx(0.0)
    assert np.all((0.0 <= report.stability) & (report.stability <= 1.0))
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_fwc_calibration.py::test_attribute_calibration_prefers_stable_nonredundant_contributor -v`  
Expected: FAIL because `calibrate_condition` does not exist.

- [ ] **Step 3: 实现可序列化校准报告和来源域算法**

```python
@dataclass(frozen=True)
class Item:
    index: int
    color: str
    size: int
    utility: float
    c: float = 0.0
    s: float = 0.0
    r: float = 0.0

@dataclass(frozen=True)
class Placement:
    item_index: int
    mask: int
    shape: tuple[int, int]

@dataclass(frozen=True)
class Layout:
    selected_indices: tuple[int, ...]
    placements: tuple[Placement, ...]
    total_utility: float
    cells_used: int
    contains_large_red: bool

@dataclass(frozen=True)
class ConditionAttributeReport:
    condition_id: str
    contribution: np.ndarray
    stability: np.ndarray
    redundancy: np.ndarray
    c_quantiles: tuple[float, float]
    delta_margin_quantiles: tuple[float, float]

def utility(c: np.ndarray, s: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.clip(c, 0, 1) * np.clip(s, 0, 1) * (1 - np.clip(r, 0, 1))
```

For contribution, compute an out-of-fold, grouped-record macro-F1 drop after masking each feature. For stability, run exactly five grouped resamples and use inverse normalized contribution standard deviation. For redundancy, compute the maximum absolute Pearson correlation against features with greater contribution; assign zero when no higher-contribution feature exists. Persist all arrays and the source-only 5th/95th quantiles with `np.savez_compressed(..., allow_pickle=False)` compatible arrays.

- [ ] **Step 4: 加入外置特征规则并测试**

```python
def select_external_features(condition_accuracies: np.ndarray) -> np.ndarray:
    return (condition_accuracies.mean(axis=0) >= 0.85) & (condition_accuracies.max(axis=0) >= 0.95)
```

Train one regularized `LogisticRegression` per feature and source condition using grouped folds; store only out-of-fold accuracies. Test the 85% mean, 95% any-condition boundary, empty result and no-target-data guard.  
Run: `pytest AGG_FWC/tests/test_fwc_calibration.py -v`  
Expected: PASS.

### Task 3: 训练并冻结属性预测器

**Files:**
- Create: `AGG_FWC/models/fwc_attributes.py`
- Create: `AGG_FWC/tests/test_fwc_attributes.py`
- Modify: `AGG_FWC/config.py`

- [ ] **Step 1: 写入形状与冻结状态的失败测试**

```python
def test_attribute_predictor_returns_bounded_three_by_512_outputs():
    predictor = AttributePredictor(feature_dim=512, hidden_dim=128)
    c_hat, s_hat, r_hat = predictor(torch.randn(3, 512))
    assert c_hat.shape == s_hat.shape == r_hat.shape == (3, 512)
    assert all(torch.all((0 <= value) & (value <= 1)) for value in (c_hat, s_hat, r_hat))
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_fwc_attributes.py::test_attribute_predictor_returns_bounded_three_by_512_outputs -v`  
Expected: FAIL because `AttributePredictor` does not exist.

- [ ] **Step 3: 实现最小预测器与保存接口**

```python
class AttributePredictor(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3 * feature_dim))
    def forward(self, z):
        c, s, r = torch.sigmoid(self.net(z)).chunk(3, dim=-1)
        return c, s, r
```

Add config fields `fwc_stage`, `attribute_hidden_dim=128`, `attribute_epoch`, and `attribute_lr`. Train only on frozen source features and source calibration labels with mean squared error. Select checkpoint only by held-out source-record loss, then set every extractor and attribute-predictor parameter `requires_grad=False`.

- [ ] **Step 4: 添加来源域隔离测试并运行**

Test that target tensors cannot be passed to `fit_attribute_predictor`, and that a loaded predictor has no trainable parameters.  
Run: `pytest AGG_FWC/tests/test_fwc_attributes.py -v`  
Expected: PASS.

### Task 4: 实现颜色、合法格数和数据驱动效用

**Files:**
- Modify: `AGG_FWC/models/fwc_types.py`
- Modify: `AGG_FWC/models/feature_combination.py`
- Modify: `AGG_FWC/tests/test_feature_combination.py`

- [ ] **Step 1: 写入颜色配额与效用的失败测试**

```python
def test_assign_items_uses_fixed_palette_and_data_derived_utility():
    c = torch.linspace(1, 0, 512).unsqueeze(0)
    s = torch.ones_like(c)
    r = torch.zeros_like(c)
    items = assign_items(c, s, r)[0]
    assert sum(item.color == "red" for item in items) <= 20
    assert all(item.utility == pytest.approx(item.c * item.s * (1 - item.r)) for item in items)
    assert all(item.size in LEGAL_SIZES[item.color] for item in items)
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_feature_combination.py::test_assign_items_uses_fixed_palette_and_data_derived_utility -v`  
Expected: FAIL because `assign_items` does not exist.

- [ ] **Step 3: 实现确定性物品生成**

```python
def assign_items(c: torch.Tensor, s: torch.Tensor, r: torch.Tensor) -> list[list[Item]]:
    """Map one batch of record-level C/S/R vectors to deterministic FWC items."""
```

Define `LEGAL_SIZES={"gray":(1,2), "green":(1,2), "blue":(1,2,3,4,6), "purple":(1,2,4,6), "gold":(1,2,3,4,6), "red":(1,2,4,6,9)}` and fixed quotas `(154,128,102,67,41,20)`. Use feature index as all ranking ties. Grant red only when it is in the top 20 contribution ranks and `Q=0.5*C+0.3*S+0.2*(1-R)>=0.80`; downgrade failures to gold. Compute `utility=C*S*(1-R)`, choose legal size from within-color redundancy rank, and reverse the low-redundancy size preference only for red items with utility at least the record's red-candidate median.

- [ ] **Step 4: 完成边界测试并运行**

Test red downgrade, no-red records, all ties, legal size constraints, red median equality, and deterministic repeated output.  
Run: `pytest AGG_FWC/tests/test_feature_combination.py -v`  
Expected: PASS.

### Task 5: 实现精确 3×3 位掩码装箱

**Files:**
- Create: `AGG_FWC/models/fwc_packing.py`
- Create: `AGG_FWC/tests/test_fwc_packing.py`

- [ ] **Step 1: 写入旋转、无重叠和红色 5% 优先的失败测试**

```python
def test_packer_uses_rotated_two_by_three_piece_when_required():
    item = Item(index=7, color="blue", size=6, utility=1.0)
    layout = exact_pack([item])
    assert layout.cells_used == 6
    assert layout.placements[0].shape in {(2, 3), (3, 2)}

def test_packer_prefers_large_red_within_five_percent():
    red = Item(index=0, color="red", size=6, utility=0.95)
    alternatives = [Item(index=i, color="blue", size=1, utility=1.0) for i in range(1, 7)]
    assert exact_pack([red, *alternatives]).contains_large_red
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_fwc_packing.py -v`  
Expected: FAIL because `exact_pack` does not exist.

- [ ] **Step 3: 实现位掩码精确搜索**

Precompute all 9-bit masks for each allowed rectangle and its rotation. Use branch-and-bound recursion over candidates sorted by `P=s*utility`; reject overlaps with `occupied_mask & placement_mask`. First find global maximum total `P`; among layouts reaching at least `0.95*P_max`, use red occupied cells, then red effective utility, then feature-index tuple as deterministic tie-breakers. Return an immutable `Layout` with selected indices, placement masks, total utility and red diagnostics.

- [ ] **Step 4: 加入穷举交叉验证并运行**

For random candidate sets of at most eight items, compare `exact_pack` to an exhaustive reference enumerator; test empty input, nine-cell red, impossible shape and rotation.  
Run: `pytest AGG_FWC/tests/test_fwc_packing.py -v`  
Expected: PASS.

### Task 6: 构建两阶段 FWC 门控推理

**Files:**
- Modify: `AGG_FWC/models/feature_combination.py`
- Modify: `AGG_FWC/models/agg_fwc_model.py`
- Create: `AGG_FWC/tests/test_fwc_inference.py`
- Modify: `AGG_FWC/tests/test_baseline_equivalence.py`

- [ ] **Step 1: 写入记录级一致性与门控失败测试**

```python
def test_same_record_uses_one_layout_for_all_windows():
    module = FWCFeatureCombination(fake_predictor())
    result = module.record_gate(torch.randn(4, 512), record_id="r1")
    assert result.window_gates.shape == (4, 512)
    assert torch.equal(result.window_gates[0], result.window_gates[3])
    assert torch.all(result.window_gates >= 1)
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_fwc_inference.py::test_same_record_uses_one_layout_for_all_windows -v`  
Expected: FAIL because `FWCFeatureCombination` does not exist.

- [ ] **Step 3: 实现基础门控与模型接口**

Keep `IdentityFeatureCombination` unchanged. Add `FWCFeatureCombination.record_gate(record_features, record_id)` that averages predictor attributes over windows, calls `assign_items` and `exact_pack`, then returns a repeated gate with `1+log(1+P_effective)` only at selected/external indices and 1 elsewhere. Add `AGGFWCModel.extract_raw_features(inputs)` and `AGGFWCModel.classify_features(features, gate=None)` so the frozen extractor is called once and the classifier receives `features*gate`.

- [ ] **Step 4: 实现藏品二次装箱**

Implement `apply_collectibles` as: initial gate and preliminary record prediction; batched, exact feature masking for every gold-or-lower non-external candidate; source-calibration 5th/95th clipping for `Δm` and `Ĉ`; dual 0.40 gates; `K=.4Δm+.6Ĉ`; zero/one/two selection thresholds `.65/.55`; `r=1+9*clip((K-.55)/.45,0,1)`; exactly one repack. Record all candidates and decisions in diagnostics. No target-wide aggregation is permitted.

- [ ] **Step 5: 运行推理测试**

Test no candidate, one and two collectibles, external exclusion, red exclusion, source-only quantile lookup, one-repack-only, gate lower bound 1 and original identity equivalence.  
Run: `pytest AGG_FWC/tests/test_fwc_inference.py AGG_FWC/tests/test_baseline_equivalence.py -v`  
Expected: PASS.

### Task 7: 改造训练与评价为严格分阶段协议

**Files:**
- Modify: `AGG_FWC/train.py`
- Modify: `AGG_FWC/evaluate.py`
- Modify: `AGG_FWC/config.py`
- Modify: `AGG_FWC/run.py`
- Modify: `AGG_FWC/tests/test_runner.py`
- Modify: `AGG_FWC/tests/test_evaluate.py`

- [ ] **Step 1: 写入目标域盲测失败测试**

```python
def test_trainer_never_iterates_target_before_final_evaluate(monkeypatch, trainer):
    trainer.dataloaders["tar"] = ExplodingLoader("target accessed too early")
    trainer.train_source_stages()
    assert trainer.source_checkpoint_path.exists()
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_runner.py::test_trainer_never_iterates_target_before_final_evaluate -v`  
Expected: FAIL because source stages and final evaluation are not separated.

- [ ] **Step 3: 拆分训练阶段**

Replace epoch-level target validation with: `train_agg_on_sources()`, `calibrate_sources()`, `fit_attribute_predictor()`, `train_classifier_with_fwc_on_sources()`, and `evaluate_target_once()`. Select every checkpoint using a held-out source-record validation loader only. Add explicit stage names to the config and checkpoint filenames. `evaluate_target_once()` must assert all frozen component hashes and raise if called before the final source checkpoint exists.

- [ ] **Step 4: 实现记录级评价和诊断文件**

Save `record_predictions.npz` containing record IDs, labels, probabilities and predictions; compute record-level Macro-F1, per-class precision/recall/F1 and confusion matrix. Save `fwc_diagnostics.jsonl` with selected items, external items, red priority decision, collectibles, effective utilities and inference milliseconds. Keep window predictions separately for audit only.

- [ ] **Step 5: 运行训练/评价测试**

Run: `pytest AGG_FWC/tests/test_runner.py AGG_FWC/tests/test_evaluate.py -v`  
Expected: PASS, including exactly one target-loader traversal after final checkpoint selection.

### Task 8: 完整验证、烟雾实验与正式实验清单

**Files:**
- Modify: `AGG_FWC/README.md`
- Modify: `AGG_FWC/run.py`
- Create: `AGG_FWC/tests/test_fwc_smoke.py`

- [ ] **Step 1: 写入无数据单元测试和合成烟雾测试**

```python
def test_fwc_smoke_writes_protocol_and_diagnostics(tmp_path, synthetic_manifest):
    result = run_fwc_smoke(tmp_path, synthetic_manifest, seed=2026)
    assert (result.result_dir / "protocol_audit.json").exists()
    assert (result.result_dir / "fwc_diagnostics.jsonl").exists()
    assert result.status["formal_result"] is False
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest AGG_FWC/tests/test_fwc_smoke.py -v`  
Expected: FAIL because the staged runner and artifacts do not exist.

- [ ] **Step 3: 增加运行模式和文档**

```python
def run_fwc_smoke(output_root: Path, manifest: pd.DataFrame, seed: int):
    """Run one synthetic source-only staged pass and return its artifact bundle."""
```

Provide exact commands for: protocol audit only; source-only smoke run; full single-seed run; five fixed seeds; each ablation; frozen external test. The README must state that smoke/adapted results are not formal results and that a `ProtocolBlockedError` means no performance number may be reported.

- [ ] **Step 4: 跑全套测试和最小 GPU 烟雾实验**

Run: `pytest AGG_FWC/tests -q`  
Expected: PASS.

Run: `python -m AGG_FWC.run --fwc-stage smoke --seed 2026 --require-cuda --run_id fwc-smoke-2026`  
Expected: `status.json` marks `formal_result: false`; it either records CUDA execution or a clearly blocked CUDA status, never silently falls back when `--require-cuda` is set.

- [ ] **Step 5: 建立正式实验执行表**

After protocol audit passes, execute one variable at a time: AGG; AGG+packing gate; +external; full FWC; no-red-priority; no-stability; no-redundancy. For every row run the agreed fixed seed set and report all values, mean±sample standard deviation, config hash, protocol-audit hash and checkpoint hash. Run HUST only after this table is frozen.

## 覆盖性自检

- 严格源域/目标盲测：Tasks 1 and 7。
- 512 维 `C/S/R`、来源校准、外置 Logistic Regression：Tasks 2 and 3。
- 金字塔颜色、红色阈值、数据驱动效用、形状：Task 4。
- 精确可旋转 3×3 装箱与红色 5% 优先：Task 5。
- 记录级属性聚合、软门控、外置特征、两阶段藏品：Task 6。
- 指标、负面结果、五种子和外部冻结测试：Tasks 7 and 8。

## 计划自检

- 无 `TODO`、`TBD` 或“以后补充”步骤。
- 所有涉及代码的任务均先写失败测试，再实现最小逻辑并运行对应测试。
- 每个新增接口均在首次定义任务中给出名称、输入和输出边界；后续任务沿用相同名称。
- 当前目录没有 Git 元数据，计划不伪造提交；运行产物和审计哈希承担最小可追溯责任。
