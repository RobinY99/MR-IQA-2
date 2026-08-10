# MR-IQA-2：视觉质量推理的细粒度 Credit Assignment

<p align="center">
  <a href="https://huggingface.co/RobinY99/MR-IQA-2">模型权重</a> |
  <a href="docs/training.md">训练</a> |
  <a href="docs/evaluation.md">评测</a> |
  <a href="docs/checkpoints.md">Checkpoint 与结果</a> |
  <a href="README.md">English</a>
</p>

MR-IQA-2 研究 token 级 reward credit 和 KL 正则路由如何影响多模态
Actor。Actor 输出图像质量 `evidence`、编辑 `solution` 与数值 `rating`；
冻结的 Editor 仅执行 `solution`，冻结 E5 Judge 评估编辑前后的质量
变化。本仓库公开训练插件、4 种训练模式、相对路径数据 manifest、
validation/泛化测试脚本与 provenance 审计。

> **默认推荐 Field-credit E5 Actor。** Completion-wide E4/E5 是用于研究
> 的对照 checkpoint，已确认存在严重 solution 坍缩，不应当作默认
> 部署模型。

## 开源结构

- `actor/`：GRPO Actor、reward/KL 路由、4-rank trajectory 合并、训练与推理。
- `editor/`：冻结 FLUX.2 Editor 服务与客户端。**本项目不包含 Editor
  训练**，Editor 在全部实验中都是外部冻结服务。
- `judge/`：确定性冻结 E5 Judge 服务和 checkpoint 合同。
- `global/`：Actor、Editor、Judge 之间的 reward、KL、调度和 provenance 合同。
- `configs/training/`：2 个正式 5×291 模式与 2 个 30-step 无 KL 对照。
- `data/`：7,000 行训练、200 行 validation 和六个共 28,270 行测试
  manifest；不分发图像。
- `environment/` 与 `requirements/`：分离、固定版本的 Actor/Judge 和 Editor
  Python 3.12.13 环境。

权重与 SHA-256 manifest 放在
[huggingface.co/RobinY99/MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2)。

## 核心实验合同

正式训练使用 4 张 Actor GPU 和 4 张 Editor/Judge service GPU。每个
optimizer update 由 4 个 rank 各 36 条 trajectory 合并，全局共 144 条。
每轮 291 updates，五轮累计 global step 1,455。所有发布模式都冻结
ViT 和 multimodal aligner。

| 模式 | Reward credit | KL 正则 | 长度 | 用途 |
| --- | --- | --- | --- | --- |
| `field_component_kl002` | 按解析字段 | reasoning 0.02 + rating 0.02 component KL | 5×291 | **推荐正式模式** |
| `completion_global_kl002` | 整个有效 completion | loss-side global KL 0.02，`kl_in_reward=false` | 5×291 | 坍缩研究对照 |
| `field_nokl_30step` | 按字段 | 无 | 30 steps | 短训练对照 |
| `completion_nokl_30step` | 整个 completion | 无 | 30 steps | 短训练对照 |

## 快速开始

完整训练需要 Linux、CUDA 13.0 和 8 张可见 NVIDIA GPU。CPU 机器可以做
发布检查与合同测试，但不等价于完整实验。

```bash
git clone https://github.com/RobinY99/MR-IQA-2.git
cd MR-IQA-2

cp .env.example .env
# 在 .env 中填写本地 Actor、Editor、Judge、图像、cache 和 wheel 路径。

conda env create -f environment/actor-judge.yml
conda env create -f environment/editor.yml
conda run -n mr_iqa_actor_judge \
  python -m pip install -r requirements/actor-judge.txt
conda run -n mr_iqa_editor \
  python -m pip install -r requirements/editor.txt
# 另行取得并校验 FlashAttention wheel 后，按 environment/README.md
# 将它安装到 Actor/Judge 环境。

python -m pip install -r requirements/publish.txt
huggingface-cli download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir checkpoints/qwen3.5-4b
huggingface-cli download RobinY99/MR-IQA-2 \
  --include "judge/source-e5/**" \
  --local-dir checkpoints/mr-iqa-2
huggingface-cli download RobinY99/MR-IQA-2 \
  training_assets/original_score_cache.sqlite \
  --local-dir checkpoints/mr-iqa-2

sha256sum checkpoints/mr-iqa-2/training_assets/original_score_cache.sqlite
# 预期: 7d5410f57f17ff1957e7cbeef951ac01973c0bce97da6f700d61bb222bdd5532

(cd data && sha256sum -c checksums.sha256)
bash scripts/test_release.sh --static
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

将下载的 portable J0 cache 写入本地 `.env`：

```dotenv
ORIGINAL_SCORE_CACHE_PATH=<repository-root>/checkpoints/mr-iqa-2/training_assets/original_score_cache.sqlite
ORIGINAL_SCORE_CACHE_SHA256=7d5410f57f17ff1957e7cbeef951ac01973c0bce97da6f700d61bb222bdd5532
ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT=10073
ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT=10073
ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS=source-e5-judge-step725-original-score
ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA=vf_original_score_cache_e5_judge_e5prompt_portable_v1
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN=0.0
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX=5.0
```

该文件精确为 15,003,648 bytes，包含 10,073 rows / 10,073 samples。J0 的
实测 min/max/mean 为 `0.83 / 4.23 / 3.1357688871239398`；`.env` 中的
`0.0/5.0` 是 Judge 接受区间，不是实测极值。Portable schema 已删除
绝对路径、GT、raw Judge completion、reasoning 字段和图像字节。
初始 Actor 为 Apache-2.0 许可的
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
固定 revision。

Judge 在 `.env` 中有两个有意区分的身份：

```dotenv
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
# 冻结 Judge prompt/cache 协议使用的源语义身份。
JUDGE_MODEL_TREE_SHA256=e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a
# Hub 公开 10-file 导出的可迁移完整性摘要。
JUDGE_MODEL_EXPORT_TREE_SHA256=21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c
```

`JUDGE_MODEL_TREE_SHA256` 是 `provenance.json` 携带的源 full-checkpoint
语义/cache 身份，并不是公开目录的完整性 hash；
`JUDGE_MODEL_EXPORT_TREE_SHA256` 才是对下载的 Hub 10-file snapshot
重新计算的摘要。不要混用这两个值。

冻结 Editor 必须由使用者从
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
获取并固定 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`。本项目不分发其权重；
该固定版本的 4B 模型使用 Apache-2.0，使用者仍需保留上游 notices 与使用说明。

先运行 1 update 端到端 smoke，再启动正式训练：

```bash
# 每次运行都使用新的 ID 和新的大容量存储目录。
export RUN_ID="smoke-field-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB="<host-appropriate-smoke-threshold>"
bash scripts/train.sh --mode field_component_kl002 --smoke

export RUN_ID="field-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB=500
bash scripts/train.sh --mode field_component_kl002
```

不要复用 `RUN_ID` 或已有输出目录。`OUTPUT_ROOT` 和 `VF_STORAGE_ROOT`
必须指向同一个新的、空间充足的目录；正式 5-epoch 运行建议预留至少
500 GiB，因此 `VF_MIN_FREE_GIB=500`。单步 smoke 可在核对宿主机实际空间后
设置更小的 host-specific 阈值。

`WANDB_MODE=offline` 受支持并且是可复现默认值。只有需要远程跟踪并已安全
配置凭据时才设置 `WANDB_MODE=online`；smoke 路径内部会禁用 W&B。

正式 launcher 会串联 5 个 291-update epoch，每轮从上一轮 checkpoint 恢复，
并在轮间执行完整 200 行 validation。
每轮只有在完成 200 行、8-shard Actor→Editor barrier→Judge 的 observational
gate 后才按 `quarantined → technically_valid → promoted` 推进；下一轮只解析
该 promoted manifest，而 `--skip-validation --epochs 1` 会有意留下不能作为
下一轮起点的 unpromoted checkpoint。

## 评测

仅评估 Actor 格式、有效性与 rating PLCC/SRCC/MAE：

```bash
EVAL_ACTOR_ONLY=1 \
ACTOR_MODEL_PATH=<actor-checkpoint> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh test
```

完整评测严格执行 8-shard Actor→完整 Editor barrier→冻结 E5 Judge：

```bash
ACTOR_MODEL_PATH=<actor-checkpoint> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh validation

ACTOR_MODEL_PATH=<actor-checkpoint> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh test
```

`all` 会一次运行 validation 和六个测试集。详见
[`docs/evaluation.md`](docs/evaluation.md)。

## 发布 checkpoint

| 产物 | Step | 公开 10-file export-tree SHA-256 | 源 full checkpoint tree SHA-256 | Validation PLCC / SRCC / MAE | 状态 |
| --- | ---: | --- | --- | --- | --- |
| Source E5 Judge | 725 | `21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c` | `e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a` | 0.947970 / 0.934169 / 0.439320 | 冻结 reward/评测模型 |
| Field Actor E5 | 1,455 | `3e372f548631e3ebbb23e9d8493cb2d50aa482b1941025deda907b35e0a97edb` | `65935012bcaef8c027fb9d233e563c5fea3515e2011e1dd046209b222afe9e94` | 0.935394 / 0.919533 / 0.354589 | **推荐；best/final** |
| Completion Actor E4 | 1,164 | `fcc36656fd15ba7e164bdf0b0be46290ad231636e88664e7bafaa0982ab59c53` | `cc1adae8b748edfbd62bcd8f63c886329769ddc8f226dad23e723018a08e6335` | 0.928128 / 0.913975 / 0.860377 | 坍缩研究对照 |
| Completion Actor E5 | 1,455 | `14d801bffb7f65217a899b10c0735d3d2e37436dd799c3b6352f085845e5b374` | `5a565e49c54c0d6fc52be57aece120c529b23541774f18b7b4d5fb404b082345` | 0.928980 / 0.915821 / 0.997127 | 坍缩研究终点 |

每个 inference-only 模型精确为 9,098,689,558 bytes。使用前需同时校验
Hugging Face 逐文件 checksum 与公开 export-tree digest。源 full-tree digest 只是
公开导出前、含训练状态额外文件的 promotion provenance，无法从 10-file
公开 snapshot 重算。详细结果见
[`docs/checkpoints.md`](docs/checkpoints.md)。

## 局限

- 两个正式实验同时改动了 reward-credit 和 KL 路由，且每个配置只有
  1 个 seed，因而不能将差异归因到单个变量。
- Completion-wide E4/E5 的 rating PLCC/SRCC 仍然较高，但 solution 已坍缩为
  通用 house 编辑。Rating 相关性不能替代 solution diversity/grounding 审计。
- Field mask 在本实验中避免了灾难性 house 坍缩，但仍存在高频词汇骨架。
  Mask 缓解跨字段 credit leakage，不等于语义一致性保证。
- GitHub 不分发图像、冻结 Editor、上游 base weights 和部分 runtime 产物；
  请使用者自行获取并遵守各自许可。
- 初始 Qwen Actor 使用 Apache-2.0；冻结
  [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
  Editor 固定在 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`，
  本仓库不分发；该固定 4B 版本使用 Apache-2.0。

详见 [`docs/reproducibility.md`](docs/reproducibility.md)、
[`docs/privacy.md`](docs/privacy.md) 与 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 许可与引用

本仓库的原创代码使用 MIT License。Qwen 派生的 Actor/Judge 权重使用
Apache-2.0；未分发的 FLUX.2-klein-4B Editor（固定 revision 见上）同样使用
Apache-2.0。数据集、依赖和 Editor 各自受上游条款约束，MIT 不对它们重新许可。请使用
[`CITATION.cff`](CITATION.cff) 引用本软件。
