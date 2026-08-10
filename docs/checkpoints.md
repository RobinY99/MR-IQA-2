# Checkpoints and released results

Model artifacts are distributed from
[RobinY99/MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2). The public
release contains the recommended Actor, the frozen Judge required by the
training/evaluation contract, and the final completion-wide ablation.

## Released inference bundle

| Artifact ID | Role and mode | Step | Validation PLCC / SRCC / MAE | Release decision |
| --- | --- | ---: | --- | --- |
| `source-e5-judge-step725` | Frozen source E5 Judge | 725 | 0.947970 / 0.934169 / 0.439320 | Required by reward/evaluation contract |
| `actor-field-e5-step1455` | Field credit + component KL 0.02/0.02 | 1,455 | 0.935394 / 0.919533 / 0.354589 | **Recommended Actor; best/final** |
| `actor-completion-e5-step1455` | Completion credit + global KL 0.02 | 1,455 | 0.928980 / 0.915821 / 0.997127 | Diagnostic final; solution collapsed |

The three inference-only model directories contain 30 model files and total
27,296,068,674 bytes. The E4 completion checkpoint is not part of the public
model release. Its validation row remains below only as part of the complete
five-epoch training history.

The completion E5 model is retained to reproduce the negative result. It
produces the same generic house-edit instruction on every row of the
six-dataset evaluation and is not a recommended deployment model.

## Checkpoint promotion

Each formal epoch starts in `quarantined`, advances to `technically_valid`
after trainer, artifact, trajectory, and provenance checks, and becomes
`promoted` only after the complete 200-row, eight-shard
Actor→Editor barrier→Judge validation. The next epoch resolves only a promoted
manifest with `usable=true`.

The public launchers verify the selected model files automatically. Users
normally need only configure `.env` and run:

```bash
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

`--skip-validation` is permitted only with `--epochs 1`; it intentionally
leaves an unpromoted checkpoint that cannot seed a later epoch.

## Portable training asset

| Hub path | Bytes | Rows / samples | Schema |
| --- | ---: | ---: | --- |
| `training_assets/original_score_cache.sqlite` | 15,003,648 | 10,073 / 10,073 | `vf_original_score_cache_e5_judge_e5prompt_portable_v1` |

The portable J0 cache contains relative image paths and deterministic source
scores. It contains no ground-truth scores, image bytes, raw Judge completions,
reasoning fields, credentials, or private absolute paths.

## Five-epoch validation history

Every checkpoint below completed the fixed 200-row validation. `Valid` counts
rows with a valid Actor rating and terminal Editor/Judge result; all 200 source
rows remain present in zero-filled summaries.

### Field credit + field component KL

| Epoch / step | Valid | PLCC | SRCC | MAE | Zero-filled reasoning reward | Normalized solution unique | House-family hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 / 291 | 200 | 0.877854 | 0.883833 | 0.909273 | 0.247956 | 199 | 0/200 |
| E2 / 582 | 197 | 0.916015 | 0.897803 | 0.329187 | 0.273144 | 196 | 0/197 |
| E3 / 873 | 200 | 0.928323 | 0.916390 | 0.722203 | 0.289506 | 199 | 0/200 |
| E4 / 1,164 | 195 | 0.938546 | 0.923312 | 0.339879 | 0.289101 | 195 | 0/195 |
| E5 / 1,455 | 200 | 0.935394 | 0.919533 | 0.354589 | 0.292849 | 199 | 0/200 |

### Completion-wide credit + one loss-side global completion KL

| Epoch / step | Valid | PLCC | SRCC | MAE | Zero-filled reasoning reward | Normalized solution unique | House-family hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 / 291 | 197 | 0.867164 | 0.878563 | 0.819417 | 0.451133 | 181 | 197/197 |
| E2 / 582 | 199 | 0.914423 | 0.899606 | 0.778155 | 0.461206 | 1 | 200/200 |
| E3 / 873 | 197 | 0.925644 | 0.909860 | 0.863673 | 0.451693 | 1 | 197/197 |
| E4 / 1,164 | 200 | 0.928128 | 0.913975 | 0.860377 | 0.462244 | 2 | 200/200 |
| E5 / 1,455 | 200 | 0.928980 | 0.915821 | 0.997127 | 0.462215 | 1 | 200/200 |

No Editor/Judge service errors occurred in these ten validations. Lower valid
counts are Actor-ineligible rows, not deleted source rows.

## Six-dataset Actor generalization

Both released Actors completed all 28,270 source rows. Metrics use valid Actor
ratings for their denominator.

### Field E5, recommended

| Dataset | Valid / rows | PLCC | SRCC | MAE |
| --- | ---: | ---: | ---: | ---: |
| AGIQA-3K | 2,951 / 2,982 | 0.808912 | 0.739095 | 0.601462 |
| CSIQ | 865 / 866 | 0.824399 | 0.785005 | 0.587842 |
| KADID-10K | 9,951 / 10,125 | 0.667023 | 0.668921 | 0.975945 |
| KonIQ-10K | 2,004 / 2,010 | 0.936707 | 0.917184 | 0.353084 |
| LIVE-W | 1,162 / 1,162 | 0.893301 | 0.863127 | 0.373541 |
| SPAQ | 11,112 / 11,125 | 0.899730 | 0.899407 | 0.366480 |

### Completion E5, diagnostic final

| Dataset | Valid / rows | PLCC | SRCC | MAE |
| --- | ---: | ---: | ---: | ---: |
| AGIQA-3K | 2,975 / 2,982 | 0.808726 | 0.735765 | 0.807246 |
| CSIQ | 866 / 866 | 0.814805 | 0.786209 | 1.522288 |
| KADID-10K | 10,125 / 10,125 | 0.675806 | 0.684116 | 0.771692 |
| KonIQ-10K | 2,010 / 2,010 | 0.931863 | 0.913624 | 0.998804 |
| LIVE-W | 1,162 / 1,162 | 0.877920 | 0.857044 | 1.186036 |
| SPAQ | 11,125 / 11,125 | 0.896005 | 0.896708 | 1.038827 |

## Editor/Judge generalization and solution health

| Actor | Success / rows | Pooled `J0` | Pooled `J1` | Delta | Zero-filled reasoning reward | Normalized solution unique | Modal / rows | Semantic house family |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Field E5 | 28,044 / 28,270 | 2.851875 | 3.922885 | 1.071010 | 0.401480 | 23,457 | 51 / 28,044 | 0 / 28,044 |
| Completion E5 | 28,270 / 28,270 | 2.854820 | 4.285586 | 1.430766 | 0.536715 | 1 | 28,270 / 28,270 | 28,270 / 28,270 |

Completion E5 applies one identical solution to different source images. The
edited images remain different, but that does not demonstrate image-conditioned
solution reasoning.

Field E5 does not exhibit the house-family or whole-sentence collapse. It is
not template-free: a `super-resolution` + `super-smooth` lexical skeleton
appears in 28,022 of 28,044 eligible solutions (99.9216%). Field masking
mitigates catastrophic cross-field collapse but does not guarantee diverse or
semantically optimal edits.

## Collapse milestones

The primary semantic detector requires a dwelling term and a collapse anchor
such as urban/suburban, luxury/wealth, landscaping, or gardens. Normalized
solutions use Unicode NFKC, case folding, article removal, punctuation
normalization, and whitespace collapse.

| Global step | Epoch / local step | Event | Count |
| ---: | --- | --- | ---: |
| 236 | E1 / 236 | Semantic house family first reaches at least 50% | 84 / 144 (58.3333%) |
| 265 | E1 / 265 | Semantic house family first reaches at least 90% | 130 / 144 (90.2778%) |
| 272 | E1 / 272 | Semantic house family first reaches at least 95% | 138 / 144 (95.8333%) |
| 520 | E2 / 229 | One normalized solution first reaches at least 50% | 77 / 144 (53.4722%) |
| 551 | E2 / 260 | One normalized solution first reaches at least 90% | 131 / 144 (90.9722%) |
| 552 | E2 / 261 | One normalized solution first reaches at least 95% | 138 / 144 (95.8333%) |
| 973 | E4 / 100 | One normalized solution begins its uninterrupted at-least-90% period through E5 | audited series |

Semantic house collapse therefore precedes exact lexical collapse. The
completion-wide run broadcasts a high-yield solution reward across evidence,
rating, format, and solution tokens; its global completion KL does not
specifically preserve solution grounding. The field run localizes rewards and
KL by field and avoids the catastrophic house solution in this comparison.

This is descriptive evidence from a joint intervention with one seed per
mode. A causal follow-up requires the complete credit-scope × KL-scope design
with multiple seeds.
