# Checkpoints, provenance, and released results

The Hugging Face repository is the distribution point for model artifacts:
[RobinY99/MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2). Download from an
immutable revision and treat its release manifest as the authority for file
paths and per-file hashes. Two tree digests are recorded for each release:

- the **source full checkpoint tree** identifies the promoted training
  checkpoint, including training-state extras that are not published;
- the **public export tree** identifies exactly the ten allowlisted inference
  files distributed on Hugging Face.

These hashes are intentionally different. A public snapshot must never be
labeled with the source full-tree hash.

## Runtime promotion identity

The formal runtime uses `selected_inference_export` as its stable checkpoint
identity. It hashes exactly these ten regular, non-empty files:

1. `model-00001-of-00002.safetensors`
2. `model-00002-of-00002.safetensors`
3. `model.safetensors.index.json`
4. `config.json`
5. `generation_config.json`
6. `preprocessor_config.json`
7. `processor_config.json`
8. `tokenizer.json`
9. `tokenizer_config.json`
10. `chat_template.jinja`

The canonical algorithm is:

```text
sha256(sorted(relative_path + NUL + decimal_size + NUL + file_sha256 + LF))
over the 10 allowlisted inference-export files
```

The resulting identity object records `semantics=selected_inference_export`,
the algorithm string above, the resulting SHA-256, and `file_count=10`.
Optimizer shards, trainer state, RNG state, caches, logs, and temporary files
are intentionally excluded. They can be required for full-state training
resumption without being suitable as a stable inference identity.

This selected-export digest is not the source full-tree hash in the release
tables below. The full-tree hash covers the promoted source checkpoint and its
additional training-state files for lineage/audit purposes; it can change when
an excluded training artifact changes even if the ten-file inference model is
identical. Validation and per-epoch promotion use the selected-export digest,
while publication records both identities.

## Per-epoch promotion state

Every formal epoch creates
`state/checkpoints/epochN.json` (`vf_checkpoint_manifest_v2`) as
`quarantined`, advances it to `technically_valid` only after trainer/artifact/
trajectory/provenance checks, and promotes it only after a complete 200-row,
eight-shard Actor→Editor barrier→Judge observational validation. The flattened
gate evidence is written to `state/checkpoints/epochN.validation.json`.
`state/epochN.json` uses schema `mr_iqa_2_epoch_chain_v2` and links the epoch,
manifest, final status, validation paths, and stable checkpoint digest.

The next epoch is assigned a parent only by resolving a manifest whose status
is `promoted` and `usable=true`; resolution re-hashes the ten-file identity.
`--skip-validation` is legal only together with `--epochs 1`. It leaves the
checkpoint `technically_valid`, unpromoted, and ineligible to seed a later
epoch.

## Minimal inference bundle

| Artifact ID | Role and mode | Step | Bytes | Public 10-file export-tree SHA-256 | Source full checkpoint tree SHA-256 | Validation PLCC / SRCC / MAE | Release decision |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `source-e5-judge-step725` | Frozen source E5 Judge | 725 | 9,098,689,558 | `21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c` | `e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a` | 0.947970 / 0.934169 / 0.439320 | Required by reward/evaluation contract |
| `actor-field-e5-step1455` | Field credit + component KL 0.02/0.02 | 1,455 | 9,098,689,558 | `3e372f548631e3ebbb23e9d8493cb2d50aa482b1941025deda907b35e0a97edb` | `65935012bcaef8c027fb9d233e563c5fea3515e2011e1dd046209b222afe9e94` | 0.935394 / 0.919533 / 0.354589 | **Recommended Actor; best/final** |
| `actor-completion-e4-step1164` | Completion credit + global KL 0.02 | 1,164 | 9,098,689,558 | `fcc36656fd15ba7e164bdf0b0be46290ad231636e88664e7bafaa0982ab59c53` | `cc1adae8b748edfbd62bcd8f63c886329769ddc8f226dad23e723018a08e6335` | 0.928128 / 0.913975 / 0.860377 | Diagnostic best; solution collapsed |
| `actor-completion-e5-step1455` | Completion credit + global KL 0.02 | 1,455 | 9,098,689,558 | `14d801bffb7f65217a899b10c0735d3d2e37436dd799c3b6352f085845e5b374` | `5a565e49c54c0d6fc52be57aece120c529b23541774f18b7b4d5fb404b082345` | 0.928980 / 0.915821 / 0.997127 | Diagnostic final; solution collapsed |

The four inference-only model directories total exactly 36,394,758,232 bytes.
This does not include training optimizer state, source images, the frozen
Editor, the original-score cache, or runtime environments.

The completion models are released to reproduce a negative result. Do not
select E4 merely because it is the completion run's internal `best`: both E4
and E5 produce a generic house-edit solution on every row of the six-dataset
test. The recommended public Actor is field E5.

## Portable training asset: original-image J0 cache

| Hub path | Bytes | SHA-256 | Rows / samples | Schema |
| --- | ---: | --- | ---: | --- |
| `training_assets/original_score_cache.sqlite` | 15,003,648 | `7d5410f57f17ff1957e7cbeef951ac01973c0bce97da6f700d61bb222bdd5532` | 10,073 / 10,073 | `vf_original_score_cache_e5_judge_e5prompt_portable_v1` |

Observed J0 ratings have minimum `0.83`, maximum `4.23`, and mean
`3.1357688871239398`. The release cache intentionally contains no absolute
filesystem paths, ground-truth scores, image bytes, raw Judge completions, or
Judge reasoning fields. This omission prevents the cache from being used as a
hidden ground-truth label source and removes private-path provenance while
preserving deterministic source-image Judge lookups.

## Dual verification for Hugging Face exports

Before publication, `source_checkpoint_tree_sha256` is retained as the
signed-off promotion provenance recorded for the full private training
checkpoint; the reduced exporter does not claim to recompute that hash. The
exporter selects exactly the ten public inference files, verifies every
relative-path/file SHA-256 entry, and computes `expected_export_tree_sha256`
over that allowlist. Training-state extras are reported but never copied.

After download, a user can verify the per-file checksums and public export-tree
digest. The source full-tree digest remains immutable promotion provenance in
the manifest, but cannot be recomputed from the reduced public snapshot. See
the Hugging Face `checkpoint_manifest.json`, `SHA256SUMS`, and
`SHA256SUMS.full`; all three must refer to the same immutable Hub revision.

The source E5 Judge maps these two checks to distinct runtime variables:

```dotenv
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
JUDGE_MODEL_TREE_SHA256=e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a
JUDGE_MODEL_EXPORT_TREE_SHA256=21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c
```

`JUDGE_MODEL_TREE_SHA256` preserves the source semantic/cache identity from
`provenance.json`; `JUDGE_MODEL_EXPORT_TREE_SHA256` verifies the downloaded
ten-file directory. Do not replace the former with the latter even though the
runtime model path points at the public export.

## Five-epoch validation history

Every checkpoint below completed the fixed 200-row validation and was promoted
as usable. `valid` is the count with a valid Actor rating and Editor/Judge
result; all 200 source rows were retained in zero-filled summaries.

### Field credit + field component KL

| Epoch / step | Source full checkpoint tree SHA-256 | Valid | PLCC | SRCC | MAE | `J0` | `J1` | Delta | Zero-filled reasoning reward | Normalized solution unique | House-family hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 / 291 | `f65e3ead6f9265a919d5dc9b28403f66a5acfd10f775d47c8bb38af548f7db00` | 200 | 0.877854 | 0.883833 | 0.909273 | 3.170300 | 3.856100 | 0.685800 | 0.247956 | 199 | 0/200 |
| E2 / 582 | `f2b109c3829793ff4bef3e8047c3de6e3296fa8616b3b5ccf09abbe2f89db045` | 197 | 0.916015 | 0.897803 | 0.329187 | 3.177360 | 3.941980 | 0.764619 | 0.273144 | 196 | 0/197 |
| E3 / 873 | `fd6577f7d010d2dfbe0e24ed13eea34aa7434cb1837969d3d020432db568d75d` | 200 | 0.928323 | 0.916390 | 0.722203 | 3.170300 | 3.963550 | 0.793250 | 0.289506 | 199 | 0/200 |
| E4 / 1,164 | `502dc73660a3f45df82b70939dae09436dc319d3ed2dbe737554ac5b40388adb` | 195 | 0.938546 | 0.923312 | 0.339879 | 3.173179 | 3.985026 | 0.811846 | 0.289101 | 195 | 0/195 |
| E5 / 1,455 | `65935012bcaef8c027fb9d233e563c5fea3515e2011e1dd046209b222afe9e94` | 200 | 0.935394 | 0.919533 | 0.354589 | 3.170300 | 3.975400 | 0.805100 | 0.292849 | 199 | 0/200 |

### Completion-wide credit + one loss-side global completion KL

| Epoch / step | Source full checkpoint tree SHA-256 | Valid | PLCC | SRCC | MAE | `J0` | `J1` | Delta | Zero-filled reasoning reward | Normalized solution unique | House-family hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 / 291 | `45b0c060a77ebd14e8fceb42a41ab54b1e9943b458335b509eec05e70d6443cd` | 197 | 0.867164 | 0.878563 | 0.819417 | 3.160914 | 4.334010 | 1.173096 | 0.451133 | 181 | 197/197 |
| E2 / 582 | `242da9746d0f3ff051288b05d17beefb8b4e6741c703f29f11684bc575081523` | 199 | 0.914423 | 0.899606 | 0.778155 | 3.170300 | 4.350900 | 1.180600 | 0.461206 | 1 | 200/200 |
| E3 / 873 | `e27238b17b1f9e9387d697b3ee0885c0f0d2cf14bd520b2abba32d8025861d2d` | 197 | 0.925644 | 0.909860 | 0.863673 | 3.182284 | 4.356904 | 1.174619 | 0.451693 | 1 | 197/197 |
| E4 / 1,164 | `cc1adae8b748edfbd62bcd8f63c886329769ddc8f226dad23e723018a08e6335` | 200 | 0.928128 | 0.913975 | 0.860377 | 3.170300 | 4.354500 | 1.184200 | 0.462244 | 2 | 200/200 |
| E5 / 1,455 | `5a565e49c54c0d6fc52be57aece120c529b23541774f18b7b4d5fb404b082345` | 200 | 0.928980 | 0.915821 | 0.997127 | 3.170300 | 4.354000 | 1.183700 | 0.462215 | 1 | 200/200 |

No Editor/Judge service errors occurred in these ten validations. Lower `valid`
counts are Actor-ineligible rows, not deleted source rows.

## Six-dataset Actor generalization

Field E5, completion E4, and completion E5 each completed all 28,270 source
rows. The following metrics use valid Actor ratings for their denominator.

### Field E5, recommended

| Dataset | Valid / rows | PLCC | SRCC | MAE |
| --- | ---: | ---: | ---: | ---: |
| AGIQA-3K | 2,951 / 2,982 | 0.808912 | 0.739095 | 0.601462 |
| CSIQ | 865 / 866 | 0.824399 | 0.785005 | 0.587842 |
| KADID-10K | 9,951 / 10,125 | 0.667023 | 0.668921 | 0.975945 |
| KonIQ-10K | 2,004 / 2,010 | 0.936707 | 0.917184 | 0.353084 |
| LIVE-W | 1,162 / 1,162 | 0.893301 | 0.863127 | 0.373541 |
| SPAQ | 11,112 / 11,125 | 0.899730 | 0.899407 | 0.366480 |

### Completion E4, diagnostic

| Dataset | Valid / rows | PLCC | SRCC | MAE |
| --- | ---: | ---: | ---: | ---: |
| AGIQA-3K | 2,975 / 2,982 | 0.807828 | 0.733838 | 0.694116 |
| CSIQ | 866 / 866 | 0.813875 | 0.784929 | 1.413810 |
| KADID-10K | 10,125 / 10,125 | 0.667236 | 0.679351 | 0.731471 |
| KonIQ-10K | 2,010 / 2,010 | 0.930945 | 0.912822 | 0.866004 |
| LIVE-W | 1,162 / 1,162 | 0.872006 | 0.854123 | 1.096796 |
| SPAQ | 11,125 / 11,125 | 0.895429 | 0.896970 | 0.955335 |

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
| Completion E4 | 28,270 / 28,270 | 2.854820 | 4.285472 | 1.430652 | 0.536632 | 4 | 27,866 / 28,270 | 28,270 / 28,270 |
| Completion E5 | 28,270 / 28,270 | 2.854820 | 4.285586 | 1.430766 | 0.536715 | 1 | 28,270 / 28,270 | 28,270 / 28,270 |

All 28,270 edited-image SHA-256 digests were unique for both completion E4 and
E5 because the same solution was applied to different source images. The high
Judge delta therefore measures a real fixed intervention across inputs, but it
does not show image-conditioned solution reasoning.

Field E5 does not exhibit the house-family or whole-sentence collapse. It is
not template-free: a `super-resolution` + `super-smooth` lexical skeleton
appears in 28,022 of 28,044 eligible generalization solutions (99.9216%). Field
masking mitigates catastrophic cross-field collapse but does not guarantee
diverse or semantically optimal edits.

## Exact collapse milestones

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

The E1 completion checkpoint has 197/197 eligible house-family solutions, but
still 181 normalized variants. By E2 it has 200/200 copies of one normalized
solution. Semantic collapse therefore precedes exact lexical collapse.

## Interpretation

The completion-wide run broadcasts the reward of a high-yield solution across
evidence, rating, format, and solution tokens. Its global completion KL
regularizes the aggregate completion and does not specifically preserve
solution grounding. The Field run localizes rewards and KL by field and avoids
the catastrophic house solution in this comparison.

This is descriptive evidence from a joint intervention: the formal comparison
changes credit routing and KL routing together and has one seed per mode. It
does not establish the mask or KL scope as the sole causal factor. A causal
follow-up requires the complete 2×2 credit-scope × KL-scope design with multiple
seeds.
