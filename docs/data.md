# Data manifests

The repository distributes metadata manifests, not dataset images. All image
locations are relative paths so one release can be used with independently
obtained datasets on different machines.

## Inventory

| File | Split | Rows | SHA-256 |
| --- | --- | ---: | --- |
| `train.jsonl` | Training mixture | 7,000 | `1d624d869721331b9831748c2d60833714ff31ae3c8496a176d5d5c37456b8fb` |
| `validation.jsonl` | Fixed validation | 200 | `c48e0f3ea7de63127a10bcd3b898d194a634f2bef6a9c22139d1d42102966f54` |
| `koniq.jsonl` | KonIQ-10K test | 2,010 | `d0a8f0369f2b1f6477587a00f39e738c628fd9dc33766a89fec2262afd33e5ae` |
| `spaq_full.jsonl` | SPAQ test | 11,125 | `31bd47569f8f06687bc34dfb18af91b6f8ed28742605571ae1f2b838fb4e08ae` |
| `livew.jsonl` | LIVE-W test | 1,162 | `63fc4cca655f9461e5e44ebc9afea4fbc98fbcf969fa325f58fbbdac9c778517` |
| `kadid_full.jsonl` | KADID-10K test | 10,125 | `67d0aa59d5d9f0d64434fc57e4aba8c3612c7996e4201d365853fc0d146d0a02` |
| `agiqa3k.jsonl` | AGIQA-3K test | 2,982 | `c9111eaa15b7a1a9cb5be939d5390803371c52c1d538ce36e0089a61eff7e0b3` |
| `csiq.jsonl` | CSIQ test | 866 | `164ecbc68c1e4051be0a95f519444602bae392daf28192ede75f87d519e5c57a` |

The six test manifests total exactly 28,270 rows. Validation is separate, so
`scripts/evaluate.sh all` processes 28,470 rows.

Verify from inside `data/` because the checksum file contains basenames:

```bash
(cd data && sha256sum -c checksums.sha256)
```

## Schemas

Training rows contain the minimal fields needed to reconstruct the public
Actor prompt:

```text
images, target_mean, target_std, sample_id, dataset_name, source_image
```

`actor/scripts/prepare_phase_a_dataset.py` resolves `images` against
`TRAIN_IMAGE_ROOT` and creates the structured evidence/solution/rating request.

Validation rows contain:

```text
id, image, gt_score, normalized_score, std, std_norm
```

Test rows share:

```text
id, dataset, image, source_image, normalized_score, gt_score_norm,
source_score, gt_score, std, std_norm
```

Some score aliases are retained so dataset-specific source values and the
normalized comparison value remain auditable. Evaluation uses the normalized
quality target declared by its parser contract.

## Image layout

Set `TRAIN_IMAGE_ROOT` and `EVAL_IMAGE_ROOT` in `.env`. The path stored in a row
is joined to the corresponding root. Release checks reject absolute image paths
and parent-directory traversal.

Datasets do not necessarily share the same folder layout. Construct a single
evaluation root whose subpaths match the committed `image` fields, or create a
read-only view with the same relative layout. Do not edit a manifest merely to
encode a private machine path.

## Redistribution and ethics

The JSONL manifests do not grant rights to the source images. Obtain each
dataset from its official distributor, review its research/commercial-use
terms, and comply with attribution, privacy, and deletion requirements. Do not
publish edited images or raw model outputs without considering whether the
source content includes people, private places, or other sensitive material.

When creating a new manifest:

1. confirm that redistribution of its metadata is permitted;
2. use stable, non-identifying sample IDs where possible;
3. keep image paths relative;
4. record score normalization and split construction;
5. update `checksums.sha256`;
6. run `bash scripts/test_release.sh --static`.
