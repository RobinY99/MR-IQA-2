# Privacy and release hygiene

## Never commit

- `.env` or shell history;
- access tokens, SSH material, cloud credentials, or W&B credentials;
- user-specific home, mount, scratch, cache, or cluster paths;
- private hostnames, IP addresses, scheduler job IDs, or service credentials;
- original-score cache databases;
- checkpoint weights, optimizer state, compiled wheels, or generated images;
- raw service logs that contain paths, prompts, or source-image metadata;
- datasets or images without explicit redistribution rights.

Keep real values in ignored `.env` files or a secret manager.

## Before publishing a change

Run:

```bash
bash scripts/test_release.sh --static
git status --short
git diff --check
```

Also inspect the staged diff manually.

## Logs and experiment trackers

Logs may capture commands, environment variables, paths, IDs, and responses.
Sanitize exports before sharing. `WANDB_MODE=offline` is the default.

## Images and generated outputs

Before sharing an evaluation bundle:

1. review the source dataset's terms;
2. remove private paths and service metadata;
3. consider whether images or text identify individuals or locations;
4. publish aggregate metrics when row-level outputs are not necessary;
5. preserve anonymized sample IDs and aggregate provenance so results remain
   auditable.

## Hugging Face publication

Upload only reviewed artifacts. Use repository secrets for authentication,
immutable revisions in reports, and large-file storage for tensors. Preserve
upstream provenance and licenses.

## Incident response

If private information is committed, stop distribution, rotate credentials,
and follow the host platform's sensitive-data removal procedure.
