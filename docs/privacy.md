# Privacy and release hygiene

The public tree was organized as a clean release rather than publishing an
experiment worktree. Checkpoints and large runtime artifacts are separated from
GitHub source, and machine configuration is represented by placeholders.

## Never commit

- `.env` or shell history;
- access tokens, SSH material, cloud credentials, or W&B credentials;
- user-specific home, mount, scratch, cache, or cluster paths;
- private hostnames, IP addresses, scheduler job IDs, or service credentials;
- original-score cache databases;
- checkpoint weights, optimizer state, compiled wheels, or generated images;
- raw service logs that contain paths, prompts, or source-image metadata;
- datasets or images without explicit redistribution rights.

Use `.env.example` as the public variable schema. Keep real values in an
ignored `.env` or a cluster secret manager.

## Before publishing a change

Run:

```bash
bash scripts/test_release.sh --static
git status --short
git diff --check
```

The static checker searches text for common credential forms, private path
patterns, and private network addresses. It also rejects common checkpoint,
database, wheel, and model-file suffixes from the GitHub tree. This is a safety
net, not proof that a release is private-data-free; manually inspect the staged
diff and generated documentation.

## Logs and experiment trackers

W&B and local logs may capture command lines, environment variables, output
directories, sample IDs, and model responses. Use a dedicated project, review
tracker visibility, and sanitize an exported report before sharing it. The
public scientific record should retain configuration and content digests, not
secret-bearing absolute locations.

`WANDB_MODE=offline` is the supported reproducible default. Online tracking is
optional and should be enabled only when credentials are provided through a
private environment or secret manager. The smoke launcher disables W&B
internally, so a smoke test cannot accidentally create a remote run.

## Images and generated outputs

An edited image can preserve identifying content from its source even when the
edit prompt is generic. A model's evidence text can also describe people,
places, signs, or other sensitive details. Dataset access permission does not
automatically grant permission to redistribute generated derivatives or raw
completions.

Before sharing an evaluation bundle:

1. review the source dataset's terms;
2. remove private paths and service metadata;
3. consider whether images or text identify individuals or locations;
4. publish aggregate metrics when row-level outputs are not necessary;
5. preserve hashes and anonymized sample IDs so aggregate results remain
   auditable.

## Hugging Face publication

Upload only artifacts listed in the reviewed checkpoint manifest. Use
repository secrets for authentication, an immutable revision for reports, and
large-file storage for model tensors. A model card must state upstream model
provenance, intended research use, known collapse behavior, and applicable
licenses. The GitHub MIT License does not supersede an upstream weight license.

The published J0 cache is rebuilt as a portable artifact rather than copied
from a private runtime database. Its portable schema removes absolute paths,
ground-truth values, image bytes, raw Judge completions, and Judge reasoning
fields. Release review verifies its 10,073 rows/samples, 15,003,648-byte size,
and SHA-256 `7d5410f57f17ff1957e7cbeef951ac01973c0bce97da6f700d61bb222bdd5532`.

## Incident response

If private information is committed, stop distribution, rotate exposed
credentials, remove the artifact from the current release, and follow the host
platform's sensitive-data removal procedure. A later deletion commit alone may
not remove data from Git history, forks, caches, or already downloaded model
revisions.
