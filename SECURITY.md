# Security policy

## Reporting a vulnerability

Please use GitHub's private security-advisory workflow for vulnerabilities that
could expose credentials, execute untrusted code, bypass checkpoint integrity
checks, or allow unsafe network access. Do not include secrets, private model
URLs, dataset samples, or private infrastructure details in a public issue.

For ordinary bugs and reproducibility questions, open a public issue with a
minimal sanitized example and the output of `bash scripts/test_release.sh
--static`.

## Supported versions

Security fixes are applied to the latest commit on the default branch. Pinned
research dependencies reproduce a specific environment but may later receive
upstream security advisories. Before deployment, review the current advisories
for every dependency and rebuild in an isolated environment.

## Operational guidance

- Keep `.env`, Hugging Face tokens, W&B credentials, caches, and private paths
  outside version control.
- Run the included artifact preflight before loading models or runtime assets.
- Bind Editor and Judge services to loopback or an authenticated private
  network. The reference services are research components, not hardened public
  endpoints.
- Treat model files, JSONL manifests, and image inputs as untrusted data. Run
  them with least privilege and without access to unrelated secrets.
- Review generated images and logs before sharing them; they can retain
  information from source images or local filesystem paths.

MR-IQA-2 is research software and is provided without warranty under the MIT
License.
