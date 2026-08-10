# Security policy

## Reporting a vulnerability

Please use GitHub's private security-advisory workflow for vulnerabilities that
could expose credentials, execute untrusted code, bypass checkpoint integrity
checks, or allow unsafe network access. Do not include secrets, private model
URLs, dataset samples, or private infrastructure details in a public issue.

For ordinary bugs, open a public issue with a sanitized example and the output
of `bash scripts/test_release.sh --static`.

## Supported versions

Security fixes target the latest default-branch commit. Review current upstream
advisories before deployment.

## Operational guidance

- Keep `.env`, Hugging Face tokens, W&B credentials, caches, and private paths
  outside version control.
- Run the included artifact preflight before loading models or runtime assets.
- Bind Editor and Judge services to loopback or an authenticated private
  network. The reference services are research components, not hardened public
  endpoints.
- Treat model files, manifests, and images as untrusted; use least privilege.
- Review generated images and logs before sharing them; they can retain
  information from source images or local filesystem paths.
