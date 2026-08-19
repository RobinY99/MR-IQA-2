#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/test_release.sh [--static]

Run deterministic release checks and CPU contract tests. --static omits
pytest and requires only Python and Bash from the standard environment.
EOF
}

while (($#)); do
  case "$1" in
    --static)
      STATIC_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/check_release.py

for mode in \
  completion_global_kl002 \
  field_component_kl002; do
  bash scripts/train.sh --mode "${mode}" --print-plan \
    | python3 -c 'import json, sys; plan = json.load(sys.stdin); assert plan["mode"] == sys.argv[1]' "${mode}"
done
bash scripts/evaluate.sh --print-plan >/dev/null
python3 environment/capture_runtime.py --role test \
  | python3 -c 'import json, sys; manifest = json.load(sys.stdin); assert manifest["schema_version"] == "mr_iqa_2_runtime_manifest_v1"'
echo "Public plan and runtime-manifest commands passed."

if ((STATIC_ONLY)); then
  echo "Static release checks passed."
  exit 0
fi

python3 - <<'PY'
import importlib.util
import sys

required = ("PIL", "pytest", "requests", "torch")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "Missing CPU test dependencies: " + ", ".join(missing) +
        ". Install requirements/test.txt and the official CPU PyTorch wheel.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

python3 -m pytest
