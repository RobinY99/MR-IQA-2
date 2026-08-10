from __future__ import annotations

import hashlib


EDITOR_PROMPT_VERSION = "vf_solution_only_v1_20260729"
EDITOR_SEMANTIC_GUARDRAIL = ""
EDITOR_PROMPT_TEMPLATE = "{solution}"
EDITOR_PROMPT_TEMPLATE_HASH = hashlib.sha256(
    EDITOR_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


def build_editor_prompt(evidence: str, solution: str) -> str:
    solution = str(solution or "").strip()
    if not solution:
        raise ValueError("Editor prompt requires nonempty solution")
    return EDITOR_PROMPT_TEMPLATE.format(solution=solution)
