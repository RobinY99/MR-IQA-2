from __future__ import annotations

from typing import Any


def _is_visual_name(name: str) -> bool:
    return ".visual." in f".{name}."


def _remove_input_grad_hooks(module: Any) -> bool:
    had_hooks = bool(getattr(module, "_require_grads_hook", None)) or bool(
        getattr(module, "_require_grads_hooks", None)
    )
    if not had_hooks:
        return False

    disable = getattr(module, "disable_input_require_grads", None)
    if callable(disable):
        disable()

    handles = []
    many = getattr(module, "_require_grads_hooks", None)
    if many:
        handles.extend(many)
    single = getattr(module, "_require_grads_hook", None)
    if single is not None:
        handles.append(single)
    seen = set()
    for handle in handles:
        if id(handle) in seen:
            continue
        seen.add(id(handle))
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()
    if hasattr(module, "_require_grads_hooks"):
        module._require_grads_hooks = []
    if hasattr(module, "_require_grads_hook"):
        del module._require_grads_hook
    return True


def deactivate_frozen_visual_checkpointing(model: Any) -> dict[str, Any]:
    """Undo runtime-only visual GC state after Swift configures a frozen tower."""
    trainable_visual = [
        name
        for name, parameter in model.named_parameters()
        if _is_visual_name(name) and parameter.requires_grad
    ]
    if trainable_visual:
        raise RuntimeError(
            "refusing to deactivate visual GC while visual parameters are trainable: "
            f"{trainable_visual[:16]}"
        )

    visual_modules = []
    disabled_gc = []
    removed_hooks = []
    for name, module in model.named_modules():
        if not _is_visual_name(name):
            continue
        visual_modules.append(name)
        if bool(getattr(module, "gradient_checkpointing", False)):
            disabled_gc.append(name)
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
        if _remove_input_grad_hooks(module):
            removed_hooks.append(name)

    return {
        "visual_modules": visual_modules,
        "disabled_gc": disabled_gc,
        "removed_input_grad_hooks": removed_hooks,
        "trainable_visual": trainable_visual,
    }
