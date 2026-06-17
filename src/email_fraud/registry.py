"""Central component registry.

All components (encoders, losses, heads, datasets) self-register via @register;
the trainer resolves them by name from YAML without any if/elif dispatch.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Global registry: {kind: {name: class}}
# Populated at import time by @register decorators on concrete classes.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict[str, type]] = {
    "encoder": {},
    "loss": {},
    "head": {},
    "dataset": {},
}

# Maps each kind to its ABC's fully-qualified import path.
# Lazily imported via _get_base() to avoid circular imports:
# registry.py imports nothing from the package; component modules import registry.py.
_KIND_BASES: dict[str, str] = {
    "encoder": "email_fraud.encoders.base.BaseEncoder",
    "loss": "email_fraud.losses.base.BaseLoss",
    "head": "email_fraud.heads.base.BaseHead",
    "dataset": "email_fraud.data.base.BaseDataset",
}


def _get_base(kind: str) -> type:
    """Import and return the ABC for *kind* on first call."""
    import importlib

    module_path, class_name = _KIND_BASES[kind].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def register(kind: str, name: str):
    """Class decorator that inserts *cls* into REGISTRY[kind][name].

    Validates that *cls* is a subclass of the canonical ABC for *kind*.
    Raises ValueError if *kind* is unknown or the subclass check fails.

    Usage::

        @register("encoder", "hf")
        class HFEncoder(BaseEncoder):
            ...
    """
    if kind not in REGISTRY:
        raise ValueError(
            f"Unknown registry kind '{kind}'. "
            f"Valid kinds: {list(REGISTRY.keys())}"
        )

    def decorator(cls: type) -> type:
        base = _get_base(kind)
        if not issubclass(cls, base):
            raise TypeError(
                f"Cannot register '{cls.__name__}' under kind '{kind}': "
                f"must be a subclass of {base.__name__}."
            )
        if name in REGISTRY[kind]:
            existing = REGISTRY[kind][name]
            if existing is not cls:
                raise ValueError(
                    f"Name '{name}' is already registered under '{kind}' "
                    f"by {existing.__name__}. Choose a different name."
                )
        REGISTRY[kind][name] = cls
        return cls

    return decorator


def resolve(kind: str, name: str) -> type:
    """Return the class registered as *name* under *kind*.

    Raises a descriptive KeyError listing available names when not found.
    """
    if kind not in REGISTRY:
        raise KeyError(
            f"Unknown registry kind '{kind}'. "
            f"Valid kinds: {list(REGISTRY.keys())}"
        )
    if name not in REGISTRY[kind]:
        available = list(REGISTRY[kind].keys())
        raise KeyError(
            f"No '{kind}' registered under name '{name}'. "
            f"Available: {available}"
        )
    return REGISTRY[kind][name]


def list_components() -> dict[str, list[str]]:
    """Return a snapshot of all registered component names, grouped by kind."""
    return {kind: list(names.keys()) for kind, names in REGISTRY.items()}
