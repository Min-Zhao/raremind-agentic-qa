"""Config loader: reads config.yaml and resolves environment variable placeholders."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Environment variable placeholders in the format ``${VAR_NAME}`` are
    resolved automatically.  If *config_path* is None, the function searches
    for ``config/config.yaml`` relative to the project root.

    Parameters
    ----------
    config_path : str or Path, optional

    Returns
    -------
    dict
    """
    if config_path is None:
        # Walk up directory tree to find config/config.yaml
        search_root = Path(__file__).resolve().parent
        for _ in range(5):
            candidate = search_root / "config" / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
            search_root = search_root.parent

    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(
            f"config.yaml not found. Provide an explicit path or place it at config/config.yaml."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Resolve ${ENV_VAR} placeholders
    resolved = _resolve_env_vars(raw)

    config = yaml.safe_load(resolved)
    return config or {}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(text: str) -> str:
    """Replace ``${VAR}`` patterns with values from the environment."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name, "")
        if not value:
            import warnings
            warnings.warn(
                f"Environment variable '{var_name}' is not set.",
                stacklevel=2,
            )
        return value

    return _ENV_VAR_RE.sub(replacer, text)
