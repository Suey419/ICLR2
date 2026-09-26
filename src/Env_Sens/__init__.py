"""Environment-sensitivity experiment helpers.

The experiment deliberately keeps provider credentials separate from the
normal LEAP run.  ``configure_provider_environment`` maps the dedicated
Env_Sens key/base URL to both the diagnostic (Astra) and simulation (Luna)
provider slots without changing their model names.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: str | Path | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def configure_provider_environment(env_file: str | Path | None = None) -> None:
    """Route both Astra and Luna through the dedicated Env_Sens endpoint.

    Models remain independently selectable through ``DOCTOR_MODEL`` and
    ``PROVIDERS_MODEL``; only credentials and endpoint are replaced.
    """
    _load_dotenv(env_file)
    key = os.getenv("Env_Sens_API_KEY")
    base_url = os.getenv("Env_Sens_BASE_URL")
    if not key or not base_url:
        raise RuntimeError("English textrequires Env_Sens_API_KEY and Env_Sens_BASE_URL")
    for name in ("DOCTOR_API_KEY", "DOCTOR_API_KEY1", "DOCTOR_API_KEY2", "PROVIDERS_API_KEY"):
        os.environ[name] = key
    for name in ("DOCTOR_API_KEY_BASE_URL", "PROVIDERS_API_KEY_BASE_URL"):
        os.environ[name] = base_url.rstrip("/")


__all__ = ["configure_provider_environment"]
