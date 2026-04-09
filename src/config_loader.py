import os
import yaml
from dotenv import load_dotenv
from typing import Any


class _AttrDict:
    """Recursively wraps a dict so values are accessible as attributes OR dict keys."""
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, _AttrDict(value) if isinstance(value, dict) else value)

    # --- dict-style access (backward compatibility) ---
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return repr(vars(self))


def load_secure_config(config_path: str = "config/prompt_config.yaml") -> "_AttrDict":
    """Load config with API key from environment, returning attribute-accessible object."""
    # Load .env file if it exists
    load_dotenv()

    # Get API key (will raise error if not found)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "   1. Create a .env file in project root\n"
            "   2. Add: GROQ_API_KEY=your_key_here\n"
            "   3. Get your key from: https://console.groq.com/keys"
        )

    # Load YAML config
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    # Inject API key into config
    raw["api_key"] = api_key

    return _AttrDict(raw)