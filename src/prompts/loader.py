from pathlib import Path


_PROMPT_DIRECTORY = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template and reject paths outside this directory."""
    path = (_PROMPT_DIRECTORY / name).resolve()
    if path.parent != _PROMPT_DIRECTORY.resolve() or path.suffix != ".txt":
        raise ValueError("Prompt name must refer to a .txt file inside prompts")
    return path.read_text(encoding="utf-8")
