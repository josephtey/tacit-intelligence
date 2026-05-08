"""Name → ProtocolGenerator class registry.

Adding a new generator requires only:
  1. A new file under eval/generators/ with a ProtocolGenerator subclass.
  2. One line here.
"""

from eval.generators.base import ProtocolGenerator
from eval.generators.gemini import GeminiGenerator
from eval.generators.openai import OpenAIGenerator

_REGISTRY: dict[str, type[ProtocolGenerator]] = {
    "gpt-5.5":        OpenAIGenerator,
    "gemini-2.5-pro": GeminiGenerator,
}


def get(name: str, **kwargs) -> ProtocolGenerator:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown generator: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def names() -> list[str]:
    return list(_REGISTRY)
