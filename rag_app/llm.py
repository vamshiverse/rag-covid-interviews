"""Thin, provider-agnostic LLM client (optional).

Only instantiated when RAG_LLM_PROVIDER is 'openai' or 'anthropic'. The rest of
the system never imports openai/anthropic unless one of these is selected, so
the default keyless 'extractive' path has zero hosted dependencies.
"""
from __future__ import annotations

from . import config


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.OPENAI_CHAT_MODEL
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = config.ANTHROPIC_CHAT_MODEL
        else:
            raise ValueError(f"LLMClient does not support provider '{self.provider}'")

    def complete(self, system: str, user: str, temperature: float = 0.0,
                 max_tokens: int = 800) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return resp.choices[0].message.content.strip()
        # anthropic
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(block.text for block in resp.content
                       if getattr(block, "type", "") == "text").strip()


def llm_available(provider: str | None = None) -> bool:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "openai":
        return bool(config.OPENAI_API_KEY)
    if provider == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    return False
