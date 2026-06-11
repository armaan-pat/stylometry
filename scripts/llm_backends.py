"""Pluggable LLM generation backends for synthetic hard-negative generation.

Every backend implements one method — ``generate_batch(prompts) -> list[str]`` —
and exposes a stable ``.name`` (``"<backend>:<model>"``) that is written to the
``generator`` column of the output dataset.  That column is the hook for the two
diversity workflows synthetic generation is meant to support:

  * diversity audits — how many rows came from each generator, and
  * held-out-generator OOD evaluation — train the detector on impostors from
    models A+B, then measure detection on impostors from model C it never saw.

Because the point of using several LLMs is to keep the detector from overfitting
to one model's fingerprint, you almost always want a *mix* of generators in one
dataset.  ``build_generators(["groq:llama-3.3-70b-versatile", "gemini:gemini-1.5-flash"])``
returns one instance per spec; the caller round-robins jobs across them.

Backends
--------
    hf:<hf-repo-id>             local transformers model (GPU; supports 4-bit)
    groq:<model>                Groq API            (free tier; OpenAI-compatible)
    openrouter:<model>          OpenRouter API      (has a free model pool)
    together:<model>            Together AI API     (cheap)
    deepseek:<model>            DeepSeek API        (very cheap)
    openai:<model>              OpenAI API          (gpt-4o-mini = cheap)
    gemini:<model>              Google Gemini API   (generous free tier)
    ollama:<model>              local Ollama daemon (free; no GPU deps here)

All API backends read their key from an environment variable (e.g. GROQ_API_KEY)
and retry on 429 / 5xx with exponential backoff.  ``requests`` is imported lazily
so an HF-only or Ollama-only run needs no extra dependency; likewise torch /
transformers are imported only inside the HF backend, so an API-only run needs
no GPU stack installed.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

# OpenAI-compatible providers share one client; only base_url + key env differ.
_OPENAI_COMPAT: dict[str, tuple[str, str]] = {
    "groq":       ("https://api.groq.com/openai/v1",   "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",     "OPENROUTER_API_KEY"),
    "together":   ("https://api.together.xyz/v1",      "TOGETHER_API_KEY"),
    "deepseek":   ("https://api.deepseek.com/v1",      "DEEPSEEK_API_KEY"),
    "openai":     ("https://api.openai.com/v1",        "OPENAI_API_KEY"),
}

_API_BACKENDS = frozenset({*_OPENAI_COMPAT, "gemini", "ollama"})
_ALL_BACKENDS = frozenset({"hf", *_API_BACKENDS})


class Generator(Protocol):
    """A text generator. ``name`` is the value written to the generator column."""

    name: str

    def generate_batch(self, prompts: list[str]) -> list[str]:
        ...


# ---------------------------------------------------------------------------
# Spec parsing / factory
# ---------------------------------------------------------------------------

def parse_spec(spec: str) -> tuple[str, str]:
    """Parse a ``"<backend>:<model>"`` spec into (backend, model).

    The model itself may contain colons (e.g. HF revisions, ollama tags), so we
    split only on the first colon.
    """
    if ":" not in spec:
        raise ValueError(
            f"Generator spec '{spec}' must be '<backend>:<model>', e.g. "
            "'groq:llama-3.3-70b-versatile' or 'hf:mistralai/Mistral-7B-Instruct-v0.3'."
        )
    backend, model = spec.split(":", 1)
    backend = backend.strip().lower()
    model = model.strip()
    if backend not in _ALL_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}' in spec '{spec}'. "
            f"Known backends: {', '.join(sorted(_ALL_BACKENDS))}."
        )
    if not model:
        raise ValueError(f"Empty model in spec '{spec}'.")
    return backend, model


def build_generators(
    specs: list[str],
    *,
    load_in_4bit: bool = False,
    temperature: float = 0.85,
    top_p: float = 0.9,
    max_new_tokens: int = 300,
    request_workers: int = 4,
    max_retries: int = 5,
) -> list[Generator]:
    """Instantiate one Generator per spec. Heavy/optional deps load lazily here."""
    gens: list[Generator] = []
    for spec in specs:
        backend, model = parse_spec(spec)
        common = dict(
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
        )
        if backend == "hf":
            gens.append(HFGenerator(model, load_in_4bit=load_in_4bit, **common))
        elif backend in _OPENAI_COMPAT:
            base_url, key_env = _OPENAI_COMPAT[backend]
            gens.append(OpenAICompatGenerator(
                backend, model, base_url=base_url, api_key_env=key_env,
                request_workers=request_workers, max_retries=max_retries, **common,
            ))
        elif backend == "gemini":
            gens.append(GeminiGenerator(
                model, request_workers=request_workers, max_retries=max_retries,
                **common,
            ))
        elif backend == "ollama":
            gens.append(OllamaGenerator(
                model, request_workers=request_workers, max_retries=max_retries,
                **common,
            ))
        else:  # pragma: no cover - guarded by parse_spec
            raise ValueError(f"Unhandled backend '{backend}'.")
    return gens


# ---------------------------------------------------------------------------
# Local transformers backend
# ---------------------------------------------------------------------------

class HFGenerator:
    """Local HuggingFace causal-LM backend (the original generation path)."""

    def __init__(
        self,
        model_name: str,
        *,
        load_in_4bit: bool = False,
        temperature: float = 0.85,
        top_p: float = 0.9,
        max_new_tokens: int = 300,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        # torch.nn.Module.set_submodule was added in PyTorch 2.5; patch older builds.
        if not hasattr(torch.nn.Module, "set_submodule"):
            def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
                atoms = target.split(".")
                parent = self.get_submodule(".".join(atoms[:-1])) if len(atoms) > 1 else self
                setattr(parent, atoms[-1], module)
            torch.nn.Module.set_submodule = _set_submodule

        self.name = f"hf:{model_name}"
        self._torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder-only models MUST left-pad for batched generation: right-padding
        # makes the model attend to pad tokens and corrupts the output, and keeps
        # the uniform output_ids[prompt_len:] slice below correct for every row.
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16 if not load_in_4bit else None,
        )
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        torch = self._torch
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        formatted = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        prompt_len = inputs["input_ids"].shape[1]
        for output_ids in outputs:
            generated_ids = output_ids[prompt_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(text)
        return results


# ---------------------------------------------------------------------------
# API backends
# ---------------------------------------------------------------------------

class _ConcurrentAPIGenerator:
    """Shared concurrency + retry for one-prompt-per-request HTTP backends.

    Subclasses implement ``_generate_one(prompt) -> str``. ``generate_batch``
    fans the batch out across a small thread pool (free tiers tolerate a few
    concurrent requests) while preserving input order.
    """

    name: str

    def __init__(self, *, request_workers: int, max_retries: int) -> None:
        self._workers = max(1, request_workers)
        self._max_retries = max(1, max_retries)
        self._rng = random.Random(0)

    def _generate_one(self, prompt: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _with_retry(self, fn, *args):
        """Call fn with exponential backoff on transient (429/5xx/network) errors."""
        import requests  # lazy

        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn(*args)
            except requests.HTTPError as exc:  # type: ignore[attr-defined]
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status != 429 and status < 500:
                    raise  # client error (bad key, bad model) — don't retry
                last_exc = exc
            except requests.RequestException as exc:  # network/timeout
                last_exc = exc
            sleep_for = delay + self._rng.uniform(0, 0.5)
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)
        # Out of retries: return empty so the row is dropped by the quality gate
        # rather than crashing a long multi-sender run.
        print(f"[{self.name}] giving up after {self._max_retries} retries: {last_exc}")
        return ""

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        if self._workers == 1:
            return [self._with_retry(self._generate_one, p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(lambda p: self._with_retry(self._generate_one, p), prompts))


class OpenAICompatGenerator(_ConcurrentAPIGenerator):
    """Any OpenAI-/chat-completions-compatible endpoint (Groq, OpenRouter, …)."""

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        base_url: str,
        api_key_env: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        request_workers: int,
        max_retries: int,
    ) -> None:
        super().__init__(request_workers=request_workers, max_retries=max_retries)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{backend} backend needs the {api_key_env} environment variable set."
            )
        self.name = f"{backend}:{model}"
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._body_common = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }

    def _generate_one(self, prompt: str) -> str:
        import requests

        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **self._body_common,
        }
        resp = requests.post(self._url, headers=self._headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class GeminiGenerator(_ConcurrentAPIGenerator):
    """Google Gemini (Generative Language API)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        request_workers: int,
        max_retries: int,
    ) -> None:
        super().__init__(request_workers=request_workers, max_retries=max_retries)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "gemini backend needs GEMINI_API_KEY (or GOOGLE_API_KEY) set."
            )
        self.name = f"gemini:{model}"
        self._api_key = api_key
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        self._gen_cfg = {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_new_tokens,
        }

    def _generate_one(self, prompt: str) -> str:
        import requests

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self._gen_cfg,
        }
        resp = requests.post(
            self._url, params={"key": self._api_key}, json=body, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:  # safety block or empty completion → drop downstream
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()


class OllamaGenerator(_ConcurrentAPIGenerator):
    """Local Ollama daemon (default http://localhost:11434). No GPU deps here."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        request_workers: int,
        max_retries: int,
    ) -> None:
        # Ollama serializes on one model anyway; keep concurrency modest.
        super().__init__(request_workers=min(request_workers, 2), max_retries=max_retries)
        self.name = f"ollama:{model}"
        self._model = model
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._url = host + "/api/chat"
        self._options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
        }

    def _generate_one(self, prompt: str) -> str:
        import requests

        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": self._options,
        }
        resp = requests.post(self._url, json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
