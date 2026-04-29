"""Guardrails de segurança para input e output do agente WeatherOps.

Todas as verificações são regex-based (sem chamadas a LLM) para garantir
latência desprezível no caminho crítico de cada requisição.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    passed: bool
    threat_type: str | None  # "prompt_injection" | "exfiltration" | "pii" | "system_leak" | "api_key_leak"
    reason: str


# ---------------------------------------------------------------------------
# Padrões compilados — Input
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(ignore|forget|disregard|override)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(you are now|act as|pretend (you are|to be)|from now on you)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(new\s+system\s+prompt|repeat\s+your\s+(instructions?|system\s+prompt|prompt))",
        re.IGNORECASE,
    ),
]

_EXFILTRATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(google_api_key|openai_api_key|secret|password|token|credential)\s*(=|:|\s)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(print|show|reveal|expose|leak|dump|output|repeat|return)\s+.{0,40}(api.?key|secret|password|token|env(ironment)?\s*var)",
        re.IGNORECASE,
    ),
]

_PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),  # CPF
    re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),  # e-mail
    re.compile(r"(?:\+55\s?)?\(?\d{2}\)?\s?\d{4,5}[-\s]?\d{4}\b"),  # telefone BR
]

# ---------------------------------------------------------------------------
# Padrões compilados — Output
# ---------------------------------------------------------------------------

# Frases distintas do system prompt — verificadas sem importar service.py
_SYSTEM_LEAK_LITERALS = [
    "Você é um assistente meteorológico",
    "Regras para escolha de tool",
    "Períodos do dia:",
]

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{35}")


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------


def check_input(message: str) -> GuardResult:
    """Verifica se a mensagem de entrada contém ameaças conhecidas.

    Retorna no primeiro match encontrado (injeção > exfiltração > PII).
    Execução: O(len(message)) sem I/O externo.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            return GuardResult(
                passed=False,
                threat_type="prompt_injection",
                reason=f"Padrão de injeção detectado: '{pattern.pattern[:60]}'",
            )

    for pattern in _EXFILTRATION_PATTERNS:
        if pattern.search(message):
            return GuardResult(
                passed=False,
                threat_type="exfiltration",
                reason=f"Padrão de exfiltração detectado: '{pattern.pattern[:60]}'",
            )

    for pattern in _PII_PATTERNS:
        if pattern.search(message):
            return GuardResult(
                passed=False,
                threat_type="pii",
                reason="Dado pessoal (CPF, e-mail ou telefone) detectado na mensagem.",
            )

    return GuardResult(passed=True, threat_type=None, reason="")


def check_output(response: str) -> GuardResult:
    """Verifica se a resposta do LLM vaza informações sensíveis.

    Degrada graciosamente: retorna GuardResult com passed=False para log/
    sanitização, mas não lança exceção. A camada de serviço decide a ação.
    """
    for literal in _SYSTEM_LEAK_LITERALS:
        if literal in response:
            return GuardResult(
                passed=False,
                threat_type="system_leak",
                reason=f"Fragmento do system prompt detectado na resposta: '{literal}'",
            )

    if _API_KEY_PATTERN.search(response):
        return GuardResult(
            passed=False,
            threat_type="api_key_leak",
            reason="Padrão de API key Google detectado na resposta.",
        )

    return GuardResult(passed=True, threat_type=None, reason="")
