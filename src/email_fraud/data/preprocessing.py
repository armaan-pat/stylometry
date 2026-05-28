"""Email preprocessing pipeline.

clean_email_raw(raw, config) — full RFC-2822 string → cleaned body (used by prepare_data.py)
preprocess(body, config) — already-extracted body → cleaned body
All patterns compiled at import time.
"""

from __future__ import annotations

import email as _email_stdlib
import email.message
import re
from typing import Optional

from email_fraud.config import PreprocessingConfig

# ftfy fixes garbled Unicode artifacts common in Enron data (e.g. â€™ → ').
try:
    import ftfy as _ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# "--- Original Message ---" and similar Outlook separators.
_RE_REPLY_CHAIN = re.compile(
    r"(-{3,}\s*Original Message\s*-{3,}.*)",
    re.IGNORECASE | re.DOTALL,
)
_RE_FORWARD_BLOCK = re.compile(
    r"(-{3,}\s*Forwarded by.*?-{3,})",
    re.IGNORECASE | re.DOTALL,
)

# Lotus Notes reply header patterns (the client used throughout Enron).
_LOTUS_NAME = r"(?:[A-Z][a-zA-Z]*\.?[ \t]+){1,5}[A-Z][a-zA-Z]+(?:[/@]\w+)*"

# "From:  user@company  01/23/2001"
_RE_LOTUS_REPLY = re.compile(
    r"\n[ \t]*From:[ \t]+\S[^\n]*@[^\n]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.IGNORECASE | re.DOTALL,
)

# "John Smith\n01/23/2001 10:30 AM"
_RE_LOTUS_NAMELINE = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*",
    re.DOTALL,
)

# "John Smith on 01/23/2001 ..."
_RE_LOTUS_ON = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.DOTALL | re.IGNORECASE,
)

# '"John Smith" <john@co.com>\n01/23/2001 10:30 AM'
_RE_QUOTED_NAME_REPLY = re.compile(
    r'\n[ \t]*"[^"\n]+"[ \t]+<[^>\n]+>\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*',
    re.DOTALL,
)

# "John Smith <john@co.com> on 01/23/2001 ..."
_RE_EXT_REPLY = re.compile(
    r'\n[ \t]*(?:"[^"\n]+"|[A-Za-z][A-Za-z\s]+?)[ \t]+<[^>\n]+>[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)

# "user@company.com on 01/23/2001 ..."
_RE_BARE_REPLY = re.compile(
    r'\n[ \t]*\S+@\S+[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)

# Inline forwarded headers ("From:", "To:", etc.) that appear in the body when
# a message is forwarded inline — distinct from the RFC-2822 envelope headers.
_RE_INLINE_HEADER = re.compile(
    r"^\s*(From|To|Sent|Date|Subject|Cc|Bcc)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

_RE_GMAIL_REPLY = re.compile(
    r"\nOn\s+\w{3},?.+?wrote:.*",
    re.IGNORECASE | re.DOTALL,
)

_RE_QUOTED_LINE = re.compile(r"^>.*$", re.MULTILINE)

# Signature separators: "-- " (RFC 3676), "---", "___".
_RE_SIG_SEPARATOR = re.compile(
    r"(\n-- \n|\n--\n|\n_{3,}\n|\n-{3,}\n).*",
    re.DOTALL,
)

# Order matters: broad patterns run first so fine-grained ones don't partially match.
_REPLY_STRIP_PATTERNS: list[re.Pattern] = [
    _RE_FORWARD_BLOCK,
    _RE_REPLY_CHAIN,
    _RE_GMAIL_REPLY,
    _RE_LOTUS_REPLY,
    _RE_LOTUS_NAMELINE,
    _RE_LOTUS_ON,
    _RE_QUOTED_NAME_REPLY,
    _RE_EXT_REPLY,
    _RE_BARE_REPLY,
    _RE_INLINE_HEADER,
    _RE_QUOTED_LINE,
]

# URLs first so "http://user@host" doesn't partially match the email pattern.
_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),                     "[URL]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),   "[EMAIL]"),
    # Enron internal routing addresses like "J.Smith/HOU/ENRON@ENRON"
    (re.compile(r"\b[A-Za-z][A-Za-z\s]+/[A-Z]+/[A-Z]+@[A-Z]+\b"),           "[EMAIL]"),
    (
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{2,4}\b"
            r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
            r"|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b",
            re.IGNORECASE,
        ),
        "[DATE]",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b", re.IGNORECASE), "[TIME]"),
    (re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b|\b\d{10}\b"),             "[PHONE]"),
    (re.compile(r"\b[A-Z]{1,4}[-_]?\d{5,}\b|\b\d{6,}\b"),                    "[ORDER_ID]"),
    (re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),         "[LOCATION]"),
]

_RE_PLACEHOLDER = re.compile(
    r"\b(EMAIL|URL|DATE|TIME|PHONE|ORDER_ID|LOCATION)\b"
)

_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_WHITESPACE  = re.compile(r"[ \t]+")


def _decode_payload(part: email.message.Message) -> Optional[str]:
    """Decode a MIME part to str. Falls back to latin-1 (can decode any byte sequence)."""
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        return None
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("latin-1", errors="replace")


def _extract_body(msg: email.message.Message) -> Optional[str]:
    """Extract plain text, preferring text/plain over HTML.
    Some Enron emails only have HTML parts, so we strip those as a fallback.
    """
    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            return _decode_payload(part)
        if part.get_content_type() == "text/html" and html_part is None:
            html_part = part  # stash first HTML part in case no plain text

    if html_part is not None:
        raw_html = _decode_payload(html_part)
        if raw_html:
            if _BS4_AVAILABLE:
                return _BeautifulSoup(raw_html, "html.parser").get_text(separator="\n")
            return re.sub(r"<[^>]+>", " ", raw_html)
    return None


def _isolate_newest_message(body: str) -> str:
    """Strip all reply/forward blocks by applying _REPLY_STRIP_PATTERNS in sequence."""
    for pattern in _REPLY_STRIP_PATTERNS:
        body = pattern.sub("", body)
    return body


def _strip_signatures(body: str) -> str:
    return _RE_SIG_SEPARATOR.sub("", body)


def _normalize_whitespace(text: str) -> str:
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _normalize_high_variance(text: str) -> str:
    """Replace URLs, emails, dates, phones, etc. with placeholder tokens."""
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def _is_usable(body: str, min_chars: int) -> bool:
    """Return True if the body meets length and placeholder-ratio thresholds.

    If >50% of tokens are placeholders after entity masking, the email's
    content has been mostly stripped and is no longer useful for stylometry.
    """
    stripped = body.strip()
    if len(stripped) < min_chars:
        return False
    words = stripped.split()
    if not words:
        return False
    token_ratio = len(_RE_PLACEHOLDER.findall(stripped)) / len(words)
    return token_ratio < 0.5


def clean_email_raw(raw: str, config: PreprocessingConfig) -> Optional[str]:
    """Parse a raw RFC-2822 string and run the preprocessing pipeline on it."""
    try:
        msg = _email_stdlib.message_from_string(raw)
    except Exception:
        return None

    body = _extract_body(msg)
    if body is None:
        return None

    if config.fix_encoding and _FTFY_AVAILABLE:
        body = _ftfy.fix_text(body)

    return preprocess(body, config)


def preprocess(text: str, config: PreprocessingConfig) -> Optional[str]:
    """Apply the configured cleaning steps to an already-extracted email body.

    Steps (each gated by a config flag): strip_quoted, strip_signatures,
    entity_masking. Whitespace normalization and truncation always run.
    Returns None if the result is too short or over-normalized.
    """
    if config.strip_quoted:
        text = _isolate_newest_message(text)

    if config.strip_signatures:
        text = _strip_signatures(text)

    if config.entity_masking:
        text = _normalize_high_variance(text)

    text = _normalize_whitespace(text)

    if config.max_body_chars and len(text) > config.max_body_chars:
        text = text[:config.max_body_chars]

    if not _is_usable(text, config.min_body_chars):
        return None

    return text


def preprocess_batch(texts: list[str], config: PreprocessingConfig) -> list[str]:
    """Apply preprocess() to a list of bodies, silently dropping None results."""
    results = []
    for t in texts:
        out = preprocess(t, config)
        if out is not None:
            results.append(out)
    return results
